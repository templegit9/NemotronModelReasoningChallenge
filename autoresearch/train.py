"""
Training script for autoresearch experiments — Nemotron-3-Nano-30B-A3B on RunPod A100 80GB.
THIS IS THE FILE THE AGENT MODIFIES.

Trains a LoRA adapter on Nemotron-3-Nano-30B-A3B using our formatted reasoning data.
Designed to run on A100 80GB in under 2 hours.
"""

import json
import os
import random
import time

import mamba_ssm  # Must import before transformers to register Mamba kernels
import numpy as np
import torch
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from prepare_data import load_train
except ImportError:
    print("WARNING: Could not import prepare_data. Make sure prepare_data.py exists.")
    def load_train():
        return []

SEED = 42
MODEL_PATH = os.environ.get("MODEL_PATH", "/workspace/nemotron")
OUTPUT_DIR = os.environ.get("ADAPTER_PATH", "/workspace/adapter")

LORA_RANK = 32
LORA_ALPHA = 64
LORA_DROPOUT = 0.05
# Regex pattern matching Mamba hybrid layers (in_proj/out_proj) and MLP layers (up_proj/down_proj)
LORA_TARGET_MODULES = r".*\.(in_proj|out_proj|up_proj|down_proj)$"

LEARNING_RATE = 2e-5
NUM_EPOCHS = 1
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 8
MAX_SEQ_LENGTH = 2048
MAX_TRAINING_STEPS = 150
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01

CATEGORY_WEIGHTS = {
    "NUMBER_SYSTEM": 1.0,
    "UNIT_CONVERSION": 1.0,
    "PHYSICS": 1.5,
    "TEXT_ENCRYPTION": 1.5,
    "BIT_MANIPULATION": 1.5,
    "SYMBOL_TRANSFORM": 1.5,
}

# Proven technique: prepend short reasoning hints to hard-category assistant messages
REASONING_STARTERS = {
    "TEXT_ENCRYPTION": "I need to analyze the encryption pattern and decode step by step.",
    "BIT_MANIPULATION": "Let me think about the bit operations in binary representation.",
    "SYMBOL_TRANSFORM": "I will trace through the symbol transformation rules carefully.",
    "PHYSICS": "I need to identify the formula, extract the values, and compute step by step.",
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)


def has_boxed_answer(text):
    return r'\boxed{' in text


def filter_valid_examples(examples):
    valid = []
    for ex in examples:
        messages = ex.get('messages', [])
        if len(messages) >= 2:
            assistant_msg = messages[-1].get('content', '')
            if has_boxed_answer(assistant_msg):
                valid.append(ex)
    print(f"Filtered: {len(examples)} -> {len(valid)} valid examples (with \\boxed{{}} answers)")
    return valid


def inject_reasoning_starter(example):
    category = example.get('category', 'UNKNOWN')
    if category not in REASONING_STARTERS:
        return example
    messages = example.get('messages', [])
    if len(messages) < 2:
        return example
    example = json.loads(json.dumps(example))
    starter = REASONING_STARTERS[category]
    assistant_content = example['messages'][-1]['content']
    example['messages'][-1]['content'] = starter + "\n" + assistant_content
    return example


class ReasoningDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        print(f"Pre-tokenizing {len(examples)} examples...")
        self.data = []
        for example in examples:
            try:
                messages = example['messages']
                full_text = tokenizer.apply_chat_template(messages, tokenize=False)
                full_tokens = tokenizer(
                    full_text, truncation=True, max_length=max_length, return_tensors="pt"
                )
                input_ids = full_tokens["input_ids"].squeeze(0)
                attention_mask = full_tokens["attention_mask"].squeeze(0)
                labels = input_ids.clone()
                prompt_messages = [m for m in messages if m['role'] != 'assistant']
                if prompt_messages:
                    prompt_text = tokenizer.apply_chat_template(
                        prompt_messages + [{"role": "assistant", "content": ""}],
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    prompt_text = ""
                prompt_len = len(tokenizer(prompt_text, truncation=True, max_length=max_length)["input_ids"])
                labels[:prompt_len] = -100
                if (labels != -100).any():
                    self.data.append({"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels})
            except Exception as e:
                print(f"Skipping example due to error: {e}")
                continue
        print(f"Pre-tokenization complete. Valid examples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn(batch, pad_token_id):
    max_len = max(item["input_ids"].size(0) for item in batch)
    input_ids, attention_mask, labels_list = [], [], []
    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        input_ids.append(torch.cat([item["input_ids"], torch.full((pad_len,), pad_token_id, dtype=torch.long)]))
        attention_mask.append(torch.cat([item["attention_mask"], torch.zeros(pad_len, dtype=torch.long)]))
        labels_list.append(torch.cat([item["labels"], torch.full((pad_len,), -100, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels_list),
    }


def apply_category_weights(examples, weights):
    weighted = []
    for ex in examples:
        cat = ex.get('category', 'UNKNOWN')
        w = weights.get(cat, 1.0)
        repeats = int(w)
        if random.random() < (w - repeats):
            repeats += 1
        weighted.extend([ex] * repeats)
    random.shuffle(weighted)
    return weighted


def main():
    start_time = time.time()

    train_data = load_train()
    if not train_data:
        print("ERROR: No training data loaded.")
        return {"loss": float('inf'), "time_minutes": 0}

    print(f"Raw training examples: {len(train_data)}")
    train_data = filter_valid_examples(train_data)
    train_data = [inject_reasoning_starter(ex) for ex in train_data]
    print(f"Injected reasoning starters into hard-category examples")
    train_data = apply_category_weights(train_data, CATEGORY_WEIGHTS)
    print(f"Training examples (after filtering and weighting): {len(train_data)}")

    print(f"Loading {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    # NOTE: Gradient checkpointing is incompatible with Mamba layers — do NOT enable it.
    model.enable_input_require_grads()

    dataset = ReasoningDataset(train_data, tokenizer, MAX_SEQ_LENGTH)
    if len(dataset) == 0:
        print("ERROR: Dataset is empty after pre-tokenization!")
        return {"loss": float('inf'), "time_minutes": 0}

    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=0, pin_memory=False,
    )
    print(f"DataLoader: {len(dataloader)} batches per epoch")

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(dataloader) * NUM_EPOCHS // GRAD_ACCUM_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=LEARNING_RATE * 0.1)

    model.train()
    global_step = 0

    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / GRAD_ACCUM_STEPS
            loss.backward()

            epoch_loss += outputs.loss.item()
            epoch_steps += 1

            if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step >= MAX_TRAINING_STEPS:
                    print(f"Reached MAX_TRAINING_STEPS={MAX_TRAINING_STEPS}, stopping early.")
                    break

                if global_step <= warmup_steps:
                    warmup_lr = LEARNING_RATE * (global_step / max(warmup_steps, 1))
                    for pg in optimizer.param_groups:
                        pg['lr'] = warmup_lr

            if (batch_idx + 1) % 100 == 0:
                avg = epoch_loss / epoch_steps
                elapsed = time.time() - start_time
                print(f"E{epoch+1} Step {batch_idx+1}/{len(dataloader)} | Loss: {avg:.4f} | {elapsed/60:.1f}min")

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch+1} done | Avg Loss: {avg_epoch_loss:.4f}")

    if epoch_steps > 0 and (batch_idx + 1) % GRAD_ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    total_time = time.time() - start_time
    print(f"Training complete! Time: {total_time/60:.1f} minutes")

    model.save_pretrained(OUTPUT_DIR)
    print(f"Adapter saved to {OUTPUT_DIR}")

    return {"loss": avg_epoch_loss, "time_minutes": total_time / 60}


if __name__ == "__main__":
    main()
