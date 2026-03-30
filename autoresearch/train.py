"""
Training script for autoresearch experiments.
THIS IS THE FILE THE AGENT MODIFIES.

Trains a LoRA adapter on Qwen2.5-3B-Instruct using our formatted reasoning data.
Designed to run on RTX 5070 Ti (16GB VRAM) in under 15 minutes.
"""

import json
import os
import random
import time

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from prepare_data import load_train

# ============================================================
# Configuration — MODIFY THESE
# ============================================================
SEED = 42
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
OUTPUT_DIR = "/tmp/autoresearch_adapter"

# LoRA config
LORA_RANK = 32
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "up_proj", "down_proj"]

# Training config
LEARNING_RATE = 2e-5
NUM_EPOCHS = 1
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
MAX_SEQ_LENGTH = 1024
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01

# Data config
CATEGORY_WEIGHTS = {
    "NUMBER_SYSTEM": 1.0,
    "UNIT_CONVERSION": 1.0,
    "PHYSICS": 1.0,
    "TEXT_ENCRYPTION": 1.0,
    "BIT_MANIPULATION": 1.0,
    "SYMBOL_TRANSFORM": 1.0,
}

# ============================================================
# Setup
# ============================================================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Dataset
# ============================================================
class ReasoningDataset(Dataset):
    def __init__(self, examples, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        messages = example['messages']

        # Tokenize full conversation
        full_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False,
        )
        full_tokens = self.tokenizer(
            full_text, truncation=True, max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = full_tokens["input_ids"].squeeze(0)
        attention_mask = full_tokens["attention_mask"].squeeze(0)

        # Create labels: mask everything before assistant response
        labels = input_ids.clone()
        non_assistant_text = self.tokenizer.apply_chat_template(
            messages[:2], tokenize=False, add_generation_prompt=True,
        )
        non_assistant_tokens = self.tokenizer(
            non_assistant_text, truncation=True, max_length=self.max_length,
        )
        prompt_length = len(non_assistant_tokens["input_ids"])
        labels[:prompt_length] = -100

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


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
    """Resample examples according to category weights."""
    weighted = []
    for ex in examples:
        cat = ex['category']
        w = weights.get(cat, 1.0)
        # Repeat proportionally (floor + probabilistic for fractional)
        repeats = int(w)
        if random.random() < (w - repeats):
            repeats += 1
        weighted.extend([ex] * repeats)
    random.shuffle(weighted)
    return weighted


# ============================================================
# Main Training
# ============================================================
def main():
    start_time = time.time()

    # Load data
    train_data = load_train()
    train_data = apply_category_weights(train_data, CATEGORY_WEIGHTS)
    print(f"Training examples (after weighting): {len(train_data)}")

    # Load model
    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )

    # Apply LoRA
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
    model.enable_input_require_grads()

    # Dataset
    dataset = ReasoningDataset(train_data, tokenizer, MAX_SEQ_LENGTH)
    dataloader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=2, pin_memory=True,
    )
    print(f"DataLoader: {len(dataloader)} batches per epoch")

    # Optimizer
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(dataloader) * NUM_EPOCHS // GRAD_ACCUM_STEPS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=LEARNING_RATE * 0.1)

    # Training loop
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

    # Handle remaining gradients
    if epoch_steps > 0 and (batch_idx + 1) % GRAD_ACCUM_STEPS != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    total_time = time.time() - start_time
    print(f"Training complete! Time: {total_time/60:.1f} minutes")

    # Save adapter
    model.save_pretrained(OUTPUT_DIR)
    print(f"Adapter saved to {OUTPUT_DIR}")

    return {"loss": avg_epoch_loss, "time_minutes": total_time / 60}


if __name__ == "__main__":
    main()
