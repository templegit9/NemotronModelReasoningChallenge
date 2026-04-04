"""
Training script supporting multiple models for strategy testing.

Proxy mode (fast iteration, ~5-15 min on H100):
    python train_h100.py --model zamba2 --data data/distilled_train.jsonl
    python train_h100.py --model qwen --data data/distilled_train.jsonl

Full mode (final submission, ~8-13 hours on H100):
    python train_h100.py --model nemotron --data data/distilled_train.jsonl
"""

import argparse
import json
import os
import random
import time
from collections import Counter

import numpy as np
import torch
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------- Model Profiles ----------
MODEL_PROFILES = {
    "nemotron": {
        "path": "/workspace/nemotron",
        "hf_id": "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",    # Attention (6 layers)
            "gate_proj", "up_proj", "down_proj",         # MLP/MoE (23 layers)
            "in_proj", "out_proj",                       # Mamba (23 layers)
        ],
        "needs_mamba_import": True,
        "max_seq_length": 4096,
        "batch_size": 1,
        "grad_accum": 16,
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "lora_rank": 32,
        "lora_alpha": 64,
    },
    "zamba2": {
        "path": "Zyphra/Zamba2-2.7B-instruct",
        "hf_id": "Zyphra/Zamba2-2.7B-instruct",
        "target_modules": ["x_proj", "out_proj"],  # Mamba hybrid targets
        "needs_mamba_import": True,
        "max_seq_length": 4096,
        "batch_size": 4,
        "grad_accum": 4,
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "lora_rank": 32,
        "lora_alpha": 64,
    },
    "qwen": {
        "path": "Qwen/Qwen2.5-7B-Instruct",
        "hf_id": "Qwen/Qwen2.5-7B-Instruct",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "needs_mamba_import": False,
        "max_seq_length": 4096,
        "batch_size": 4,
        "grad_accum": 4,
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "lora_rank": 32,
        "lora_alpha": 64,
    },
    "qwen3b": {
        "path": "Qwen/Qwen2.5-3B-Instruct",
        "hf_id": "Qwen/Qwen2.5-3B-Instruct",
        "target_modules": [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        "needs_mamba_import": False,
        "max_seq_length": 4096,
        "batch_size": 8,
        "grad_accum": 2,
        "learning_rate": 5e-5,
        "num_epochs": 3,
        "lora_rank": 32,
        "lora_alpha": 64,
    },
}

# ---------- Config ----------
SEED = 42
LORA_DROPOUT = 0.05
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
LOG_EVERY = 25

# Category oversampling weights
CATEGORY_WEIGHTS = {
    "NUMBER_SYSTEM": 1.0,
    "UNIT_CONVERSION": 1.0,
    "PHYSICS": 1.0,
    "TEXT_ENCRYPTION": 1.0,
    "BIT_MANIPULATION": 2.5,
    "SYMBOL_TRANSFORM": 2.5,
}

# ---------- Reproducibility ----------
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


class ReasoningDataset(Dataset):
    """Pre-tokenized dataset with loss masking on non-assistant tokens."""

    def __init__(self, examples, tokenizer, max_length):
        print(f"Pre-tokenizing {len(examples)} examples (max_length={max_length})...")
        self.data = []
        skipped = 0
        lengths = []

        for example in examples:
            try:
                messages = example['messages']

                # Try with enable_thinking, fall back without (not all models support it)
                try:
                    full_text = tokenizer.apply_chat_template(
                        messages, tokenize=False, enable_thinking=True
                    )
                except Exception:
                    full_text = tokenizer.apply_chat_template(
                        messages, tokenize=False
                    )

                full_tokens = tokenizer(
                    full_text, truncation=True, max_length=max_length, return_tensors="pt"
                )
                input_ids = full_tokens["input_ids"].squeeze(0)
                attention_mask = full_tokens["attention_mask"].squeeze(0)

                # Create labels: mask everything before assistant response
                labels = input_ids.clone()
                prompt_messages = [m for m in messages if m['role'] != 'assistant']
                try:
                    prompt_text = tokenizer.apply_chat_template(
                        prompt_messages + [{"role": "assistant", "content": ""}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=True,
                    )
                except Exception:
                    prompt_text = tokenizer.apply_chat_template(
                        prompt_messages + [{"role": "assistant", "content": ""}],
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                prompt_len = len(tokenizer(
                    prompt_text, truncation=True, max_length=max_length
                )["input_ids"])
                labels[:prompt_len] = -100

                if (labels != -100).any():
                    self.data.append({
                        "input_ids": input_ids,
                        "attention_mask": attention_mask,
                        "labels": labels,
                    })
                    lengths.append(len(input_ids))
                else:
                    skipped += 1
            except Exception as e:
                skipped += 1
                continue

        if lengths:
            avg_len = sum(lengths) / len(lengths)
            print(f"Tokenization done. Valid: {len(self.data)}, Skipped: {skipped}")
            print(f"Avg sequence length: {avg_len:.0f}, Max: {max(lengths)}, Min: {min(lengths)}")
        else:
            print(f"WARNING: No valid examples! Skipped: {skipped}")

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


def load_data(data_path):
    examples = []
    with open(data_path) as f:
        for line in f:
            obj = json.loads(line)
            if 'messages' in obj and len(obj['messages']) >= 2:
                assistant_msg = obj['messages'][-1].get('content', '')
                if '\\boxed{' in assistant_msg:
                    examples.append(obj)
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='nemotron',
                        choices=list(MODEL_PROFILES.keys()),
                        help='Model to train (nemotron, zamba2, qwen, qwen3b)')
    parser.add_argument('--model-path', default='',
                        help='Override model path (local or HF ID)')
    parser.add_argument('--data', required=True, help='Path to training JSONL')
    parser.add_argument('--output', default='', help='Output directory for adapter')
    parser.add_argument('--lr', type=float, default=0, help='Override learning rate')
    parser.add_argument('--epochs', type=int, default=0, help='Override epoch count')
    parser.add_argument('--lora-rank', type=int, default=0, help='Override LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=0, help='Override LoRA alpha')
    parser.add_argument('--max-seq-len', type=int, default=0, help='Override max sequence length')
    parser.add_argument('--cat-weight-hard', type=float, default=0,
                        help='Override category weight for hard categories')
    args = parser.parse_args()

    profile = MODEL_PROFILES[args.model]

    # Apply overrides
    model_path = args.model_path or profile["path"]
    output_dir = args.output or f"/workspace/adapter_{args.model}"
    lr = args.lr or profile["learning_rate"]
    epochs = args.epochs or profile["num_epochs"]
    lora_rank = args.lora_rank or profile["lora_rank"]
    lora_alpha = args.lora_alpha or profile["lora_alpha"]
    max_seq = args.max_seq_len or profile["max_seq_length"]
    batch_size = profile["batch_size"]
    grad_accum = profile["grad_accum"]
    target_modules = profile["target_modules"]

    if args.cat_weight_hard > 0:
        CATEGORY_WEIGHTS["BIT_MANIPULATION"] = args.cat_weight_hard
        CATEGORY_WEIGHTS["SYMBOL_TRANSFORM"] = args.cat_weight_hard

    os.makedirs(output_dir, exist_ok=True)

    # Import mamba_ssm if needed (must be before model load)
    if profile["needs_mamba_import"]:
        try:
            import mamba_ssm
            print(f"mamba_ssm loaded: {mamba_ssm.__version__}")
        except ImportError:
            print("WARNING: mamba_ssm not available. May fail for Mamba models.")

    start_time = time.time()

    # Load data
    print(f"Loading data from {args.data}...")
    train_data = load_data(args.data)
    print(f"Loaded {len(train_data)} valid examples")

    if not train_data:
        print("ERROR: No training data loaded!")
        return

    cats = Counter(ex.get('category', 'UNKNOWN') for ex in train_data)
    print("Category distribution:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")

    train_data = apply_category_weights(train_data, CATEGORY_WEIGHTS)
    print(f"After oversampling: {len(train_data)} examples")

    # Load model
    print(f"\nLoading model: {args.model} from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.enable_input_require_grads()

    # Prepare dataset
    dataset = ReasoningDataset(train_data, tokenizer, max_seq)
    if len(dataset) == 0:
        print("ERROR: Dataset is empty!")
        return

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        num_workers=0, pin_memory=False,
    )

    # Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    total_steps = len(dataloader) * epochs // grad_accum
    scheduler = CosineAnnealingLR(optimizer, T_max=max(total_steps, 1), eta_min=lr * 0.1)
    warmup_steps = int(total_steps * WARMUP_RATIO)

    print(f"\n{'='*60}")
    print(f"Training: {args.model}")
    print(f"  Model: {model_path}")
    print(f"  Data: {args.data} ({len(train_data)} examples)")
    print(f"  Output: {output_dir}")
    print(f"  LoRA: rank={lora_rank}, alpha={lora_alpha}, targets={target_modules}")
    print(f"  LR: {lr}, Epochs: {epochs}, Batch: {batch_size}x{grad_accum}={batch_size*grad_accum}")
    print(f"  Max seq: {max_seq}, Total steps: {total_steps}, Warmup: {warmup_steps}")
    print(f"{'='*60}\n")

    # Training loop
    model.train()
    global_step = 0
    best_loss = float('inf')

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / grad_accum
            loss.backward()

            epoch_loss += outputs.loss.item()
            epoch_steps += 1

            if (batch_idx + 1) % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                if global_step <= warmup_steps:
                    warmup_lr = lr * (global_step / max(warmup_steps, 1))
                    for pg in optimizer.param_groups:
                        pg['lr'] = warmup_lr
                else:
                    scheduler.step()

            if (batch_idx + 1) % LOG_EVERY == 0:
                avg = epoch_loss / epoch_steps
                elapsed = time.time() - start_time
                cur_lr = optimizer.param_groups[0]['lr']
                print(f"E{epoch+1} Step {batch_idx+1}/{len(dataloader)} | "
                      f"Loss: {avg:.4f} | LR: {cur_lr:.2e} | {elapsed/60:.1f}min")

        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        elapsed = time.time() - start_time
        print(f"\n=== Epoch {epoch+1}/{epochs} | Loss: {avg_epoch_loss:.4f} | {elapsed/60:.1f}min ===\n")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            model.save_pretrained(output_dir)
            print(f"Saved best adapter (loss={best_loss:.4f}) to {output_dir}")

    # Final save
    if epoch_steps > 0 and (batch_idx + 1) % grad_accum != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=MAX_GRAD_NORM)
        optimizer.step()
        optimizer.zero_grad()

    total_time = time.time() - start_time
    model.save_pretrained(output_dir)
    print(f"\nTraining complete! {total_time/60:.1f} minutes")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Adapter: {output_dir}")


if __name__ == "__main__":
    main()
