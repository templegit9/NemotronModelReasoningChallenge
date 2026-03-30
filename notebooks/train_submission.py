"""
Kaggle Training Notebook — NVIDIA Nemotron Model Reasoning Challenge
====================================================================

SETUP BEFORE RUNNING:
1. Add Input: Competition data (nvidia-nemotron-model-reasoning-challenge)
2. Add Input: Model (metric/nemotron-3-nano-30b-a3b-bf16)
3. Add Input: Your uploaded train_formatted.jsonl as a Dataset
4. Add Input: Kernel data source "ryanholbrook/nvidia-utility-script"
5. Session options: GPU (RTX Pro 6000), Internet OFF
6. Run all cells

SUBMISSION: After run completes, click "Submit to competition" on the right sidebar.
"""

# ============================================================
# CELL 1: Setup and Configuration
# ============================================================

import os
import json
import glob
import time
import random

import torch
import numpy as np

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ---- Configuration ----
LORA_RANK = 32          # Max allowed by competition
LORA_ALPHA = 16         # From starter notebook
LORA_DROPOUT = 0.05     # From starter notebook
LEARNING_RATE = 2e-5    # Conservative starting point
NUM_EPOCHS = 2          # Start conservative
BATCH_SIZE = 2          # Per-device batch size
GRAD_ACCUM_STEPS = 8    # Effective batch size = 2 * 8 = 16
MAX_SEQ_LENGTH = 2048   # Sufficient for most prompts + CoT + answer
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 0.01
OUTPUT_DIR = "/kaggle/working"
LOGGING_STEPS = 50

print(f"Configuration:")
print(f"  LoRA rank={LORA_RANK}, alpha={LORA_ALPHA}, dropout={LORA_DROPOUT}")
print(f"  LR={LEARNING_RATE}, epochs={NUM_EPOCHS}, batch={BATCH_SIZE}×{GRAD_ACCUM_STEPS}")
print(f"  Max seq length={MAX_SEQ_LENGTH}")

# ============================================================
# CELL 2: Find model and data paths
# ============================================================

# Find model path (added as Kaggle Model input)
model_configs = glob.glob("/kaggle/input/**/config.json", recursive=True)
MODEL_PATH = None
for cfg in model_configs:
    cfg_dir = os.path.dirname(cfg)
    if os.path.exists(os.path.join(cfg_dir, "tokenizer_config.json")):
        MODEL_PATH = cfg_dir
        break

if not MODEL_PATH:
    raise FileNotFoundError(f"Model not found. config.json files: {model_configs}")

print(f"Model path: {MODEL_PATH}")

# Find training data (uploaded as dataset)
TRAIN_DATA_PATH = None
for pattern in [
    "/kaggle/input/**/train_formatted.jsonl",
    "/kaggle/input/*/train_formatted.jsonl",
]:
    matches = glob.glob(pattern, recursive=True)
    if matches:
        TRAIN_DATA_PATH = matches[0]
        break

if not TRAIN_DATA_PATH:
    # Fallback: use the competition train.csv and format on-the-fly
    print("WARNING: train_formatted.jsonl not found. Will format from train.csv.")
    TRAIN_DATA_PATH = None

print(f"Training data: {TRAIN_DATA_PATH}")

# ============================================================
# CELL 3: Setup CUTLASS (required for Mamba layers)
# ============================================================

import site
import subprocess

# Step 1: Fix ptxas binary permissions (required for Triton/Mamba kernels)
# This is exactly what the official metric notebook does
ptxas_binaries = glob.glob("/kaggle/usr/lib/notebooks/*/triton/backends/nvidia/bin/ptxas*")
for ptxas_path in ptxas_binaries:
    subprocess.run(f"chmod +x {ptxas_path}", shell=True)
    print(f"Set execute permission: {ptxas_path}")

# Step 2: Find and add CUTLASS python packages path
cutlass_search_patterns = [
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia-utility-script/nvidia_cutlass_dsl/python_packages/",
    "/kaggle/usr/lib/notebooks/ryanholbrook/nvidia_utility_script/nvidia_cutlass_dsl/python_packages/",
    "/kaggle/usr/lib/notebooks/*/nvidia_cutlass_dsl/python_packages/",
    "/kaggle/input/**/nvidia_cutlass_dsl/python_packages/",
]

cutlass_found = False
for pattern in cutlass_search_patterns:
    matches = glob.glob(pattern, recursive=True)
    if matches:
        site.addsitedir(matches[0])
        print(f"Added CUTLASS path: {matches[0]}")
        cutlass_found = True
        break

if not cutlass_found:
    print("WARNING: CUTLASS path not found. Mamba layers may not work.")

# ============================================================
# CELL 4: Load Model and Tokenizer
# ============================================================

import mamba_ssm  # Must import after CUTLASS setup
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
print(f"Model loaded. Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'N/A'}")

# ============================================================
# CELL 5: Apply LoRA Adapter
# ============================================================

lora_config = LoraConfig(
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    target_modules=r".*\.(in_proj|out_proj|up_proj|down_proj)$",
    lora_dropout=LORA_DROPOUT,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()
model.enable_input_require_grads()

# ============================================================
# CELL 6: Prepare Dataset
# ============================================================

from torch.utils.data import Dataset, DataLoader

BOXED_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'


class ReasoningDataset(Dataset):
    """Dataset that tokenizes chat messages and masks loss on non-assistant tokens."""

    def __init__(self, data_path, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        if data_path and data_path.endswith('.jsonl'):
            # Load pre-formatted JSONL
            with open(data_path) as f:
                for line in f:
                    row = json.loads(line)
                    self.examples.append(row['messages'])
        else:
            # Fallback: load from CSV and format on-the-fly
            import csv
            csv_path = glob.glob("/kaggle/input/**/train.csv", recursive=True)
            if csv_path:
                with open(csv_path[0]) as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        messages = [
                            {'role': 'system', 'content': ''},
                            {'role': 'user', 'content': row['prompt'] + BOXED_SUFFIX},
                            {'role': 'assistant', 'content': f"<think>\n{row['answer']}\n</think>\n\\boxed{{{row['answer']}}}"},
                        ]
                        self.examples.append(messages)

        print(f"Loaded {len(self.examples)} training examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        messages = self.examples[idx]

        # Tokenize the full conversation
        full_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=True,
        )

        full_tokens = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = full_tokens["input_ids"].squeeze(0)
        attention_mask = full_tokens["attention_mask"].squeeze(0)

        # Create labels: mask everything before the assistant response
        # Find where the assistant content starts (after "<|im_start|>assistant\n")
        labels = input_ids.clone()

        # Tokenize just the system + user part to find where assistant starts
        non_assistant_messages = messages[:2]  # system + user only
        non_assistant_text = self.tokenizer.apply_chat_template(
            non_assistant_messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        non_assistant_tokens = self.tokenizer(
            non_assistant_text,
            truncation=True,
            max_length=self.max_length,
        )
        prompt_length = len(non_assistant_tokens["input_ids"])

        # Mask prompt tokens (set to -100 so they don't contribute to loss)
        labels[:prompt_length] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def collate_fn(batch):
    """Pad batch to same length."""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids = []
    attention_mask = []
    labels = []

    for item in batch:
        pad_len = max_len - item["input_ids"].size(0)
        # Pad on the right
        input_ids.append(torch.cat([
            item["input_ids"],
            torch.full((pad_len,), tokenizer.pad_token_id, dtype=torch.long),
        ]))
        attention_mask.append(torch.cat([
            item["attention_mask"],
            torch.zeros(pad_len, dtype=torch.long),
        ]))
        labels.append(torch.cat([
            item["labels"],
            torch.full((pad_len,), -100, dtype=torch.long),
        ]))

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


dataset = ReasoningDataset(TRAIN_DATA_PATH, tokenizer, MAX_SEQ_LENGTH)
dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn,
    num_workers=2,
    pin_memory=True,
)

print(f"DataLoader: {len(dataloader)} batches per epoch")

# Quick check: show a sample
sample = dataset[0]
print(f"Sample input_ids shape: {sample['input_ids'].shape}")
print(f"Sample labels masked: {(sample['labels'] == -100).sum().item()} / {sample['labels'].size(0)}")

# ============================================================
# CELL 7: Training Loop
# ============================================================

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Optimizer
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
    betas=(0.9, 0.999),
)

# Scheduler
total_steps = len(dataloader) * NUM_EPOCHS // GRAD_ACCUM_STEPS
warmup_steps = int(total_steps * WARMUP_RATIO)

scheduler = CosineAnnealingLR(optimizer, T_max=total_steps, eta_min=LEARNING_RATE * 0.1)

print(f"Total optimization steps: {total_steps}")
print(f"Warmup steps: {warmup_steps}")
print(f"Training for {NUM_EPOCHS} epochs...")

model.train()
global_step = 0
start_time = time.time()

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    epoch_steps = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        # Forward pass
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss / GRAD_ACCUM_STEPS

        # Backward pass
        loss.backward()

        epoch_loss += outputs.loss.item()
        epoch_steps += 1

        # Gradient accumulation step
        if (batch_idx + 1) % GRAD_ACCUM_STEPS == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Warmup: linear warmup for first N steps
            if global_step <= warmup_steps:
                warmup_lr = LEARNING_RATE * (global_step / warmup_steps)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr

        # Logging
        if (batch_idx + 1) % LOGGING_STEPS == 0:
            avg_loss = epoch_loss / epoch_steps
            elapsed = time.time() - start_time
            lr_current = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch+1}/{NUM_EPOCHS} | "
                f"Step {batch_idx+1}/{len(dataloader)} | "
                f"Loss: {avg_loss:.4f} | "
                f"LR: {lr_current:.2e} | "
                f"Time: {elapsed/60:.1f}min"
            )

    avg_epoch_loss = epoch_loss / epoch_steps if epoch_steps > 0 else 0
    elapsed = time.time() - start_time
    print(f"\n=== Epoch {epoch+1}/{NUM_EPOCHS} done | Avg Loss: {avg_epoch_loss:.4f} | Time: {elapsed/60:.1f}min ===\n")

# Handle remaining gradients
if (batch_idx + 1) % GRAD_ACCUM_STEPS != 0:
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

total_time = time.time() - start_time
print(f"\nTraining complete! Total time: {total_time/60:.1f} minutes")

# ============================================================
# CELL 8: Save Adapter
# ============================================================

print(f"Saving adapter to {OUTPUT_DIR}...")
model.save_pretrained(OUTPUT_DIR)
print(f"Saved. Files: {os.listdir(OUTPUT_DIR)}")

# ============================================================
# CELL 9: Package Submission
# ============================================================

import subprocess

os.chdir(OUTPUT_DIR)
subprocess.run("zip -r submission.zip .", shell=True, check=True)
print(f"Submission packaged: {os.path.getsize('submission.zip') / 1024 / 1024:.1f} MB")
print("Done! Click 'Submit to competition' on the right sidebar.")
