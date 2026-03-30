# Research Program: Nemotron Reasoning Challenge

## Objective
Maximize accuracy on 6 types of "Alice's Wonderland" reasoning puzzles by finding the best LoRA fine-tuning configuration. We train a proxy model (Qwen2.5-3B) locally for fast iteration, then transfer the best configuration to the real Nemotron model on Kaggle.

## Metric
- **Primary**: Overall validation accuracy (correct answers / total)
- **Secondary**: Per-category accuracy breakdown (6 categories)
- **Categories**: NUMBER_SYSTEM, UNIT_CONVERSION, PHYSICS, TEXT_ENCRYPTION, BIT_MANIPULATION, SYMBOL_TRANSFORM

## Current Best
- Check `results.tsv` for the best experiment so far
- Leaderboard top score: 0.82 (82%)

## What You Can Modify
You may ONLY modify `train.py`. Changes you can make:

### 1. Chain-of-Thought Format
- Reasoning style (step-by-step, brief, detailed)
- Whether to include CoT for easy vs hard categories
- Think tag format and content

### 2. Training Data Mix
- Category sampling weights (oversample hard categories?)
- Data filtering (remove low-quality CoT examples?)
- Sequence length limits

### 3. Hyperparameters
- Learning rate (try: 1e-5 to 5e-4)
- Number of epochs (try: 1-5)
- Batch size / gradient accumulation
- LoRA rank (try: 8, 16, 32)
- LoRA alpha (try: 16, 32, 64)
- LoRA dropout (try: 0, 0.05, 0.1)
- LoRA target modules (add gate_proj, q_proj, k_proj, v_proj?)
- Weight decay
- Warmup ratio
- Max sequence length (try: 512, 1024, 2048)

### 4. Training Techniques
- Loss weighting per category
- Curriculum learning (easy first, hard later)
- Different optimizers (AdamW, SGD, Adafactor)
- Learning rate schedules (cosine, linear, constant)

## What You CANNOT Modify
- `evaluate.py` (evaluation logic is fixed)
- `prepare_data.py` (data loading is fixed)
- The proxy model choice (Qwen2.5-3B)
- The validation split

## Constraints
- Each experiment must complete within 15 minutes on RTX 5070 Ti (16GB VRAM)
- Do not exceed 14GB VRAM usage
- The adapter must use LoRA (no full fine-tuning)
- LoRA rank must be <= 32 (competition constraint)

## Strategy Tips
- Start with small changes, measure impact
- The 6 categories have different difficulty levels:
  - Easy (solvable algorithmically): NUMBER_SYSTEM, UNIT_CONVERSION, PHYSICS
  - Medium: TEXT_ENCRYPTION
  - Hard: BIT_MANIPULATION, SYMBOL_TRANSFORM
- Changes that improve hard categories are most valuable
- The real model (Nemotron) has Mamba+Transformer hybrid architecture; Qwen is pure Transformer. Data/format changes transfer better than architecture-specific hyperparams.
- Test one hypothesis at a time for clear signal

## Results Format
After each experiment, report:
- Overall accuracy
- Per-category accuracy
- What you changed and why
- Whether to keep or discard
