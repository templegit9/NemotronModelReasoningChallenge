# Experiment Plan: Nemotron 0.55 → 0.82+

Each experiment is a single focused change. Run them in order.
Use **Zamba2-2.7B** (hybrid Mamba-Transformer, ~5 min/run) as proxy on H100.
After finding best strategy, apply to full Nemotron for Kaggle submission.

## Quick Commands
```bash
# Proxy (fast, ~5-10 min):
python train_h100.py --model zamba2 --data data/distilled_train.jsonl

# Alternative proxy (pure transformer, ~10-15 min):
python train_h100.py --model qwen --data data/distilled_train.jsonl

# Full Nemotron (final submission, ~8-13 hours):
python train_h100.py --model nemotron --data data/distilled_train.jsonl
```

## Phase A: Data Strategy (on Zamba2 proxy, ~5 min each)

### Exp 1: Baseline — Algorithmic CoT on Zamba2
**What**: Test existing algorithmic CoT data on Zamba2 proxy.
```
python train_h100.py --model zamba2 --data data/train_formatted.jsonl
```
**Purpose**: Establish proxy baseline.

### Exp 2: Distilled CoT on Zamba2
**What**: Test Claude-distilled CoT data on Zamba2.
```
python distill_cot.py --input data/train_split.csv --output data/distilled_train.jsonl
python train_h100.py --model zamba2 --data data/distilled_train.jsonl
```
**Purpose**: Measure impact of data quality (biggest expected gain).

### Exp 3: Hybrid — Algorithmic Easy + Distilled Hard
**What**: Use solvers for easy categories, Claude for hard categories.
```
python distill_cot.py --input data/train_split.csv --output data/distilled_bit.jsonl --category BIT_MANIPULATION
python distill_cot.py --input data/train_split.csv --output data/distilled_sym.jsonl --category SYMBOL_TRANSFORM
python merge_data.py --easy data/train_formatted.jsonl \
    --hard data/distilled_bit.jsonl data/distilled_sym.jsonl \
    --output data/merged_train.jsonl
python train_h100.py --model zamba2 --data data/merged_train.jsonl
```
**Purpose**: Test if hybrid approach is better than full distillation.

---

## Phase B: Hyperparameter Sweep (on Zamba2 proxy, ~5 min each)

Use the best data from Phase A for all these experiments.

### Exp 4: Learning Rate Sweep
```
python train_h100.py --model zamba2 --data data/BEST.jsonl --lr 2e-5
python train_h100.py --model zamba2 --data data/BEST.jsonl --lr 5e-5
python train_h100.py --model zamba2 --data data/BEST.jsonl --lr 1e-4
```

### Exp 5: Epoch Count
```
python train_h100.py --model zamba2 --data data/BEST.jsonl --epochs 2
python train_h100.py --model zamba2 --data data/BEST.jsonl --epochs 3
python train_h100.py --model zamba2 --data data/BEST.jsonl --epochs 4
```

### Exp 6: LoRA Alpha Tuning
```
python train_h100.py --model zamba2 --data data/BEST.jsonl --lora-alpha 32
python train_h100.py --model zamba2 --data data/BEST.jsonl --lora-alpha 64
python train_h100.py --model zamba2 --data data/BEST.jsonl --lora-alpha 128
```

### Exp 7: Category Weight Tuning
```
python train_h100.py --model zamba2 --data data/BEST.jsonl --cat-weight-hard 2.0
python train_h100.py --model zamba2 --data data/BEST.jsonl --cat-weight-hard 3.0
python train_h100.py --model zamba2 --data data/BEST.jsonl --cat-weight-hard 4.0
```

---

## Phase C: Data Quality Experiments (on Zamba2 proxy)

### Exp 8: Multi-Attempt Distillation for Hard Categories
**Change**: For BIT_MANIPULATION and SYMBOL_TRANSFORM, run Claude 5 times per puzzle at temperature=0.7.
Keep the best correct CoT. This dramatically increases the % of verified-correct reasoning.
Modify `distill_cot.py`:
```python
# In solve_puzzle: generate 5 attempts, pick any correct one
MULTI_ATTEMPT_CATEGORIES = {"BIT_MANIPULATION", "SYMBOL_TRANSFORM"}
NUM_ATTEMPTS = 5
```

### Exp 9: Synthetic Data Generation
**Change**: For easy categories where solvers work perfectly, generate 2x more training examples by:
- Varying numbers in NUMBER_SYSTEM problems
- Varying conversion factors in UNIT_CONVERSION
- Varying gravity/time in PHYSICS
Use solvers to generate both problems and answers, then distill CoT.

### Exp 10: Rejection Sampling Fine-Tuning
**Change**: After best adapter from experiments above:
1. Generate 8 solutions per val puzzle at temperature=0.7
2. Keep only correct solutions
3. Retrain on these self-generated correct solutions + original data
This is the technique NVIDIA used for OpenMath-Nemotron.

---

## Phase D: Transfer to Nemotron (Final Submission)

### Exp T1: Best Strategy on Nemotron
**What**: Apply winning data + hyperparams from Phases A-C to full Nemotron.
```
python train_h100.py --model nemotron --data data/BEST.jsonl \
    --lr BEST_LR --epochs BEST_EPOCHS --lora-alpha BEST_ALPHA
```
**Time**: ~8-13 hours. Submit to Kaggle immediately after.

### Exp T2: Nemotron-specific Module Ablation
**What**: Test which LoRA target modules matter on Nemotron (can't test on proxy).
This is Nemotron-specific due to its unique Mamba-Transformer-MoE architecture.

---

## Phase E: Advanced Experiments (If still below 0.82)

### Exp 11: Max Sequence Length Increase
**Change**: `MAX_SEQ_LENGTH = 6144` (from 4096).
Allows longer reasoning chains, especially for hard categories.
Monitor VRAM usage on H100.

### Exp 12: Target Module Ablation
**Change**: Test different LoRA target module combinations:
- Exp 12a: Full set (9 modules, current)
- Exp 12b: Mamba only: in_proj, out_proj
- Exp 12c: Attention only: q_proj, k_proj, v_proj, o_proj
- Exp 12d: MLP only: gate_proj, up_proj, down_proj
Identify which module type contributes most.

### Exp 13: DPO (Direct Preference Optimization)
**Change**: After SFT, apply DPO using:
- Preferred: correct CoT + correct answer
- Rejected: incorrect CoT + wrong answer (from failed Claude attempts)
Requires saving reject traces during distillation.

### Exp 14: Curriculum Learning
**Change**: Train epochs in order of difficulty:
- Epoch 1: Only easy categories (NUMBER_SYSTEM, UNIT_CONVERSION, PHYSICS)
- Epoch 2: Add medium (TEXT_ENCRYPTION)
- Epoch 3: All categories with heavy hard weighting

---

## Experiment Tracking Template

| Exp | Description | Key Change | Val Acc | Notes |
|-----|-------------|-----------|---------|-------|
| 1   | Expanded LoRA + current data | +Mamba layers, alpha=64 | | |
| 2   | Distilled CoT | Claude-generated reasoning | | |
| 3   | Distilled hard + algorithmic easy | Hybrid data | | |
| 4a  | LR=2e-5 | | | |
| 4b  | LR=5e-5 | | | |
| 4c  | LR=1e-4 | | | |
| 5a  | 2 epochs | | | |
| 5b  | 3 epochs | | | |
| 5c  | 4 epochs | | | |
| ... | ... | ... | ... | ... |

## Decision Rules
- If Exp 1 already > 0.65: LoRA config was the bottleneck, focus on hyperparameter tuning
- If Exp 2 >> Exp 1: Data quality is the bottleneck, focus on distillation quality
- If Exp 8 helps significantly: invest in multi-attempt and rejection sampling
- Stop hyperparameter search when improvements < 0.5%
- Always Kaggle-submit the best local eval result (5/day limit)
