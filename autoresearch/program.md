# Research Program: Nemotron Reasoning Challenge

## Objective
Maximize accuracy on 6 "Alice's Wonderland" reasoning puzzle categories.
We fine-tune Qwen2.5-3B locally as a proxy, then transfer findings to the Nemotron submission.

## Metric
- **Primary**: Overall validation accuracy (correct / total)
- **Categories**: NUMBER_SYSTEM, UNIT_CONVERSION, PHYSICS, TEXT_ENCRYPTION, BIT_MANIPULATION, SYMBOL_TRANSFORM
- **Leaderboard top**: 0.82 (82%) — that is our target

## 6 Experiment Dimensions (from the official challenge)

### 1. Prompting Strategies
Modify how training examples are formatted in the dataset:
- Change the system prompt (add reasoning instructions, role descriptions)
- Reformat the CoT reasoning: step-by-step numbered, bullet points, or brief
- Add category-specific instructions (e.g. "Think in binary for BIT_MANIPULATION")
- Add few-shot examples in the system prompt
- Enforce output format (e.g. "Answer: X" on the last line)

### 2. Data Filtering and Curation
Control which examples get trained on:
- Filter out examples where the reasoning is very short (low quality)
- Filter by sequence length (skip examples that get heavily truncated)
- Oversample hard categories (BIT_MANIPULATION, SYMBOL_TRANSFORM, TEXT_ENCRYPTION)
- Undersample easy categories if they dominate training
- Filter examples where prompt_length >= token_count (all-masked labels → NaN)

### 3. Synthetic Data Generation
Augment training data at runtime within train.py:
- Duplicate hard-category examples with paraphrased system prompts
- Create reversed examples (answer → question) for harder categories
- Apply text augmentation to existing examples (synonym replacement in reasoning)

### 4. Reinforcement Learning / Alternative Objectives
Replace or augment the standard cross-entropy loss:
- Upweight loss on the final answer token (encourage correct answer)
- Upweight loss on reasoning steps vs. padding
- Ignore loss on trivially correct steps, focus on hard transitions
- Apply label smoothing to avoid overconfidence

### 5. Lightweight Fine-tuning (LoRA Hyperparameters)
The baseline approach — still important to optimize:
- LoRA rank: 8, 16, 32 (higher = more capacity, slower)
- LoRA alpha: 16, 32, 64 (scaling factor, often set = rank or 2× rank)
- LoRA dropout: 0, 0.05, 0.1 (regularization)
- Target modules: q_proj only, q+v, q+k+v, all attention + MLP (gate_proj, up_proj)
- Learning rate: 1e-5 to 1e-4
- Warmup ratio: 0.03 to 0.1
- Optimizer: AdamW, Adafactor

### 6. Other Approaches
- Curriculum learning: train easy categories first, hard ones later
- Loss masking: only compute loss on the final answer, not the full chain
- Different CoT length targets per category (short for easy, long for hard)

## Hard Constraints (DO NOT VIOLATE)
- `BATCH_SIZE = 1` — safety default; A100 80GB could handle 2 but keep at 1 for stability
- `NUM_EPOCHS = 1` — more epochs would exceed time budget
- `MAX_SEQ_LENGTH = 2048` — do NOT lower (truncates Nemotron reasoning chains)
- `MAX_TRAINING_STEPS = 150` — confirmed safe on A100 (~20-30 min); do NOT exceed 200
- `LORA_RANK <= 32` — competition constraint (currently set to 32)
- `LORA_TARGET_MODULES = r".*\.(in_proj|out_proj|up_proj|down_proj)$"` — Mamba hybrid layers; do NOT use q_proj/v_proj (those are transformer attention, not all Nemotron layers have them)
- Only import: standard library, torch, transformers, peft, numpy, mamba_ssm
- `import mamba_ssm` must stay at top (registers Mamba CUDA kernels before transformers loads model)
- `prepare_data` import must stay unchanged
- Pre-tokenize ALL data in `ReasoningDataset.__init__`, not in `__getitem__`
- Do NOT add gradient checkpointing — incompatible with Mamba layers (causes RuntimeError)

## Strategy
- Hard categories (BIT_MANIPULATION, SYMBOL_TRANSFORM) score near 0 — biggest opportunity
- Prompting and data-mix changes transfer better to Nemotron than hyperparams
- Make ONE focused change per experiment — clear signal
- If a category is at 0%, try: (1) oversample it 1.5×, (2) add category hint to system prompt
- CATEGORY_WEIGHTS max value is 1.5 — higher values cause tokenisation to exceed the 45-min timeout
- Check `results.tsv` for the best run so far and build on it

## Learned Findings (READ BEFORE PROPOSING ANY CHANGE)

### What works
- **Exp 18 is the best result (20.80% overall)** — use it as the base for all future experiments
- **Simple reasoning starters prepended to assistant messages** for hard categories is the proven winning technique:
  ```python
  REASONING_STARTERS = {
      "TEXT_ENCRYPTION": "I need to analyze the encryption pattern and decode step by step.",
      "BIT_MANIPULATION": "Let me think about the bit operations in binary representation.",
      "SYMBOL_TRANSFORM": "I will trace through the symbol transformation rules carefully.",
  }
  ```
  Inject via: `inject_reasoning_starter(example)` that prepends starter to assistant message content BEFORE tokenization.
- **LORA_RANK=16** outperforms LORA_RANK=32 — more capacity without guidance dilutes learning
- **Category oversampling at 1.5×** for hard categories helps (keep this in all experiments)

### What fails — DO NOT repeat these
- **Any tensor manipulation** (loss masking, loss scaling, gradient weighting) → always TRAIN_FAILED
- **Modifying the chat template or system prompt per-category at template level** → TRAIN_FAILED
- **Multiple simultaneous changes** (few-shot + augmentation + prompting together) → TRAIN_FAILED
- **Structured reasoning frameworks / multi-step markers injected into content** → TRAIN_FAILED
- **Reverting to baseline without reasoning starters** → drops to 12-13% (confirmed exp 17)

### Pattern warning
After a successful experiment, DO NOT immediately try something complex. The pattern of success → complex attempt → TRAIN_FAILED has repeated 3 times. After any OK result, make the **smallest possible next step** building on what worked.

### Current confirmed baseline state (Nemotron on RunPod A100 — experiment 1 start)
The train.py you are working with should have:
- `LORA_RANK = 32` (Nemotron baseline; max allowed by competition)
- `LORA_ALPHA = 64` (2× rank — theoretically optimal)
- `LEARNING_RATE = 2e-5`
- `MAX_TRAINING_STEPS = 150` (safe on A100 ~20-30 min)
- `LORA_TARGET_MODULES = r".*\.(in_proj|out_proj|up_proj|down_proj)$"` (regex, not list)
- REASONING_STARTERS dict with inject_reasoning_starter() function PRESENT (proven +5-8% on Qwen proxy)
- CATEGORY_WEIGHTS with 1.5× for hard categories
- `import mamba_ssm` at top (required before transformers)
- NO gradient checkpointing

Before proposing any change, READ the current values in train.py and confirm this state.

### Recommended next experiments (in priority order)
1. **Baseline first** — run exp 1 with the default Nemotron baseline to establish what accuracy we get on this model before any changes
2. **Tune LEARNING_RATE**: try 5e-5 — one number change to `LEARNING_RATE = 5e-5`
3. **Refine reasoning starters**: make them more specific (e.g. for BIT_MANIPULATION: "Let me convert each value to binary and apply the bitwise operation step by step.") — change only the string text
4. **Try LORA_ALPHA=32** — one number change (2× rank might be better for this model size)
5. Only after the above are exhausted: simple data filtering (skip examples with assistant content shorter than 50 tokens)

## Infrastructure Notes (READ CAREFULLY before interpreting results)
- Running on RunPod A100 80GB. Model is Nemotron-3-Nano-30B-A3B (Mamba-Transformer hybrid, ~60GB bf16).
- MODEL_PATH=/workspace/nemotron, ADAPTER_PATH=/workspace/adapter (set as env vars by run_loop.py)
- ALL past EVAL_FAILEDs were caused by evaluation timing out — NOT a broken eval script. max_new_tokens=128 is now set to prevent this.
- DO NOT change output formats, answer formats, or add "ANSWER:" prefixes to fix EVAL_FAILED. The eval script uses `\boxed{}` and fallback parsing that already works.
- Experiments showing TRAIN_FAILED mean either a code error OR a training timeout (4-hour limit). Check: `model.enable_input_require_grads()` present, no `torch.no_grad()` in training loop, loss tensor not detached.
- **CRITICAL**: `import mamba_ssm` must stay at the TOP of train.py. Removing it breaks model loading.
- **CRITICAL**: Do NOT add `gradient_checkpointing_enable()` — this crashes with Mamba layers.
- **MAX_TRAINING_STEPS = 150** is safe on A100 (~20-30 min). Keep between 100 and 200.
- When you see failures, make ONE small targeted change. Do NOT rewrite large sections.
- NEVER remove `model.enable_input_require_grads()` — causes `RuntimeError: element 0 of tensors does not require grad`.
- NEVER add custom answer format tokens like "ANSWER:" or change the extract_answer logic — breaks the evaluator.
