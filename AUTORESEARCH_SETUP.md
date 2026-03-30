# Autoresearch Setup Guide

Autonomous experiment loop that uses Claude API to iteratively optimize LoRA fine-tuning on your local RTX 5070 Ti.

## How It Works

Each experiment (~15 min):
1. Claude reads the research program + all past results
2. Proposes one focused change to `train.py` (e.g., "increase LR to 5e-5", "oversample BIT_MANIPULATION 2x")
3. Trains Qwen2.5-3B with LoRA on your GPU
4. Evaluates on the 947 validation examples
5. Logs per-category accuracy to `results.tsv`
6. Repeats with knowledge of what worked and what didn't

Launch before bed, wake up to ~50+ experiments tried.

## Prerequisites

- NVIDIA RTX 5070 Ti (16GB VRAM)
- Python 3.10+
- CUDA toolkit installed
- Anthropic API key

## Setup

### 1. Install Dependencies

```bash
pip install torch transformers peft anthropic numpy
```

### 2. Download Proxy Model (one-time, ~6GB)

```bash
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')"
```

### 3. Set API Key

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

### 4. Launch

```bash
cd autoresearch
python run_loop.py --max-experiments 50
```

## Options

```
--max-experiments N    Max experiments to run (default: 50)
--model MODEL          Claude model for the agent (default: claude-sonnet-4-20250514)
--start-from N         Starting experiment number (default: 1)
```

## Monitoring

- **Live output**: Watch the terminal for per-experiment results
- **Results log**: `autoresearch/results.tsv` — tab-separated with overall + per-category accuracy
- **API costs**: Token usage printed after each agent call (cumulative tracked)
- **Stop anytime**: Ctrl+C cleanly stops the loop

## File Structure

```
autoresearch/
  program.md        # Research program — what to optimize, constraints, tips
  train.py          # THE file Claude modifies each iteration
  evaluate.py       # Fixed evaluation script (DO NOT MODIFY)
  prepare_data.py   # Fixed data loading (DO NOT MODIFY)
  run_loop.py       # The autonomous loop driver
  results.tsv       # Experiment log (auto-generated)
  backups/          # Every train.py version saved (auto-generated)
```

## Safety

- Every `train.py` is backed up before modification (`backups/train_exp001.py`, etc.)
- If training or evaluation fails, the previous `train.py` is automatically restored
- Hard timeout of 20 min per training run prevents runaway experiments
- All experiments are logged even if they fail

## Why a Proxy Model?

The real competition model (Nemotron-3-Nano-30B-A3B, ~60GB in bf16) won't fit on 16GB VRAM. Qwen2.5-3B-Instruct (~6GB) serves as a stand-in for fast iteration.

**What transfers well** (proxy -> Nemotron):
- Training data format and CoT quality
- Category sampling weights
- Data filtering decisions
- Rough hyperparameter ranges

**What may not transfer**:
- Exact learning rate values
- Architecture-specific settings (Nemotron has Mamba layers, Qwen doesn't)

## Workflow

1. Run autoresearch overnight on RTX 5070 Ti
2. Review `results.tsv` in the morning
3. Take the best configuration and adapt it for `notebooks/train_submission.py`
4. Submit to Kaggle (5 submissions/day)
5. Track Kaggle scores in `experiments.csv`
6. Repeat
