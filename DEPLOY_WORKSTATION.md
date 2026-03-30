# Deploy Autoresearch on Workstation

Quick setup guide for your RTX 5070 Ti workstation (10.0.0.222).

## 1. Clone the Repo

```bash
cd ~
git clone https://github.com/templegit9/NemotronModelReasoningChallenge.git
cd NemotronModelReasoningChallenge
```

## 2. Install Dependencies

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft anthropic numpy
```

> If you already have PyTorch with CUDA, skip the first line.

Verify GPU is visible:
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

Expected: `CUDA: True, GPU: NVIDIA GeForce RTX 5070 Ti`

## 3. Download the Proxy Model (one-time, ~6GB)

```bash
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading Qwen2.5-3B-Instruct...')
AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')
print('Done!')
"
```

## 4. Verify Data Files Exist

```bash
ls -la data/train_formatted.jsonl data/val_formatted.jsonl
```

You should see:
- `train_formatted.jsonl` — ~8,553 lines
- `val_formatted.jsonl` — ~947 lines

Quick check:
```bash
wc -l data/train_formatted.jsonl data/val_formatted.jsonl
```

## 5. Test Run (Dry Check)

Make sure everything loads without errors:
```bash
cd autoresearch
python prepare_data.py
```

Expected output:
```
Train: 8553 examples
Val: 947 examples
```

## 6. Set Your API Key

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

To persist across sessions, add it to your shell profile:
```bash
echo 'export ANTHROPIC_API_KEY="sk-ant-..."' >> ~/.bashrc
source ~/.bashrc
```

## 7. Test with 1 Experiment

```bash
cd ~/NemotronModelReasoningChallenge/autoresearch
python run_loop.py --max-experiments 1
```

This will:
1. Call Claude API to propose a modification (~5 sec)
2. Train Qwen2.5-3B with LoRA (~10-15 min)
3. Evaluate on 947 val examples (~5-10 min)
4. Log results to `results.tsv`

If it completes successfully, you're ready for overnight runs.

## 8. Launch Overnight Run

```bash
# Use nohup so it survives SSH disconnection
cd ~/NemotronModelReasoningChallenge/autoresearch
nohup python run_loop.py --max-experiments 50 > autoresearch.log 2>&1 &
echo $!  # Note this PID to check on it later
```

## 9. Monitor Progress

**Option A — Terminal dashboard** (SSH into workstation):
```bash
cd ~/NemotronModelReasoningChallenge/autoresearch
python dashboard.py
```

**Option B — Web dashboard** (view from any device):
```bash
pip install dash plotly  # one-time
cd ~/NemotronModelReasoningChallenge/autoresearch
python dashboard.py --web --port 8050
```
Then open `http://10.0.0.222:8050` in your browser.

**Option C — Quick check**:
```bash
# View latest results
cat ~/NemotronModelReasoningChallenge/autoresearch/results.tsv | column -t -s $'\t'

# Check if still running
ps aux | grep run_loop

# Tail the log
tail -f ~/NemotronModelReasoningChallenge/autoresearch/autoresearch.log
```

## 10. Stop Early (if needed)

```bash
# Graceful stop (finishes current experiment)
kill -INT <PID>

# Or find and kill
pkill -f run_loop.py
```

## 11. Pull Results to Your Mac

From your Mac:
```bash
scp 10.0.0.222:~/NemotronModelReasoningChallenge/autoresearch/results.tsv ./autoresearch/
```

Or push from workstation and pull on Mac:
```bash
# On workstation
cd ~/NemotronModelReasoningChallenge
git add autoresearch/results.tsv
git commit -m "Add overnight experiment results"
git push

# On Mac
git pull
```

---

## Troubleshooting

**CUDA out of memory**
- Reduce `MAX_SEQ_LENGTH` to 512 in `train.py`
- Reduce `BATCH_SIZE` to 1
- Kill other GPU processes: `nvidia-smi` then `kill <PID>`

**Module not found**
```bash
pip install transformers peft anthropic
```

**Permission denied on git push**
```bash
git config credential.helper store
git push  # enter credentials once, saved for future
```

**API rate limit**
- The loop handles this gracefully, logs it as an error, and continues
- Use `--model claude-haiku-4-5-20251001` for cheaper/faster agent calls

---

## Estimated Costs

| Model | Per Experiment | 50 Experiments |
|-------|---------------|----------------|
| Claude Sonnet | ~$0.06 | ~$3.00 |
| Claude Haiku | ~$0.01 | ~$0.50 |

GPU compute is free (your own hardware). The only cost is Claude API calls.

---

## Summary Cheatsheet

```bash
# First time setup
git clone https://github.com/templegit9/NemotronModelReasoningChallenge.git
cd NemotronModelReasoningChallenge
pip install torch transformers peft anthropic numpy
export ANTHROPIC_API_KEY="sk-ant-..."

# Launch overnight
cd autoresearch
nohup python run_loop.py --max-experiments 50 > autoresearch.log 2>&1 &

# Monitor
python dashboard.py          # terminal
python dashboard.py --web    # browser at :8050
tail -f autoresearch.log     # raw log
```
