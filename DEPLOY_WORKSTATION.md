# Deploy Autoresearch on Workstation (Windows)

Quick setup guide for your RTX 5070 Ti Windows workstation (10.0.0.222).

All commands below are for **PowerShell** or **Command Prompt**.

## 1. Clone the Repo

```powershell
cd %USERPROFILE%
git clone https://github.com/templegit9/NemotronModelReasoningChallenge.git
cd NemotronModelReasoningChallenge
```

## 2. Create Virtual Environment (recommended)

```powershell
python -m venv venv
venv\Scripts\activate
```

## 3. Install Dependencies

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft anthropic numpy
```

> If you already have PyTorch with CUDA, skip the first line.

Verify GPU is visible:
```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```

Expected: `CUDA: True, GPU: NVIDIA GeForce RTX 5070 Ti`

## 3. Download the Proxy Model (one-time, ~6GB)

```powershell
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; print('Downloading...'); AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct'); print('Done!')"
```

## 4. Verify Data Files Exist

```powershell
dir data\train_formatted.jsonl data\val_formatted.jsonl
```

You should see:
- `train_formatted.jsonl` — ~8,553 lines
- `val_formatted.jsonl` — ~947 lines

Quick line count:
```powershell
find /c /v "" data\train_formatted.jsonl data\val_formatted.jsonl
```

## 5. Test Run (Dry Check)

```powershell
cd autoresearch
python prepare_data.py
```

Expected output:
```
Train: 8553 examples
Val: 947 examples
```

## 6. Set Your API Key

**PowerShell** (current session only):
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
```

**Command Prompt** (current session only):
```cmd
set ANTHROPIC_API_KEY=sk-ant-...
```

**Persist permanently** (PowerShell as Admin):
```powershell
[System.Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "sk-ant-...", "User")
```

Or: Settings > System > About > Advanced system settings > Environment Variables > New

## 7. Test with 1 Experiment

```powershell
cd %USERPROFILE%\NemotronModelReasoningChallenge\autoresearch
python run_loop.py --max-experiments 1
```

This will:
1. Call Claude API to propose a modification (~5 sec)
2. Train Qwen2.5-3B with LoRA (~10-15 min)
3. Evaluate on 947 val examples (~5-10 min)
4. Log results to `results.tsv`

If it completes successfully, you're ready for overnight runs.

## 8. Launch Overnight Run

**Option A — Keep PowerShell open** (simplest):
```powershell
cd %USERPROFILE%\NemotronModelReasoningChallenge\autoresearch
python run_loop.py --max-experiments 50 2>&1 | tee autoresearch.log
```

**Option B — Run in background** (survives closing the terminal):
```powershell
Start-Process -NoNewWindow python -ArgumentList "run_loop.py --max-experiments 50" -RedirectStandardOutput autoresearch.log -RedirectStandardError autoresearch_err.log
```

**Option C — Use `pythonw`** (fully detached):
```powershell
start /B pythonw run_loop.py --max-experiments 50
```

> Tip: Disable Windows sleep/hibernate so the run isn't interrupted:
> Settings > System > Power & sleep > set Screen and Sleep to "Never" while running.

## 9. Monitor Progress

**Option A — Terminal dashboard** (open a second PowerShell):
```powershell
cd %USERPROFILE%\NemotronModelReasoningChallenge\autoresearch
python dashboard.py
```

**Option B — Web dashboard** (view from any device on your network):
```powershell
pip install dash plotly
python dashboard.py --web --port 8050
```
Then open `http://10.0.0.222:8050` in your browser (Mac, phone, etc.)

> If the web dashboard doesn't load from another device, allow it through Windows Firewall:
> ```powershell
> netsh advfirewall firewall add rule name="Autoresearch Dashboard" dir=in action=allow protocol=TCP localport=8050
> ```

**Option C — Quick check**:
```powershell
# View results
type %USERPROFILE%\NemotronModelReasoningChallenge\autoresearch\results.tsv

# Check if still running
tasklist | findstr python

# Tail the log (PowerShell)
Get-Content autoresearch.log -Tail 20 -Wait
```

## 10. Stop Early (if needed)

```powershell
# Press Ctrl+C in the running terminal

# Or kill from another terminal
tasklist | findstr python
taskkill /PID <PID> /F
```

## 11. Pull Results to Your Mac

From your Mac terminal:
```bash
scp user@10.0.0.222:C:/Users/<username>/NemotronModelReasoningChallenge/autoresearch/results.tsv ./autoresearch/
```

Or push from workstation and pull on Mac:
```powershell
# On workstation
cd %USERPROFILE%\NemotronModelReasoningChallenge
git add autoresearch\results.tsv
git commit -m "Add overnight experiment results"
git push
```
```bash
# On Mac
git pull
```

---

## Troubleshooting

**CUDA out of memory**
- Reduce `MAX_SEQ_LENGTH` to 512 in `train.py`
- Reduce `BATCH_SIZE` to 1
- Check what's using the GPU: `nvidia-smi`
- Kill other GPU processes: `taskkill /PID <PID> /F`

**`python` not found**
- Use `python3` instead, or ensure Python is in your PATH
- If using Anaconda: `conda activate base`

**Module not found**
```powershell
pip install transformers peft anthropic numpy
```

**Permission denied on git push**
```powershell
git config credential.helper manager
git push
```

**API rate limit**
- The loop handles this gracefully, logs it as an error, and continues
- Use `--model claude-haiku-4-5-20251001` for cheaper/faster agent calls

**Windows Defender blocking scripts**
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## Estimated Costs

| Model | Per Experiment | 50 Experiments |
|-------|---------------|----------------|
| Claude Sonnet | ~$0.06 | ~$3.00 |
| Claude Haiku | ~$0.01 | ~$0.50 |

GPU compute is free (your own hardware). The only cost is Claude API calls.

---

## Summary Cheatsheet (PowerShell)

```powershell
# First time setup
git clone https://github.com/templegit9/NemotronModelReasoningChallenge.git
cd NemotronModelReasoningChallenge
python -m venv venv
venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft anthropic numpy
$env:ANTHROPIC_API_KEY = "sk-ant-..."

# Download proxy model
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct')"

# Test
cd autoresearch
python prepare_data.py
python run_loop.py --max-experiments 1

# Launch overnight
python run_loop.py --max-experiments 50 2>&1 | tee autoresearch.log

# Monitor (second terminal)
python dashboard.py            # terminal
python dashboard.py --web      # browser at :8050
```
