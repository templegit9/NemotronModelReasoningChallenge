# Claude Code — Workstation Setup Instructions

You are Claude Code running on a Windows workstation with an RTX 5070 Ti (16GB VRAM).
Your job is to set up and run the autoresearch experiment loop.

## Machine Info
- OS: Windows
- GPU: NVIDIA GeForce RTX 5070 Ti (16GB VRAM)
- CUDA: 13.2
- Python 3.14 is installed but TOO NEW for PyTorch. You must use Python 3.12.
- IP: 10.0.0.222

## Step-by-Step Setup

### 1. Install Python 3.12
```powershell
winget install Python.Python.3.12
```

### 2. Clone repo and create venv
```powershell
cd %USERPROFILE%
git clone https://github.com/templegit9/NemotronModelReasoningChallenge.git
cd NemotronModelReasoningChallenge
py -3.12 -m venv venv
venv\Scripts\activate
```

If `py -3.12` doesn't work, try the full path:
```powershell
"C:\Users\oluyi\AppData\Local\Programs\Python\Python312\python.exe" -m venv venv
```

### 3. Install dependencies
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install transformers peft anthropic numpy
```

### 4. Verify GPU works
```powershell
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
```
Must show `CUDA: True`.

### 5. Download proxy model (one-time, ~6GB)
```powershell
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; print('Downloading...'); AutoTokenizer.from_pretrained('Qwen/Qwen2.5-3B-Instruct'); AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-3B-Instruct'); print('Done!')"
```

### 6. Verify data files
```powershell
python autoresearch\prepare_data.py
```
Must show: `Train: 8553 examples, Val: 947 examples`

### 7. Set API key
The user will provide their Anthropic API key. Set it:
```powershell
$env:ANTHROPIC_API_KEY = "the-key-here"
```

### 8. Test with 1 experiment
```powershell
cd autoresearch
python run_loop.py --max-experiments 1
```
This takes ~20-25 minutes (train + evaluate). If it completes with results in results.tsv, everything works.

### 9. Launch full run
```powershell
python run_loop.py --max-experiments 50
```

### 10. Launch dashboard (optional, in second terminal)
```powershell
pip install dash plotly
cd %USERPROFILE%\NemotronModelReasoningChallenge\autoresearch
python dashboard.py --web --port 8050
```

## What the Autoresearch Loop Does
- Calls Claude API to propose a change to `autoresearch/train.py`
- Runs the modified training script (LoRA fine-tuning on Qwen2.5-3B)
- Evaluates on 947 validation examples
- Logs results to `autoresearch/results.tsv`
- Backs up every version of train.py to `autoresearch/backups/`
- Repeats

## Troubleshooting
- If CUDA OOM: edit `autoresearch/train.py`, reduce `BATCH_SIZE` to 1 or `MAX_SEQ_LENGTH` to 512
- If torch install fails: try `pip install torch` without the index URL
- If `py -3.12` not found: check `where py` or install from python.org directly
- The OUTPUT_DIR in train.py uses `/tmp/` which doesn't exist on Windows — change to `C:\temp\autoresearch_adapter` if needed

## IMPORTANT: Windows Path Fix
The train.py has `OUTPUT_DIR = "/tmp/autoresearch_adapter"`. On Windows, change this to:
```python
OUTPUT_DIR = os.path.join(os.environ.get("TEMP", "C:\\temp"), "autoresearch_adapter")
```
