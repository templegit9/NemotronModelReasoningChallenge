#!/bin/bash
# Setup script for RunPod H100
# Run: bash /workspace/repo/setup_runpod.sh

set -e
echo "=== Setting up RunPod for Nemotron training ==="

# Install dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install transformers==4.46.3 peft==0.13.2 accelerate==1.1.1 datasets==3.1.0
pip install mamba-ssm==2.2.2 causal-conv1d==1.4.0
pip install vllm==0.6.4.post1
pip install anthropic  # For distillation

echo ""
echo "=== Verifying installations ==="
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import mamba_ssm; print(f'Mamba SSM: {mamba_ssm.__version__}')"
python -c "import vllm; print(f'vLLM: {vllm.__version__}')"
python -c "import anthropic; print(f'Anthropic: {anthropic.__version__}')"

echo ""
echo "=== Checking models ==="
if [ -d "/workspace/nemotron" ]; then
    echo "Nemotron found at /workspace/nemotron"
else
    echo "Nemotron not found. Download: huggingface-cli download nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16 --local-dir /workspace/nemotron"
fi

# Pre-download Zamba2 proxy model for fast iteration
echo "Pre-downloading Zamba2-2.7B proxy model..."
python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; \
AutoTokenizer.from_pretrained('Zyphra/Zamba2-2.7B-instruct', trust_remote_code=True); \
print('Zamba2 tokenizer cached')" 2>/dev/null || echo "Zamba2 download will happen on first use"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Set API key:    export ANTHROPIC_API_KEY=sk-ant-..."
echo "  2. Distill CoT:    python /workspace/repo/distill_cot.py \\"
echo "       --input /workspace/repo/data/train_split.csv \\"
echo "       --output /workspace/repo/data/distilled_train.jsonl"
echo "  3. Fast proxy training (~5 min):"
echo "       python /workspace/repo/train_h100.py --model zamba2 \\"
echo "       --data /workspace/repo/data/distilled_train.jsonl"
echo "  4. Full Nemotron training (~8-13 hrs):"
echo "       python /workspace/repo/train_h100.py --model nemotron \\"
echo "       --data /workspace/repo/data/distilled_train.jsonl"
echo "  5. Evaluate:       python /workspace/repo/evaluate_local.py \\"
echo "       --adapter /workspace/adapter_zamba2 \\"
echo "       --data /workspace/repo/data/val_split.csv"
