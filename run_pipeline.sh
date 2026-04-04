#!/bin/bash
# End-to-end pipeline: distill -> train -> evaluate -> package
# Run: bash /workspace/repo/run_pipeline.sh
set -e

REPO="/workspace/repo"
DATA="$REPO/data"

echo "=== Step 1: Distill CoT from Claude ==="
echo "This takes ~30-60 min for 8,553 examples"
python $REPO/distill_cot.py \
    --input $DATA/train_split.csv \
    --output $DATA/distilled_train.jsonl \
    --batch-size 50 \
    --resume

echo ""
echo "=== Also distill validation set ==="
python $REPO/distill_cot.py \
    --input $DATA/val_split.csv \
    --output $DATA/distilled_val.jsonl \
    --batch-size 50 \
    --resume

echo ""
echo "=== Step 2: Train LoRA adapter ==="
echo "This takes ~8-13 hours on H100"
python $REPO/train_h100.py \
    --data $DATA/distilled_train.jsonl \
    --output /workspace/adapter

echo ""
echo "=== Step 3: Evaluate locally ==="
python $REPO/evaluate_local.py \
    --adapter /workspace/adapter \
    --data $DATA/val_split.csv

echo ""
echo "=== Step 4: Package for Kaggle submission ==="
cd /workspace/adapter
zip -r /workspace/submission.zip .
echo "Submission packaged: $(du -h /workspace/submission.zip | cut -f1)"
echo ""
echo "=== DONE ==="
echo "Upload /workspace/submission.zip to Kaggle"
