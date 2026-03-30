"""Prepare data for autoresearch experiments. DO NOT MODIFY — this is fixed."""

import json
import os

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TRAIN_JSONL = os.path.join(DATA_DIR, "train_formatted.jsonl")
VAL_JSONL = os.path.join(DATA_DIR, "val_formatted.jsonl")


def load_data(path):
    """Load JSONL data, returning list of dicts with 'messages', 'category', 'id'."""
    examples = []
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            examples.append(row)
    return examples


def load_train():
    return load_data(TRAIN_JSONL)


def load_val():
    return load_data(VAL_JSONL)


if __name__ == "__main__":
    train = load_train()
    val = load_val()
    print(f"Train: {len(train)} examples")
    print(f"Val: {len(val)} examples")

    # Category breakdown
    from collections import Counter
    train_cats = Counter(e['category'] for e in train)
    val_cats = Counter(e['category'] for e in val)
    print(f"\nTrain categories: {dict(train_cats)}")
    print(f"Val categories: {dict(val_cats)}")
