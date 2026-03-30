"""Categorize training data into 6 problem types and create stratified train/val split."""

import csv
import random
from collections import defaultdict

SEED = 42
VAL_RATIO = 0.10  # 90/10 split


def classify(prompt: str) -> str:
    p = prompt.lower()
    if 'bit manipulation' in p:
        return 'BIT_MANIPULATION'
    elif 'encrypt' in p or 'decrypt' in p:
        return 'TEXT_ENCRYPTION'
    elif 'numeral' in p:
        return 'NUMBER_SYSTEM'
    elif 'unit conversion' in p:
        return 'UNIT_CONVERSION'
    elif 'gravitational' in p:
        return 'PHYSICS'
    elif 'transformation rule' in p:
        return 'SYMBOL_TRANSFORM'
    else:
        return 'UNKNOWN'


def main():
    with open('data/train.csv') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    # Add category column
    for row in rows:
        row['category'] = classify(row['prompt'])

    # Check for unknowns
    unknowns = [r for r in rows if r['category'] == 'UNKNOWN']
    if unknowns:
        print(f"WARNING: {len(unknowns)} rows could not be classified!")
        for r in unknowns[:5]:
            print(f"  {r['prompt'][:100]}...")
        return

    # Group by category for stratified split
    by_category = defaultdict(list)
    for row in rows:
        by_category[row['category']].append(row)

    random.seed(SEED)
    train_rows = []
    val_rows = []

    print("Category breakdown:")
    for cat in sorted(by_category):
        items = by_category[cat]
        random.shuffle(items)
        n_val = int(len(items) * VAL_RATIO)
        val_rows.extend(items[:n_val])
        train_rows.extend(items[n_val:])
        print(f"  {cat}: {len(items)} total -> {len(items) - n_val} train / {n_val} val")

    # Shuffle final sets
    random.shuffle(train_rows)
    random.shuffle(val_rows)

    # Write outputs
    fieldnames = ['id', 'prompt', 'answer', 'category']

    with open('data/train_split.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(train_rows)

    with open('data/val_split.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(val_rows)

    print(f"\nTotal: {len(train_rows)} train / {len(val_rows)} val")
    print("Saved to data/train_split.csv and data/val_split.csv")


if __name__ == '__main__':
    main()
