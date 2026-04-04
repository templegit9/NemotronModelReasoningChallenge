"""
Merge training data from different sources:
- Algorithmic CoT for easy categories (from generate_training_data.py)
- Distilled CoT for hard categories (from distill_cot.py)

Usage:
    python merge_data.py \
        --easy data/train_formatted.jsonl \
        --hard data/distilled_bit.jsonl data/distilled_sym.jsonl \
        --output data/merged_train.jsonl
"""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--easy', required=True, help='JSONL with algorithmic CoT (all categories)')
    parser.add_argument('--hard', nargs='+', required=True, help='JSONL files with distilled CoT for hard categories')
    parser.add_argument('--distilled-all', default='', help='Optional: fully distilled JSONL to use for all categories')
    parser.add_argument('--output', required=True, help='Output merged JSONL')
    parser.add_argument('--hard-categories', nargs='+',
                        default=['BIT_MANIPULATION', 'SYMBOL_TRANSFORM'],
                        help='Categories to replace with distilled CoT')
    args = parser.parse_args()

    # Load easy (algorithmic) data
    easy_examples = {}
    with open(args.easy) as f:
        for line in f:
            obj = json.loads(line)
            easy_examples[obj['id']] = obj

    # Load hard (distilled) data
    hard_examples = {}
    for path in args.hard:
        with open(path) as f:
            for line in f:
                obj = json.loads(line)
                hard_examples[obj['id']] = obj

    # If we have a fully distilled file, use it for categories not in hard_categories
    if args.distilled_all:
        with open(args.distilled_all) as f:
            for line in f:
                obj = json.loads(line)
                cat = obj.get('category', '')
                if cat not in args.hard_categories and obj['id'] not in hard_examples:
                    # Only use distilled for easy categories if it was solved correctly
                    if obj.get('solved_correctly', False):
                        easy_examples[obj['id']] = obj

    # Merge: use hard data for hard categories, easy data for everything else
    merged = []
    for eid, ex in easy_examples.items():
        cat = ex.get('category', '')
        if cat in args.hard_categories and eid in hard_examples:
            merged.append(hard_examples[eid])
        else:
            merged.append(ex)

    # Add any hard examples not in easy set
    for eid, ex in hard_examples.items():
        if eid not in easy_examples:
            merged.append(ex)

    # Write output
    with open(args.output, 'w') as f:
        for ex in merged:
            f.write(json.dumps(ex) + '\n')

    # Stats
    from collections import Counter
    cats = Counter(ex.get('category', 'UNKNOWN') for ex in merged)
    print(f"Merged {len(merged)} examples:")
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")
    print(f"Output: {args.output}")


if __name__ == '__main__':
    main()
