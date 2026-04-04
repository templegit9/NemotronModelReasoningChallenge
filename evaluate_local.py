"""
Local evaluation using vLLM — matches Kaggle inference settings exactly.

Usage:
    python evaluate_local.py --adapter /workspace/adapter --data /workspace/repo/data/val_split.csv
"""

import argparse
import csv
import math
import re
from collections import Counter, defaultdict

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

MODEL_PATH = "/workspace/nemotron"
BOXED_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'


def categorize_prompt(prompt):
    prompt_lower = prompt.lower()
    if "roman numeral" in prompt_lower or "write the number" in prompt_lower:
        return "NUMBER_SYSTEM"
    elif "unit" in prompt_lower and "convert" in prompt_lower:
        return "UNIT_CONVERSION"
    elif "falling distance" in prompt_lower or "free fall" in prompt_lower or "distance =" in prompt_lower:
        return "PHYSICS"
    elif "decrypt" in prompt_lower or "encrypt" in prompt_lower:
        return "TEXT_ENCRYPTION"
    elif "bit manipulation" in prompt_lower or ("8-bit" in prompt_lower and "binary" in prompt_lower):
        return "BIT_MANIPULATION"
    elif "transformation rules" in prompt_lower or "symbol" in prompt_lower:
        return "SYMBOL_TRANSFORM"
    return "UNKNOWN"


def extract_answer(text):
    """Extract answer from model output, matching Kaggle metric logic."""
    # Priority 1: \boxed{...}
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()

    # Priority 2: last number
    numbers = re.findall(r'-?\d+\.?\d*', text)
    if numbers:
        return numbers[-1]

    return text.strip().split('\n')[-1].strip() if text.strip() else ""


def answers_match(predicted, ground_truth, tolerance=1e-2):
    if predicted is None:
        return False
    if predicted.strip().lower() == ground_truth.strip().lower():
        return True
    try:
        pred_num = float(predicted)
        gt_num = float(ground_truth)
        if gt_num == 0:
            return abs(pred_num) < 1e-5
        return abs(pred_num - gt_num) / abs(gt_num) < tolerance
    except (ValueError, ZeroDivisionError):
        pass
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adapter', required=True, help='Path to LoRA adapter')
    parser.add_argument('--data', required=True, help='Path to validation CSV')
    parser.add_argument('--model', default=MODEL_PATH, help='Base model path')
    parser.add_argument('--limit', type=int, default=0, help='Limit examples (0=all)')
    args = parser.parse_args()

    # Load validation data
    with open(args.data) as f:
        rows = list(csv.DictReader(f))
    if 'category' not in rows[0]:
        for row in rows:
            row['category'] = categorize_prompt(row['prompt'])
    if args.limit > 0:
        rows = rows[:args.limit]

    print(f"Loaded {len(rows)} validation examples")
    cats = Counter(r['category'] for r in rows)
    for cat, count in sorted(cats.items()):
        print(f"  {cat}: {count}")

    # Build prompts matching Kaggle format
    prompts = []
    for row in rows:
        messages = [
            {"role": "system", "content": ""},
            {"role": "user", "content": row['prompt'] + BOXED_SUFFIX},
        ]
        prompts.append(messages)

    # Initialize vLLM with LoRA
    print(f"\nLoading model from {args.model} with LoRA from {args.adapter}...")
    llm = LLM(
        model=args.model,
        enable_lora=True,
        max_lora_rank=32,
        max_num_seqs=64,
        gpu_memory_utilization=0.85,
        max_model_len=8192,
        trust_remote_code=True,
        dtype="bfloat16",
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=7680,
    )

    lora_request = LoRARequest("nemotron_adapter", 1, args.adapter)

    # Format prompts using tokenizer
    tokenizer = llm.get_tokenizer()
    formatted_prompts = []
    for msgs in prompts:
        text = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        formatted_prompts.append(text)

    # Generate
    print(f"Generating {len(formatted_prompts)} responses...")
    outputs = llm.generate(formatted_prompts, sampling_params, lora_request=lora_request)

    # Evaluate
    results = defaultdict(lambda: {"correct": 0, "total": 0, "errors": []})
    total_correct = 0

    for row, output in zip(rows, outputs):
        generated_text = output.outputs[0].text
        predicted = extract_answer(generated_text)
        ground_truth = row['answer']
        category = row['category']

        correct = answers_match(predicted, ground_truth)
        results[category]["total"] += 1
        if correct:
            results[category]["correct"] += 1
            total_correct += 1
        else:
            if len(results[category]["errors"]) < 3:
                results[category]["errors"].append({
                    "id": row['id'],
                    "predicted": predicted,
                    "expected": ground_truth,
                    "output_preview": generated_text[:200],
                })

    # Report
    total = len(rows)
    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {total_correct}/{total} ({100*total_correct/total:.2f}%)")
    print(f"{'='*60}")
    print(f"\nPer-category breakdown:")
    print(f"{'Category':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print(f"{'-'*48}")
    for cat in sorted(results.keys()):
        r = results[cat]
        acc = 100 * r['correct'] / r['total'] if r['total'] > 0 else 0
        print(f"{cat:<20} {r['correct']:>8} {r['total']:>8} {acc:>9.1f}%")

    print(f"\nSample errors per category:")
    for cat in sorted(results.keys()):
        errors = results[cat]["errors"]
        if errors:
            print(f"\n  {cat}:")
            for e in errors[:2]:
                print(f"    ID: {e['id']}")
                print(f"    Expected: {e['expected']}")
                print(f"    Got: {e['predicted']}")
                print(f"    Output: {e['output_preview'][:100]}...")


if __name__ == "__main__":
    main()
