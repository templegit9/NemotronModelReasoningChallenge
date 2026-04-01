"""Evaluate a trained adapter on the validation set. DO NOT MODIFY."""

import json
import math
import os
import re
import sys
import torch
from collections import Counter, defaultdict
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


BOXED_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'


def extract_answer(text):
    """Extract answer from model output, matching competition metric logic."""
    # Priority 1: \boxed{...}
    boxed = re.findall(r'\\boxed\{([^}]*)\}', text)
    if boxed:
        return boxed[-1].strip()

    # Priority 2: "final answer is: ..."
    final = re.search(r'final answer is:?\s*(.+?)(?:\.|$)', text, re.IGNORECASE)
    if final:
        return final.group(1).strip()

    # Priority 3: last number
    numbers = re.findall(r'[\d.]+', text)
    if numbers:
        return numbers[-1]

    return text.strip().split('\n')[-1].strip()


def verify(predicted, expected):
    """Check if predicted answer matches expected, matching competition logic."""
    pred = predicted.strip()
    exp = expected.strip()

    # Binary strings: exact match
    if re.match(r'^[01]+$', exp):
        return pred == exp

    # Numeric: relative tolerance
    try:
        pred_num = float(pred)
        exp_num = float(exp)
        return math.isclose(pred_num, exp_num, rel_tol=1e-2)
    except (ValueError, OverflowError):
        pass

    # String: case-insensitive
    return pred.lower() == exp.lower()


def evaluate_adapter(model_name, adapter_path, val_data, max_new_tokens=512,
                     temperature=0.0, device="cuda"):
    """Run inference and evaluate accuracy."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )

    if adapter_path and os.path.exists(adapter_path):
        print(f"Loading adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()

    correct = 0
    total = 0
    category_correct = defaultdict(int)
    category_total = defaultdict(int)
    results = []

    print(f"Evaluating {len(val_data)} examples...")
    for i, example in enumerate(val_data):
        category = example['category']
        messages = example['messages']
        expected_answer = messages[2]['content']  # assistant content

        # Extract the ground truth answer from the assistant content
        gt_answer = extract_answer(expected_answer)

        # Build inference prompt (system + user only)
        prompt_messages = [
            {"role": "user", "content": messages[1]['content']},
        ]

        # Apply chat template
        prompt_text = tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        with torch.no_grad():
            if temperature > 0:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    top_p=1.0,
                    pad_token_id=tokenizer.pad_token_id,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )

        generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:],
                                      skip_special_tokens=True)
        predicted_answer = extract_answer(generated)

        is_correct = verify(predicted_answer, gt_answer)
        correct += int(is_correct)
        total += 1
        category_correct[category] += int(is_correct)
        category_total[category] += 1

        results.append({
            'id': example.get('id', i),
            'category': category,
            'expected': gt_answer,
            'predicted': predicted_answer,
            'correct': is_correct,
        })

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(val_data)}] Running accuracy: {correct/total:.4f}")

    # Summary
    overall_acc = correct / total if total > 0 else 0
    print(f"\n{'='*60}")
    print(f"OVERALL ACCURACY: {overall_acc:.4f} ({correct}/{total})")
    print(f"{'='*60}")

    category_accs = {}
    for cat in sorted(category_total.keys()):
        cat_acc = category_correct[cat] / category_total[cat]
        category_accs[cat] = cat_acc
        print(f"  {cat:20s}: {cat_acc:.4f} ({category_correct[cat]}/{category_total[cat]})")

    return {
        'overall_accuracy': overall_acc,
        'category_accuracies': category_accs,
        'correct': correct,
        'total': total,
        'results': results,
    }


if __name__ == "__main__":
    from prepare_data import load_val

    model_name = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen2.5-3B-Instruct"
    adapter_path = sys.argv[2] if len(sys.argv) > 2 else None
    max_new_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 512

    val_data = load_val()
    results = evaluate_adapter(model_name, adapter_path, val_data, max_new_tokens=max_new_tokens)
    print(f"\nDone. Overall: {results['overall_accuracy']:.4f}")
