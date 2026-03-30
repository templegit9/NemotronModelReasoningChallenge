#!/usr/bin/env python3
"""
Autoresearch Loop — Autonomous experiment runner.

Uses Claude API to iteratively modify train.py, run experiments,
evaluate results, and log findings. Designed for RTX 5070 Ti (16GB).

Usage:
    export ANTHROPIC_API_KEY="your-key-here"
    python run_loop.py [--max-experiments 50] [--model claude-sonnet-4-20250514]

Each iteration:
1. Agent reads program.md + results.tsv + current train.py
2. Agent proposes a modification to train.py
3. Experiment runs (train + evaluate)
4. Results are logged to results.tsv
5. Repeat
"""

import argparse
import csv
import datetime
import os
import shutil
import subprocess
import sys
import time
import traceback

import anthropic


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(SCRIPT_DIR, "train.py")
EVALUATE_PY = os.path.join(SCRIPT_DIR, "evaluate.py")
PROGRAM_MD = os.path.join(SCRIPT_DIR, "program.md")
RESULTS_TSV = os.path.join(SCRIPT_DIR, "results.tsv")
BACKUP_DIR = os.path.join(SCRIPT_DIR, "backups")


def read_file(path):
    with open(path) as f:
        return f.read()


def write_file(path, content):
    with open(path, 'w') as f:
        f.write(content)


def load_results():
    if not os.path.exists(RESULTS_TSV):
        return "No experiments run yet."
    with open(RESULTS_TSV) as f:
        return f.read()


def append_result(experiment_num, description, overall_acc, category_accs,
                  train_loss, train_time, status):
    """Append a row to results.tsv."""
    file_exists = os.path.exists(RESULTS_TSV) and os.path.getsize(RESULTS_TSV) > 0
    with open(RESULTS_TSV, 'a') as f:
        if not file_exists:
            headers = [
                "exp", "timestamp", "description", "overall_acc",
                "NUMBER_SYSTEM", "UNIT_CONVERSION", "PHYSICS",
                "TEXT_ENCRYPTION", "BIT_MANIPULATION", "SYMBOL_TRANSFORM",
                "train_loss", "train_time_min", "status"
            ]
            f.write("\t".join(headers) + "\n")

        cats = ["NUMBER_SYSTEM", "UNIT_CONVERSION", "PHYSICS",
                "TEXT_ENCRYPTION", "BIT_MANIPULATION", "SYMBOL_TRANSFORM"]
        cat_values = [f"{category_accs.get(c, 0):.4f}" for c in cats]

        row = [
            str(experiment_num),
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            description,
            f"{overall_acc:.4f}" if overall_acc is not None else "N/A",
            *cat_values,
            f"{train_loss:.4f}" if train_loss is not None else "N/A",
            f"{train_time:.1f}" if train_time is not None else "N/A",
            status,
        ]
        f.write("\t".join(row) + "\n")


def backup_train_py(experiment_num):
    """Save a copy of the current train.py before modification."""
    os.makedirs(BACKUP_DIR, exist_ok=True)
    dst = os.path.join(BACKUP_DIR, f"train_exp{experiment_num:03d}.py")
    shutil.copy2(TRAIN_PY, dst)
    return dst


def get_agent_modification(client, model, experiment_num, program, results, current_train):
    """Ask Claude to propose a modification to train.py."""
    prompt = f"""You are an ML research agent running automated experiments to optimize LoRA fine-tuning.

## Research Program
{program}

## Past Results
{results}

## Current train.py
```python
{current_train}
```

## Your Task
This is experiment #{experiment_num}. Based on the past results (or lack thereof), propose a SINGLE focused modification to train.py.

Respond with EXACTLY this format:
1. First line: A short description of what you're changing and why (max 100 chars)
2. Then the COMPLETE modified train.py file wrapped in ```python ... ```

Rules:
- Make ONE focused change at a time
- Keep the overall structure intact
- The script must remain runnable as `python train.py`
- Must complete in under 15 minutes on RTX 5070 Ti (16GB VRAM)
- LoRA rank must be <= 32
- Only import from standard libraries, torch, transformers, peft, numpy
- The prepare_data import must stay as-is
"""

    response = client.messages.create(
        model=model,
        max_tokens=8000,
        messages=[{"role": "user", "content": prompt}],
    )

    text = response.content[0].text
    usage = response.usage

    # Parse description (first line) and code
    lines = text.strip().split('\n')
    description = lines[0].strip().rstrip('.')[:100]

    # Extract python code block
    code_match = text.split('```python')
    if len(code_match) < 2:
        raise ValueError("Agent response did not contain a ```python code block")
    code = code_match[1].split('```')[0].strip()

    return description, code, usage


def run_experiment():
    """Run train.py and return results."""
    print("  Running training...")
    result = subprocess.run(
        [sys.executable, TRAIN_PY],
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        timeout=1200,  # 20 min hard timeout
    )

    if result.returncode != 0:
        print(f"  TRAINING FAILED:\n{result.stderr[-1000:]}")
        return None, None

    # Parse training output for loss and time
    train_loss = None
    train_time = None
    for line in result.stdout.split('\n'):
        if 'Avg Loss:' in line:
            try:
                train_loss = float(line.split('Avg Loss:')[1].split('|')[0].strip())
            except (ValueError, IndexError):
                pass
        if 'Time:' in line and 'minutes' in line.lower():
            try:
                train_time = float(line.split('Time:')[1].split('minute')[0].strip())
            except (ValueError, IndexError):
                pass

    print(f"  Training done. Loss: {train_loss}, Time: {train_time}min")
    print(result.stdout[-500:])

    return train_loss, train_time


def run_evaluation():
    """Run evaluate.py and return results."""
    print("  Running evaluation...")
    result = subprocess.run(
        [sys.executable, EVALUATE_PY, "Qwen/Qwen2.5-3B-Instruct", "/tmp/autoresearch_adapter"],
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min timeout for eval
    )

    if result.returncode != 0:
        print(f"  EVALUATION FAILED:\n{result.stderr[-1000:]}")
        return None, {}

    # Parse evaluation output
    overall_acc = None
    category_accs = {}

    for line in result.stdout.split('\n'):
        if 'OVERALL ACCURACY:' in line:
            try:
                overall_acc = float(line.split('OVERALL ACCURACY:')[1].split('(')[0].strip())
            except (ValueError, IndexError):
                pass
        # Parse per-category lines like "  NUMBER_SYSTEM      : 0.9500 (19/20)"
        for cat in ["NUMBER_SYSTEM", "UNIT_CONVERSION", "PHYSICS",
                     "TEXT_ENCRYPTION", "BIT_MANIPULATION", "SYMBOL_TRANSFORM"]:
            if cat in line and ':' in line:
                try:
                    acc = float(line.split(':')[1].split('(')[0].strip())
                    category_accs[cat] = acc
                except (ValueError, IndexError):
                    pass

    print(f"  Evaluation done. Overall: {overall_acc}")
    print(result.stdout[-800:])

    return overall_acc, category_accs


def main():
    parser = argparse.ArgumentParser(description="Autoresearch experiment loop")
    parser.add_argument("--max-experiments", type=int, default=50,
                        help="Maximum number of experiments to run")
    parser.add_argument("--model", type=str, default="claude-sonnet-4-20250514",
                        help="Claude model to use for the agent")
    parser.add_argument("--start-from", type=int, default=1,
                        help="Starting experiment number")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: Set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    total_input_tokens = 0
    total_output_tokens = 0

    print(f"{'='*60}")
    print(f"AUTORESEARCH LOOP")
    print(f"Model: {args.model}")
    print(f"Max experiments: {args.max_experiments}")
    print(f"{'='*60}\n")

    for exp_num in range(args.start_from, args.start_from + args.max_experiments):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT #{exp_num}")
        print(f"{'='*60}")

        try:
            # Read current state
            program = read_file(PROGRAM_MD)
            results = load_results()
            current_train = read_file(TRAIN_PY)

            # Backup current train.py
            backup_train_py(exp_num)

            # Get agent's modification
            print("  Asking agent for modification...")
            description, new_code, usage = get_agent_modification(
                client, args.model, exp_num, program, results, current_train
            )
            total_input_tokens += usage.input_tokens
            total_output_tokens += usage.output_tokens

            print(f"  Agent says: {description}")
            print(f"  Tokens — input: {usage.input_tokens}, output: {usage.output_tokens}")
            print(f"  Cumulative — input: {total_input_tokens}, output: {total_output_tokens}")

            # Apply modification
            write_file(TRAIN_PY, new_code)

            # Run experiment
            train_loss, train_time = run_experiment()
            if train_loss is None:
                append_result(exp_num, description, None, {}, None, None, "TRAIN_FAILED")
                # Restore previous train.py
                backup = os.path.join(BACKUP_DIR, f"train_exp{exp_num:03d}.py")
                shutil.copy2(backup, TRAIN_PY)
                print("  Restored previous train.py")
                continue

            # Evaluate
            overall_acc, category_accs = run_evaluation()
            if overall_acc is None:
                append_result(exp_num, description, None, {}, train_loss, train_time, "EVAL_FAILED")
                backup = os.path.join(BACKUP_DIR, f"train_exp{exp_num:03d}.py")
                shutil.copy2(backup, TRAIN_PY)
                print("  Restored previous train.py")
                continue

            # Log results
            append_result(exp_num, description, overall_acc, category_accs,
                          train_loss, train_time, "OK")

            print(f"\n  RESULT: {overall_acc:.4f} overall accuracy")
            print(f"  Description: {description}")

        except KeyboardInterrupt:
            print("\n\nStopped by user.")
            break
        except Exception as e:
            print(f"  ERROR: {e}")
            traceback.print_exc()
            append_result(exp_num, f"ERROR: {str(e)[:80]}", None, {}, None, None, "ERROR")
            # Restore on error
            backup = os.path.join(BACKUP_DIR, f"train_exp{exp_num:03d}.py")
            if os.path.exists(backup):
                shutil.copy2(backup, TRAIN_PY)
            continue

    print(f"\n{'='*60}")
    print(f"AUTORESEARCH COMPLETE")
    print(f"Total API tokens — input: {total_input_tokens}, output: {total_output_tokens}")
    print(f"Results saved to: {RESULTS_TSV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
