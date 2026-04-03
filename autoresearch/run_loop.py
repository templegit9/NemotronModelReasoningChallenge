#!/usr/bin/env python3
"""
Autoresearch Loop — Autonomous experiment runner.

Uses Claude (claude-haiku-4-5) to iteratively modify train.py, run experiments,
evaluate results, and log findings. Designed for RTX 5070 Ti (16GB).

Usage:
    python run_loop.py [--max-experiments 50] [--start-from 1]

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
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback

try:
    import anthropic
except ImportError:
    print("ERROR: anthropic package not installed. Run: pip install anthropic")
    sys.exit(1)

# Load .env from repo root if present
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
if os.path.exists(_env_path):
    with open(_env_path, encoding='utf-8') as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith('#') and '=' in _line:
                _k, _v = _line.split('=', 1)
                os.environ.setdefault(_k.strip(), _v.strip())


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PY = os.path.join(SCRIPT_DIR, "train.py")
EVALUATE_PY = os.path.join(SCRIPT_DIR, "evaluate.py")
PROGRAM_MD = os.path.join(SCRIPT_DIR, "program.md")
RESULTS_TSV = os.path.join(SCRIPT_DIR, "results.tsv")
BACKUP_DIR = os.path.join(SCRIPT_DIR, "backups")
STATUS_JSON = os.path.join(SCRIPT_DIR, "status.json")
LOCK_FILE = os.path.join(SCRIPT_DIR, "run_loop.lock")

CLAUDE_MODEL = "claude-sonnet-4-6"


# ── helpers ──────────────────────────────────────────────────────────────────

def write_status(phase, exp_num, description="", plan="", max_experiments=0, started_at="", exp_started_at=""):
    try:
        with open(STATUS_JSON, 'w', encoding='utf-8') as f:
            json.dump({
                "phase": phase,
                "plan": plan,
                "exp_num": exp_num,
                "max_experiments": max_experiments,
                "description": description,
                "started_at": started_at,
                "exp_started_at": exp_started_at,
                "updated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }, f)
    except Exception:
        pass


def read_file(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def load_results():
    if not os.path.exists(RESULTS_TSV):
        return "No experiments run yet."
    with open(RESULTS_TSV, encoding='utf-8') as f:
        lines = f.readlines()
    # Keep header + last 10 rows to control token usage
    if len(lines) > 11:
        lines = lines[:1] + lines[-10:]
    return ''.join(lines)


def append_result(experiment_num, description, overall_acc, category_accs,
                  train_loss, train_time, status):
    file_exists = os.path.exists(RESULTS_TSV) and os.path.getsize(RESULTS_TSV) > 0
    with open(RESULTS_TSV, 'a', encoding='utf-8') as f:
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
    os.makedirs(BACKUP_DIR, exist_ok=True)
    dst = os.path.join(BACKUP_DIR, f"train_exp{experiment_num:03d}.py")
    shutil.copy2(TRAIN_PY, dst)
    return dst


# ── agent ────────────────────────────────────────────────────────────────────

def get_best_backup():
    """Return the backup path of the highest-accuracy OK experiment, or None."""
    if not os.path.exists(RESULTS_TSV):
        return None
    best_acc = -1
    best_exp = None
    with open(RESULTS_TSV, encoding='utf-8') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            if row.get('status') == 'OK':
                try:
                    acc = float(row['overall_acc'])
                    if acc > best_acc:
                        best_acc = acc
                        best_exp = int(row['exp'])
                except (ValueError, KeyError):
                    pass
    if best_exp is None:
        return None
    path = os.path.join(BACKUP_DIR, f"train_exp{best_exp:03d}.py")
    return path if os.path.exists(path) else None


AGENT_REQUEST_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_request.json")
AGENT_RESPONSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_response.json")
AGENT_POLL_INTERVAL = 5    # seconds between checks for response
AGENT_TIMEOUT = 7200       # 2 hours max wait for human response (Nemotron experiments are slow)


def get_agent_modification(experiment_num, program, results, current_train, last_error=None):
    """Write a request file and wait for Claude Code (human) to respond with a modification."""

    # Clear any stale response from a previous round
    if os.path.exists(AGENT_RESPONSE_FILE):
        os.remove(AGENT_RESPONSE_FILE)

    request = {
        "experiment_num": experiment_num,
        "program": program,
        "results": results,
        "current_train": current_train,
        "last_error": last_error,
        "requested_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(AGENT_REQUEST_FILE, 'w', encoding='utf-8') as f:
        json.dump(request, f, indent=2)

    print(f"  Waiting for Claude Code to respond (request written to agent_request.json)...")

    elapsed = 0
    while elapsed < AGENT_TIMEOUT:
        time.sleep(AGENT_POLL_INTERVAL)
        elapsed += AGENT_POLL_INTERVAL
        if os.path.exists(AGENT_RESPONSE_FILE):
            with open(AGENT_RESPONSE_FILE, encoding='utf-8') as f:
                response = json.load(f)
            os.remove(AGENT_RESPONSE_FILE)
            os.remove(AGENT_REQUEST_FILE)
            break
    else:
        if os.path.exists(AGENT_REQUEST_FILE):
            os.remove(AGENT_REQUEST_FILE)
        raise TimeoutError(f"No response after {AGENT_TIMEOUT}s — Claude Code session may be inactive")

    code = response.get("code", "")
    description = response.get("description", "No description")[:150]
    plan = response.get("plan", description)

    # Enforce hard constraints
    code = re.sub(r'BATCH_SIZE\s*=\s*[2-9]\d*', 'BATCH_SIZE = 1', code)
    code = re.sub(r'NUM_EPOCHS\s*=\s*[2-9]\d*', 'NUM_EPOCHS = 1', code)
    code = re.sub(r'MAX_SEQ_LENGTH\s*=\s*\d+[^\n]*',
                  lambda m: f'MAX_SEQ_LENGTH = {min(int(re.search(r"\\d+", m.group(0)).group()), 2048)}', code)
    code = re.sub(
        r'MAX_TRAINING_STEPS\s*=\s*(\d+)',
        lambda m: f'MAX_TRAINING_STEPS = {max(100, min(int(m.group(1)), 200))}',
        code,
    )
    code = re.sub(
        r'LORA_RANK\s*=\s*(\d+)',
        lambda m: f'LORA_RANK = {min(int(m.group(1)), 32)}',
        code,
    )
    code = re.sub(
        r'("[\w_]+")\s*:\s*([2-9](?:\.\d+)?|\d+\.\d+)',
        lambda m: f'{m.group(1)}: {min(float(m.group(2)), 1.5)}' if float(m.group(2)) > 1.5 else m.group(0),
        code,
    )
    if 'enable_input_require_grads' not in code:
        raise ValueError("Response code is missing model.enable_input_require_grads() — rejecting")

    class _Usage:
        input_tokens = 0
        output_tokens = 0
        cache_creation = 0
        cache_read = 0
        cost = 0.0

    return description, plan, code, _Usage()


# ── subprocess helpers ────────────────────────────────────────────────────────

def _kill_proc(proc):
    """Kill a subprocess and drain its pipes to avoid deadlock on Windows."""
    try:
        proc.kill()
    except OSError:
        pass
    try:
        proc.communicate(timeout=30)
    except Exception:
        pass


def run_experiment():
    """Run train.py and return (train_loss, train_time_minutes, stderr) or (None, None, stderr)."""
    print("  Running training...")
    train_env = os.environ.copy()
    train_env.update({
        "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:256,garbage_collection_threshold:0.8",
        "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONIOENCODING": "utf-8",  # prevent UnicodeEncodeError on Windows cp1252
    })
    proc = subprocess.Popen(
        [sys.executable, TRAIN_PY],
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
        env=train_env,
    )
    try:
        stdout, stderr = proc.communicate(timeout=14400)  # 4 hour hard limit (Nemotron on A100)
    except subprocess.TimeoutExpired:
        print("  TRAINING TIMED OUT — killing process...")
        _kill_proc(proc)
        return None, None, "Training timed out after 4 hours"

    if proc.returncode != 0:
        print(f"  TRAINING FAILED:\n{stderr[-1000:]}")
        return None, None, stderr[-1000:]

    train_loss = None
    train_time = None
    for line in stdout.split('\n'):
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
    print(stdout[-500:])
    return train_loss, train_time, None


def run_evaluation():
    """Run evaluate.py and return (overall_acc, category_accs) or (None, {})."""
    print("  Running evaluation...")
    eval_proc = subprocess.Popen(
        [sys.executable, EVALUATE_PY,
         os.environ.get("MODEL_PATH", "/workspace/nemotron"),
         os.environ.get("ADAPTER_PATH", "/workspace/adapter"),
         "128"],   # max_new_tokens=128: enough for answers + short reasoning; 512 causes timeout
        cwd=SCRIPT_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding='utf-8',
        errors='replace',
    )
    try:
        eval_stdout, eval_stderr = eval_proc.communicate(timeout=7200)  # 2 hours — 947 examples on 30B Nemotron
    except subprocess.TimeoutExpired:
        print("  EVALUATION TIMED OUT — killing process...")
        _kill_proc(eval_proc)
        return None, {}

    if eval_proc.returncode != 0:
        print(f"  EVALUATION FAILED:\n{eval_stderr[-1000:]}")
        return None, {}

    overall_acc = None
    category_accs = {}
    for line in eval_stdout.split('\n'):
        if 'OVERALL ACCURACY:' in line:
            try:
                overall_acc = float(line.split('OVERALL ACCURACY:')[1].split('(')[0].strip())
            except (ValueError, IndexError):
                pass
        for cat in ["NUMBER_SYSTEM", "UNIT_CONVERSION", "PHYSICS",
                    "TEXT_ENCRYPTION", "BIT_MANIPULATION", "SYMBOL_TRANSFORM"]:
            if cat in line and ':' in line:
                try:
                    acc = float(line.split(':')[1].split('(')[0].strip())
                    category_accs[cat] = acc
                except (ValueError, IndexError):
                    pass

    print(f"  Evaluation done. Overall: {overall_acc}")
    print(eval_stdout[-800:])
    return overall_acc, category_accs


# ── main loop ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Autoresearch experiment loop")
    parser.add_argument("--max-experiments", type=int, default=50)
    parser.add_argument("--start-from", type=int, default=1)
    args = parser.parse_args()

    # Lock file — prevent duplicate instances
    if os.path.exists(LOCK_FILE):
        try:
            pid = int(open(LOCK_FILE).read().strip())
            print(f"ERROR: Another loop instance is already running (PID {pid}).")
            print(f"  If it crashed, delete the lock file and retry:")
            print(f"  del {LOCK_FILE}")
        except Exception:
            print(f"ERROR: Lock file exists at {LOCK_FILE}. Delete it if no loop is running.")
        sys.exit(1)

    with open(LOCK_FILE, 'w') as f:
        f.write(str(os.getpid()))

    total_input_tokens = 0
    total_output_tokens = 0
    total_cost = 0.0
    run_started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    consecutive_failures = 0
    last_error = None

    print(f"{'='*60}")
    print(f"AUTORESEARCH LOOP")
    print(f"Model: {CLAUDE_MODEL}")
    print(f"Max experiments: {args.max_experiments}")
    print(f"{'='*60}\n")

    try:
        for exp_num in range(args.start_from, args.start_from + args.max_experiments):
            print(f"\n{'='*60}")
            print(f"EXPERIMENT #{exp_num}")
            print(f"{'='*60}")
            exp_started_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            try:
                # Auto-reset to best baseline after 3 consecutive failures
                if consecutive_failures >= 3:
                    best_backup = get_best_backup()
                    if best_backup:
                        shutil.copy2(best_backup, TRAIN_PY)
                        print(f"  AUTO-RESET: {consecutive_failures} consecutive failures — restored best backup: {os.path.basename(best_backup)}")
                        consecutive_failures = 0
                        last_error = None

                program = read_file(PROGRAM_MD)
                results = load_results()
                current_train = read_file(TRAIN_PY)

                backup_train_py(exp_num)

                print("  Asking Claude for modification...")
                write_status("asking_agent", exp_num, max_experiments=args.max_experiments,
                             started_at=run_started_at, exp_started_at=exp_started_at)

                description, plan, new_code, usage = get_agent_modification(
                    exp_num, program, results, current_train, last_error=last_error
                )

                total_input_tokens += usage.input_tokens
                total_output_tokens += usage.output_tokens
                total_cost += usage.cost

                print(f"  Agent says: {description}")
                print(f"  Tokens — in: {usage.input_tokens}, out: {usage.output_tokens} | cost: ${usage.cost:.4f}")
                print(f"  Cumulative — in: {total_input_tokens}, out: {total_output_tokens} | total cost: ${total_cost:.4f}")

                # Syntax-check before writing — catches 80% of agent errors instantly
                try:
                    compile(new_code, "train.py", "exec")
                except SyntaxError as e:
                    last_error = f"SyntaxError in generated code: {e}"
                    consecutive_failures += 1
                    print(f"  SYNTAX ERROR in agent code: {e} — skipping, restoring previous train.py")
                    append_result(exp_num, description, None, {}, None, None, "TRAIN_FAILED")
                    write_status("idle", exp_num, description, max_experiments=args.max_experiments,
                                 started_at=run_started_at, exp_started_at=exp_started_at)
                    continue

                write_file(TRAIN_PY, new_code)

                write_status("training", exp_num, description, plan=plan, max_experiments=args.max_experiments,
                             started_at=run_started_at, exp_started_at=exp_started_at)
                train_loss, train_time, train_stderr = run_experiment()

                if train_loss is None:
                    last_error = train_stderr or "Training subprocess returned non-zero exit code"
                    consecutive_failures += 1
                    append_result(exp_num, description, None, {}, None, None, "TRAIN_FAILED")
                    backup = os.path.join(BACKUP_DIR, f"train_exp{exp_num:03d}.py")
                    shutil.copy2(backup, TRAIN_PY)
                    print("  Restored previous train.py")
                    write_status("idle", exp_num, description, max_experiments=args.max_experiments,
                                 started_at=run_started_at, exp_started_at=exp_started_at)
                    continue

                write_status("evaluating", exp_num, description, max_experiments=args.max_experiments,
                             started_at=run_started_at, exp_started_at=exp_started_at)
                overall_acc, category_accs = run_evaluation()

                if overall_acc is None:
                    last_error = "Evaluation failed — model output may not contain \\boxed{} format"
                    consecutive_failures += 1
                    append_result(exp_num, description, None, {}, train_loss, train_time, "EVAL_FAILED")
                    backup = os.path.join(BACKUP_DIR, f"train_exp{exp_num:03d}.py")
                    shutil.copy2(backup, TRAIN_PY)
                    print("  Restored previous train.py")
                    write_status("idle", exp_num, description, max_experiments=args.max_experiments,
                                 started_at=run_started_at, exp_started_at=exp_started_at)
                    continue

                consecutive_failures = 0
                last_error = None
                append_result(exp_num, description, overall_acc, category_accs,
                              train_loss, train_time, "OK")
                write_status("idle", exp_num, description, max_experiments=args.max_experiments,
                             started_at=run_started_at, exp_started_at=exp_started_at)

                print(f"\n  RESULT: {overall_acc:.4f} overall accuracy")
                for cat, acc in category_accs.items():
                    print(f"    {cat}: {acc:.4f}")

            except KeyboardInterrupt:
                print("\n\nStopped by user.")
                break
            except anthropic.AuthenticationError as e:
                print(f"\n  FATAL: Claude API authentication failed — check ANTHROPIC_API_KEY ({e})")
                break
            except anthropic.RateLimitError as e:
                print(f"\n  WARNING: Rate limited. Waiting 60s... ({e})")
                time.sleep(60)
                continue
            except anthropic.APIError as e:
                print(f"\n  API ERROR: {e}")
                append_result(exp_num, f"API_ERROR: {str(e)[:80]}", None, {}, None, None, "ERROR")
                time.sleep(10)
                continue
            except Exception as e:
                print(f"  ERROR: {e}")
                traceback.print_exc()
                last_error = f"{type(e).__name__}: {e}"
                consecutive_failures += 1
                append_result(exp_num, f"ERROR: {str(e)[:80]}", None, {}, None, None, "ERROR")
                backup = os.path.join(BACKUP_DIR, f"train_exp{exp_num:03d}.py")
                if os.path.exists(backup):
                    shutil.copy2(backup, TRAIN_PY)
                continue

    finally:
        if os.path.exists(LOCK_FILE):
            os.remove(LOCK_FILE)

    print(f"\n{'='*60}")
    print(f"AUTORESEARCH COMPLETE")
    print(f"Total tokens — in: {total_input_tokens}, out: {total_output_tokens}")
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Results: {RESULTS_TSV}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
