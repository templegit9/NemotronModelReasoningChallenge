"""
Distill high-quality Chain-of-Thought reasoning from Claude for all training examples.

For each puzzle:
1. Send to Claude with instructions to solve step-by-step
2. If Claude's answer matches ground truth -> use that CoT (verified)
3. If Claude's answer is wrong -> re-prompt with the correct answer to generate explanatory CoT
4. Save results as JSONL for training

Usage:
    python distill_cot.py --input data/train_split.csv --output data/distilled_train.jsonl
    python distill_cot.py --input data/val_split.csv --output data/distilled_val.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
import time
from pathlib import Path

import anthropic

# ---------- Config ----------
MODEL = "claude-sonnet-4-20250514"
MAX_CONCURRENT = 30  # Claude API concurrency limit
TEMPERATURE_SOLVE = 0.3  # Low temp for solving
TEMPERATURE_EXPLAIN = 0.5  # Slightly higher for explanation variety
MAX_TOKENS_SOLVE = 4096
MAX_TOKENS_EXPLAIN = 3000
RETRY_ATTEMPTS = 2

BOXED_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'

# ---------- Prompts ----------

SOLVE_PROMPT = """You are solving an Alice's Wonderland reasoning puzzle. You MUST solve it correctly.

{puzzle_text}

Think through this step by step:
1. Carefully examine ALL the input->output examples provided
2. Identify the exact transformation rule by testing hypotheses against every example
3. Verify your rule works for ALL examples before applying it
4. Apply the rule to the target input
5. Double-check your answer

Show your complete reasoning, then put your final answer in \\boxed{{}} format.
IMPORTANT: Your answer inside \\boxed{{}} must be ONLY the final value, no extra text."""

EXPLAIN_PROMPT = """You are explaining how to solve an Alice's Wonderland reasoning puzzle. The correct answer is known.

{puzzle_text}

The correct answer is: {answer}

Generate a detailed step-by-step chain-of-thought explanation showing how to arrive at this answer:
1. Examine the input->output examples
2. Identify the transformation pattern/rule
3. Show how the rule applies to each example for verification
4. Apply the rule to the target to get the answer: {answer}

Be thorough but concise. Show your work clearly."""

# Category-specific solve prompts for better accuracy
CATEGORY_PROMPTS = {
    "NUMBER_SYSTEM": """You are solving a number system conversion puzzle.

{puzzle_text}

This involves converting a number to Roman numerals (or similar number system).
Work through it step by step:
1. Identify what number needs to be converted
2. Break it down into place values
3. Convert each component
4. Combine the result

Put your final answer in \\boxed{{}} format.""",

    "UNIT_CONVERSION": """You are solving a unit conversion puzzle.

{puzzle_text}

Steps:
1. Extract the conversion examples (input measurement -> output measurement)
2. Calculate the conversion factor from EACH example
3. Verify all factors are consistent
4. Apply the factor to the target measurement
5. Round to match the format shown in examples

Put your final answer in \\boxed{{}} format.""",

    "PHYSICS": """You are solving a physics free-fall puzzle.

{puzzle_text}

This uses the equation d = 0.5 * g * t^2 where g is gravitational acceleration.
Steps:
1. From each example (t, d), compute g = 2*d/t^2
2. Average the g values
3. Apply d = 0.5 * g * t^2 for the target time
4. Round appropriately

Put your final answer in \\boxed{{}} format.""",

    "TEXT_ENCRYPTION": """You are solving a cipher/encryption puzzle.

{puzzle_text}

This is a monoalphabetic substitution cipher.
Steps:
1. From each plaintext->ciphertext pair, build a character mapping
2. For each position, map the encrypted character to the decrypted one
3. Apply the mapping to decrypt the target text
4. If any character has no mapping, keep it unchanged or mark with '?'

Put your final answer in \\boxed{{}} format.""",

    "BIT_MANIPULATION": """You are solving a bit manipulation puzzle with 8-bit binary numbers.

{puzzle_text}

CRITICAL: You must find the EXACT rule that transforms ALL inputs to their outputs.
Approach:
1. Convert each input/output to decimal for easier analysis
2. Try these operations systematically:
   - XOR with a constant
   - NOT (flip all bits)
   - Bit rotation (left or right by N positions)
   - Bit shifts
   - Bit reversal
   - Combinations: e.g., rotate then XOR, reverse then NOT, etc.
3. For each hypothesis, verify against ALL examples
4. Once you find a rule that works for ALL examples, apply it to the target

Show your binary analysis step by step. Put your final answer (8-bit binary) in \\boxed{{}} format.""",

    "SYMBOL_TRANSFORM": """You are solving a symbol transformation puzzle.

{puzzle_text}

CRITICAL: Find the exact transformation rule applied to the input to produce the output.
Common patterns in these puzzles:
- Character deletion (removing specific characters)
- Character substitution (mapping one char to another)
- Positional operations (swapping, reversing, extracting positions)
- For math-like expressions: the operator might be reinterpreted
- String concatenation or rearrangement rules

Approach:
1. Compare each input to its output character by character
2. Look for consistent patterns across ALL examples
3. Test your hypothesis against every example
4. Apply to the target

Put your final answer in \\boxed{{}} format.""",
}


def extract_boxed_answer(text: str) -> str | None:
    """Extract answer from \\boxed{...} in generated text."""
    # Find the last \boxed{...} in the text
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def answers_match(predicted: str, ground_truth: str, tolerance: float = 1e-2) -> bool:
    """Check if predicted answer matches ground truth (string or numeric)."""
    if predicted is None:
        return False

    # Exact string match (case-insensitive)
    if predicted.strip().lower() == ground_truth.strip().lower():
        return True

    # Try numeric comparison
    try:
        pred_num = float(predicted)
        gt_num = float(ground_truth)
        if gt_num == 0:
            return abs(pred_num) < 1e-5
        return abs(pred_num - gt_num) / abs(gt_num) < tolerance
    except (ValueError, ZeroDivisionError):
        pass

    return False


def categorize_prompt(prompt: str) -> str:
    """Auto-detect puzzle category from prompt text."""
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


async def solve_puzzle(client: anthropic.AsyncAnthropic, puzzle_text: str, category: str, semaphore: asyncio.Semaphore) -> tuple[str, str | None]:
    """Ask Claude to solve a puzzle. Returns (reasoning, extracted_answer)."""
    prompt_template = CATEGORY_PROMPTS.get(category, SOLVE_PROMPT)
    user_msg = prompt_template.format(puzzle_text=puzzle_text)

    async with semaphore:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS_SOLVE,
                    temperature=TEMPERATURE_SOLVE,
                    messages=[{"role": "user", "content": user_msg}],
                )
                text = response.content[0].text
                answer = extract_boxed_answer(text)
                # Remove the \boxed{} from reasoning to keep it clean
                reasoning = re.sub(r'\\boxed\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', '', text).strip()
                return reasoning, answer
            except anthropic.RateLimitError:
                await asyncio.sleep(5 * (attempt + 1))
            except Exception as e:
                if attempt == RETRY_ATTEMPTS - 1:
                    print(f"  Error solving: {e}")
                    return None, None
                await asyncio.sleep(2)
    return None, None


async def explain_answer(client: anthropic.AsyncAnthropic, puzzle_text: str, answer: str, category: str, semaphore: asyncio.Semaphore) -> str | None:
    """Ask Claude to explain how to arrive at a known answer. Returns reasoning."""
    user_msg = EXPLAIN_PROMPT.format(puzzle_text=puzzle_text, answer=answer)

    async with semaphore:
        for attempt in range(RETRY_ATTEMPTS):
            try:
                response = await client.messages.create(
                    model=MODEL,
                    max_tokens=MAX_TOKENS_EXPLAIN,
                    temperature=TEMPERATURE_EXPLAIN,
                    messages=[{"role": "user", "content": user_msg}],
                )
                return response.content[0].text.strip()
            except anthropic.RateLimitError:
                await asyncio.sleep(5 * (attempt + 1))
            except Exception as e:
                if attempt == RETRY_ATTEMPTS - 1:
                    print(f"  Error explaining: {e}")
                    return None
                await asyncio.sleep(2)
    return None


async def process_example(
    client: anthropic.AsyncAnthropic,
    row: dict,
    semaphore: asyncio.Semaphore,
    stats: dict,
) -> dict | None:
    """Process a single training example through the distillation pipeline."""
    puzzle_id = row['id']
    puzzle_text = row['prompt']
    ground_truth = row['answer']
    category = row.get('category', categorize_prompt(puzzle_text))

    # Step 1: Try to solve
    reasoning, predicted = await solve_puzzle(client, puzzle_text, category, semaphore)

    if reasoning and answers_match(predicted, ground_truth):
        # Claude solved it correctly - use this verified CoT
        stats['correct'] += 1
        stats[f'{category}_correct'] = stats.get(f'{category}_correct', 0) + 1
        cot = reasoning
    else:
        # Claude got it wrong or failed - ask for explanation with correct answer
        stats['wrong'] += 1
        stats[f'{category}_wrong'] = stats.get(f'{category}_wrong', 0) + 1
        explanation = await explain_answer(client, puzzle_text, ground_truth, category, semaphore)
        if explanation:
            cot = explanation
        elif reasoning:
            # Fallback: use Claude's reasoning but with correct answer
            cot = reasoning
        else:
            # Total failure - use minimal CoT
            cot = f"Analyzing the transformation pattern from the examples and applying it to the target gives us the answer."

    # Format as training example
    user_content = puzzle_text + BOXED_SUFFIX
    assistant_content = f"<think>\n{cot}\n</think>\n\\boxed{{{ground_truth}}}"

    return {
        'id': puzzle_id,
        'category': category,
        'solved_correctly': answers_match(predicted, ground_truth) if predicted else False,
        'messages': [
            {'role': 'system', 'content': ''},
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': assistant_content},
        ],
    }


async def process_batch(client, rows, semaphore, stats, batch_num, total_batches):
    """Process a batch of examples concurrently."""
    tasks = [process_example(client, row, semaphore, stats) for row in rows]
    results = await asyncio.gather(*tasks)
    valid = [r for r in results if r is not None]
    total_done = stats['correct'] + stats['wrong']
    print(f"  Batch {batch_num}/{total_batches} done | "
          f"Total: {total_done} | "
          f"Correct: {stats['correct']} ({100*stats['correct']/max(total_done,1):.1f}%) | "
          f"Wrong: {stats['wrong']}")
    return valid


async def main():
    parser = argparse.ArgumentParser(description='Distill CoT from Claude')
    parser.add_argument('--input', required=True, help='Input CSV (train_split.csv or val_split.csv)')
    parser.add_argument('--output', required=True, help='Output JSONL path')
    parser.add_argument('--limit', type=int, default=0, help='Limit number of examples (0=all)')
    parser.add_argument('--category', type=str, default='', help='Only process this category')
    parser.add_argument('--resume', action='store_true', help='Resume from existing output file')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for concurrent processing')
    args = parser.parse_args()

    # Load data
    with open(args.input) as f:
        rows = list(csv.DictReader(f))

    # Auto-categorize if needed
    if 'category' not in rows[0]:
        for row in rows:
            row['category'] = categorize_prompt(row['prompt'])

    # Filter by category if specified
    if args.category:
        rows = [r for r in rows if r['category'] == args.category]
        print(f"Filtered to {len(rows)} examples in category {args.category}")

    # Resume support
    done_ids = set()
    existing_results = []
    if args.resume and Path(args.output).exists():
        with open(args.output) as f:
            for line in f:
                obj = json.loads(line)
                done_ids.add(obj['id'])
                existing_results.append(obj)
        print(f"Resuming: {len(done_ids)} already processed")
        rows = [r for r in rows if r['id'] not in done_ids]

    if args.limit > 0:
        rows = rows[:args.limit]

    print(f"Processing {len(rows)} examples...")
    from collections import Counter
    cat_counts = Counter(r['category'] for r in rows)
    for cat, count in sorted(cat_counts.items()):
        print(f"  {cat}: {count}")

    # Setup
    client = anthropic.AsyncAnthropic()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    stats = {'correct': 0, 'wrong': 0}

    # Process in batches
    all_results = list(existing_results)
    batch_size = args.batch_size
    total_batches = math.ceil(len(rows) / batch_size)

    start_time = time.time()
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        batch_num = i // batch_size + 1
        results = await process_batch(client, batch, semaphore, stats, batch_num, total_batches)
        all_results.extend(results)

        # Save incrementally
        with open(args.output, 'w') as f:
            for r in all_results:
                f.write(json.dumps(r) + '\n')

    elapsed = time.time() - start_time

    # Final stats
    print(f"\n{'='*60}")
    print(f"Distillation complete!")
    print(f"Total: {len(all_results)} examples in {elapsed/60:.1f} minutes")
    print(f"Solve accuracy: {stats['correct']}/{stats['correct']+stats['wrong']} "
          f"({100*stats['correct']/max(stats['correct']+stats['wrong'],1):.1f}%)")
    print(f"\nPer-category solve accuracy:")
    for cat in sorted(set(r.get('category', 'UNKNOWN') for r in rows)):
        correct = stats.get(f'{cat}_correct', 0)
        wrong = stats.get(f'{cat}_wrong', 0)
        total = correct + wrong
        if total > 0:
            print(f"  {cat}: {correct}/{total} ({100*correct/total:.1f}%)")
    print(f"\nOutput saved to: {args.output}")


if __name__ == '__main__':
    asyncio.run(main())
