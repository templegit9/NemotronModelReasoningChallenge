"""Generate formatted training data with chain-of-thought reasoning.

Outputs JSONL with chat messages matching the exact Nemotron chat template format:
  - system: (empty)
  - user: {prompt}\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`
  - assistant: <think>\n{reasoning}\n</think>\n\\boxed{answer}

For categories where our solvers work (roman, unit, physics, encryption),
we generate detailed CoT. For hard categories (bit manipulation, symbol transform),
we use a generic CoT that shows the examples and concludes with the answer.
"""

import csv
import json
import re

from solvers import roman_numeral, unit_conversion, physics, caesar_cipher


BOXED_SUFFIX = '\nPlease put your final answer inside `\\boxed{}`. For example: `\\boxed{your answer}`'


def make_user_content(prompt):
    """Format the user message exactly as the metric code does."""
    return prompt + BOXED_SUFFIX


def make_assistant_content(reasoning, answer):
    """Format assistant response with thinking tags and boxed answer."""
    if reasoning:
        return f"<think>\n{reasoning}\n</think>\n\\boxed{{{answer}}}"
    else:
        # No reasoning — still use think tags (empty thinking is valid)
        return f"<think>\n{answer}\n</think>\n\\boxed{{{answer}}}"


def generate_cot_roman(prompt, answer):
    """Generate CoT for NUMBER_SYSTEM."""
    match = re.search(r'write the number (\d+)', prompt, re.IGNORECASE)
    if not match:
        return None
    n = int(match.group(1))
    roman_map = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
        (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
        (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I'),
    ]
    steps = []
    remaining = n
    for value, numeral in roman_map:
        if remaining >= value:
            count = remaining // value
            used = numeral * count
            steps.append(f"{value} × {count} = {used}")
            remaining -= value * count
            if remaining == 0:
                break
    steps_text = ", ".join(steps)
    return f"I need to convert {n} to Roman numerals.\nBreaking it down: {steps_text}\nSo {n} = {answer}"


def generate_cot_unit(prompt, answer):
    """Generate CoT for UNIT_CONVERSION."""
    examples = re.findall(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt)
    target_match = re.search(r'convert the following measurement:\s*([\d.]+)\s*m', prompt, re.IGNORECASE)
    if not examples or not target_match:
        return None

    target = target_match.group(1)
    lines = ["I need to find the conversion factor from the examples."]
    ratios = []
    for inp, out in examples:
        ratio = float(out) / float(inp)
        ratios.append(ratio)
        lines.append(f"{out} / {inp} = {ratio:.4f}")

    import statistics
    avg = statistics.mean(ratios)
    lines.append(f"Average conversion factor: {avg:.4f}")
    lines.append(f"Applying to {target}: {target} × {avg:.4f} = {answer}")
    return "\n".join(lines)


def generate_cot_physics(prompt, answer):
    """Generate CoT for PHYSICS."""
    examples = re.findall(r'For t\s*=\s*([\d.]+)s?,\s*distance\s*=\s*([\d.]+)\s*m', prompt)
    target_match = re.search(r'falling distance for t\s*=\s*([\d.]+)s?', prompt, re.IGNORECASE)
    if not examples or not target_match:
        return None

    target_t = target_match.group(1)
    lines = ["Using d = 0.5 × g × t², I solve for g from each example:"]

    import statistics
    g_values = []
    for t, d in examples:
        g = 2 * float(d) / (float(t) ** 2)
        g_values.append(g)
        lines.append(f"t={t}s, d={d}m → g = 2×{d}/{t}² = {g:.2f}")

    avg_g = statistics.mean(g_values)
    lines.append(f"Average g = {avg_g:.2f}")
    lines.append(f"For t = {target_t}s: d = 0.5 × {avg_g:.2f} × {target_t}² = {answer}")
    return "\n".join(lines)


def generate_cot_encryption(prompt, answer):
    """Generate CoT for TEXT_ENCRYPTION (monoalphabetic substitution)."""
    pairs = re.findall(r'([a-zA-Z ]+?)\s*->\s*([a-zA-Z ]+?)(?:\n)', prompt)
    target_match = re.search(r'decrypt the following text:\s*(.+?)$', prompt, re.IGNORECASE | re.MULTILINE)
    if not pairs or not target_match:
        return None

    target = target_match.group(1).strip()

    # Build the mapping
    char_map = {}
    for enc, dec in pairs:
        enc_clean = enc.strip().lower().replace(' ', '')
        dec_clean = dec.strip().lower().replace(' ', '')
        if len(enc_clean) != len(dec_clean):
            continue
        for e, d in zip(enc_clean, dec_clean):
            char_map[e] = d

    # Show a subset of discovered mappings
    sorted_map = sorted(char_map.items())
    map_str = ", ".join(f"{k}→{v}" for k, v in sorted_map[:20])

    lines = [
        "This is a monoalphabetic substitution cipher.",
        f"From the examples, I build the letter mapping: {map_str}",
        f"Applying the mapping to '{target}':",
        f"Result: {answer}",
    ]
    return "\n".join(lines)


def generate_cot_bit(prompt, answer):
    """Generate CoT for BIT_MANIPULATION — generic analysis since most rules are complex."""
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    target_match = re.search(r'determine the output for:\s*([01]{8})', prompt, re.IGNORECASE)
    if not pairs or not target_match:
        return None

    target = target_match.group(1)
    lines = ["I need to find the bit transformation rule from the examples."]
    lines.append(f"Given {len(pairs)} input→output pairs:")
    for inp, out in pairs[:4]:
        xor = int(inp, 2) ^ int(out, 2)
        lines.append(f"  {inp} → {out} (XOR: {format(xor, '08b')})")
    if len(pairs) > 4:
        lines.append(f"  ... and {len(pairs) - 4} more pairs")
    lines.append(f"After analyzing all pairs, I determine the transformation rule.")
    lines.append(f"Applying to {target}: the output is {answer}")
    return "\n".join(lines)


def generate_cot_symbol(prompt, answer):
    """Generate CoT for SYMBOL_TRANSFORM — generic analysis."""
    target_match = re.search(r'determine the result for:\s*(.+?)$', prompt, re.IGNORECASE | re.MULTILINE)
    if not target_match:
        return None

    target = target_match.group(1).strip()
    lines = [
        "I need to find the transformation rule from the examples.",
        "Analyzing the pattern across all provided examples...",
        f"Applying the discovered rule to '{target}': {answer}",
    ]
    return "\n".join(lines)


COT_GENERATORS = {
    'NUMBER_SYSTEM': generate_cot_roman,
    'UNIT_CONVERSION': generate_cot_unit,
    'PHYSICS': generate_cot_physics,
    'TEXT_ENCRYPTION': generate_cot_encryption,
    'BIT_MANIPULATION': generate_cot_bit,
    'SYMBOL_TRANSFORM': generate_cot_symbol,
}


def format_row(row):
    """Convert a training row to chat message format."""
    prompt = row['prompt']
    answer = row['answer']
    category = row['category']

    user_content = make_user_content(prompt)

    # Generate CoT reasoning
    generator = COT_GENERATORS.get(category)
    reasoning = None
    if generator:
        reasoning = generator(prompt, answer)

    assistant_content = make_assistant_content(reasoning, answer)

    return {
        'id': row['id'],
        'category': category,
        'messages': [
            {'role': 'system', 'content': ''},
            {'role': 'user', 'content': user_content},
            {'role': 'assistant', 'content': assistant_content},
        ],
    }


def main():
    # Process training split
    with open('data/train_split.csv') as f:
        train_rows = list(csv.DictReader(f))

    with open('data/val_split.csv') as f:
        val_rows = list(csv.DictReader(f))

    for split_name, rows, output_path in [
        ('train', train_rows, 'data/train_formatted.jsonl'),
        ('val', val_rows, 'data/val_formatted.jsonl'),
    ]:
        with open(output_path, 'w') as f:
            for row in rows:
                formatted = format_row(row)
                f.write(json.dumps(formatted) + '\n')

        print(f"{split_name}: {len(rows)} rows → {output_path}")

    # Show a sample
    sample = format_row(train_rows[0])
    print(f"\n--- Sample ({sample['category']}) ---")
    for msg in sample['messages']:
        print(f"[{msg['role']}]:")
        print(msg['content'][:300])
        print()


if __name__ == '__main__':
    main()
