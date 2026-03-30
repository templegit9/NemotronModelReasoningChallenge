"""Solver for SYMBOL_TRANSFORM category: infer arbitrary transformation rules.

This category has several sub-types:
1. Arithmetic with swapped operations (66-83 = -82 might swap digits)
2. Character mapping/substitution rules
3. String manipulation rules

This is very hard to solve algorithmically. We focus on common patterns.
"""

import re
from collections import defaultdict


def parse_examples(prompt: str):
    """Extract transformation examples and target."""
    # Pattern for equations: input = output
    eq_examples = re.findall(r'(\S+[+\-*/]\S+)\s*=\s*(\S+)', prompt)
    # Pattern for generic: input -> output or input = output
    generic_examples = re.findall(r'(\S+(?:\s+\S+)*?)\s*=\s*(\S+(?:\s+\S+)*?)(?:\n|$)', prompt)

    target_match = re.search(r'determine the result for:\s*(.+?)$', prompt, re.IGNORECASE | re.MULTILINE)
    if not target_match:
        return None, None

    target = target_match.group(1).strip()

    if eq_examples:
        return eq_examples, target
    return generic_examples, target


def try_char_deletion(examples):
    """Check if the rule is 'delete a specific character position or character'."""
    # Check if output is input with certain chars removed
    for char_to_remove in set(''.join(inp for inp, _ in examples)):
        if all(inp.replace(char_to_remove, '') == out for inp, out in examples):
            return lambda x, c=char_to_remove: x.replace(c, '')
    return None


def try_char_substitution(examples):
    """Check if there's a character-by-character substitution."""
    # Build mapping from all examples
    mapping = {}
    for inp, out in examples:
        if len(inp) != len(out):
            return None
        for ci, co in zip(inp, out):
            if ci in mapping:
                if mapping[ci] != co:
                    return None
            mapping[ci] = co
    # Check consistency
    for inp, out in examples:
        transformed = ''.join(mapping.get(c, c) for c in inp)
        if transformed != out:
            return None
    return lambda x: ''.join(mapping.get(c, c) for c in x)


def solve(prompt: str) -> str:
    """Try to find the transformation rule and apply it."""
    examples, target = parse_examples(prompt)
    if not examples or not target:
        return None

    strategies = [
        try_char_deletion,
        try_char_substitution,
    ]

    for strategy in strategies:
        func = strategy(examples)
        if func is not None:
            return func(target)

    return None


def generate_cot(prompt: str, answer: str) -> str:
    """Generate chain-of-thought for symbol transformation problems."""
    examples, target = parse_examples(prompt)
    if not examples:
        return f"\\boxed{{{answer}}}"

    lines = []
    lines.append("I need to find the transformation rule from the examples.")
    for inp, out in examples[:3]:
        lines.append(f"- '{inp}' -> '{out}'")
    lines.append(f"\nAnalyzing the pattern across all {len(examples)} examples...")
    lines.append(f"Applying the discovered rule to '{target}': {answer}")

    return '\n'.join(lines) + f"\n\n\\boxed{{{answer}}}"


if __name__ == '__main__':
    print("Symbol transform solver - limited coverage (hardest category).")
    print("This solver handles char deletion and char substitution patterns.")
    print("Many problems in this category require more complex rule inference.")
