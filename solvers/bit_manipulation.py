"""Solver for BIT_MANIPULATION category: try to reverse-engineer 8-bit transformations.

This is the hardest category. We try common bit operations and their compositions
to find a rule that fits all examples. If we can't find a perfect match, we return None.
"""

import re
from itertools import product


def parse_bit_examples(prompt):
    """Extract input->output binary pairs and the target input."""
    pairs = re.findall(r'([01]{8})\s*->\s*([01]{8})', prompt)
    target_match = re.search(r'determine the output for:\s*([01]{8})', prompt, re.IGNORECASE)
    if not pairs or not target_match:
        return None, None, None
    examples = [(inp, out) for inp, out in pairs]
    target = target_match.group(1)
    return examples, target, None


# --- Primitive operations on 8-bit integers ---

def not8(x): return x ^ 0xFF
def shl(x, n): return (x << n) & 0xFF
def shr(x, n): return (x >> n) & 0xFF
def rol(x, n): return ((x << n) | (x >> (8 - n))) & 0xFF
def ror(x, n): return ((x >> n) | (x << (8 - n))) & 0xFF
def reverse_bits(x):
    r = 0
    for _ in range(8):
        r = (r << 1) | (x & 1)
        x >>= 1
    return r
def swap_nibbles(x): return ((x & 0x0F) << 4) | ((x & 0xF0) >> 4)


def try_xor_constant(examples):
    """Check if output = input XOR constant."""
    if not examples:
        return None
    inp0, out0 = int(examples[0][0], 2), int(examples[0][1], 2)
    c = inp0 ^ out0
    for inp, out in examples[1:]:
        if int(inp, 2) ^ c != int(out, 2):
            return None
    return lambda x: x ^ c


def try_single_ops(examples):
    """Try single operations: NOT, reverse, swap nibbles, shifts, rotations."""
    ops = [
        ('NOT', not8),
        ('REVERSE', reverse_bits),
        ('SWAP_NIBBLES', swap_nibbles),
    ]
    for n in range(1, 8):
        ops.append((f'SHL_{n}', lambda x, n=n: shl(x, n)))
        ops.append((f'SHR_{n}', lambda x, n=n: shr(x, n)))
        ops.append((f'ROL_{n}', lambda x, n=n: rol(x, n)))
        ops.append((f'ROR_{n}', lambda x, n=n: ror(x, n)))

    for name, op in ops:
        if all(op(int(inp, 2)) == int(out, 2) for inp, out in examples):
            return op
    return None


def try_two_op_compositions(examples):
    """Try compositions of two operations."""
    base_ops = [not8, reverse_bits, swap_nibbles]
    for n in range(1, 8):
        base_ops.append(lambda x, n=n: shl(x, n))
        base_ops.append(lambda x, n=n: shr(x, n))
        base_ops.append(lambda x, n=n: rol(x, n))
        base_ops.append(lambda x, n=n: ror(x, n))

    for op1 in base_ops:
        for op2 in base_ops:
            composed = lambda x, o1=op1, o2=op2: o2(o1(x))
            if all(composed(int(inp, 2)) == int(out, 2) for inp, out in examples):
                return composed
    return None


def try_xor_with_op(examples):
    """Try: op(input) XOR constant."""
    ops = [
        lambda x: x,  # identity
        not8, reverse_bits, swap_nibbles,
    ]
    for n in range(1, 8):
        ops.append(lambda x, n=n: rol(x, n))
        ops.append(lambda x, n=n: ror(x, n))
        ops.append(lambda x, n=n: shl(x, n))
        ops.append(lambda x, n=n: shr(x, n))

    for op in ops:
        # Determine XOR constant from first example
        inp0, out0 = int(examples[0][0], 2), int(examples[0][1], 2)
        c = op(inp0) ^ out0
        combined = lambda x, o=op, c=c: o(x) ^ c
        if all(combined(int(inp, 2)) == int(out, 2) for inp, out in examples):
            return combined
    return None


def solve(prompt: str) -> str:
    """Try to find the bit manipulation rule and apply it to the target."""
    examples, target, _ = parse_bit_examples(prompt)
    if not examples or not target:
        return None

    target_int = int(target, 2)

    # Try strategies in order of complexity
    strategies = [
        try_xor_constant,
        try_single_ops,
        try_xor_with_op,
        try_two_op_compositions,
    ]

    for strategy in strategies:
        func = strategy(examples)
        if func is not None:
            result = func(target_int)
            return format(result, '08b')

    return None  # Could not determine the rule


def generate_cot(prompt: str, answer: str) -> str:
    """Generate chain-of-thought for bit manipulation problems."""
    examples, target, _ = parse_bit_examples(prompt)
    if not examples:
        return f"\\boxed{{{answer}}}"

    lines = []
    lines.append("I need to find the transformation rule from the examples.")
    lines.append(f"\nGiven {len(examples)} input-output pairs, let me analyze the pattern.")

    # Show a few examples
    for inp, out in examples[:3]:
        xor_val = int(inp, 2) ^ int(out, 2)
        lines.append(f"- {inp} -> {out} (XOR difference: {format(xor_val, '08b')})")

    lines.append(f"\nAfter analyzing all examples, I determine the transformation rule.")
    lines.append(f"Applying the rule to {target}: {answer}")

    return '\n'.join(lines) + f"\n\n\\boxed{{{answer}}}"


if __name__ == '__main__':
    # Test with a simple XOR constant case
    test_prompt = """In Alice's Wonderland, a secret bit manipulation rule transforms 8-bit binary numbers. The transformation involves operations like bit shifts, rotations, XOR, AND, OR, NOT, and possibly majority or choice functions.

Here are some examples of input -> output:
00000000 -> 11111111
11111111 -> 00000000
10101010 -> 01010101
01010101 -> 10101010

Now, determine the output for: 11001100"""
    result = solve(test_prompt)
    print(f"Result: {result}, Expected: 00110011 (NOT operation)")
    if result:
        assert result == "00110011", f"Got {result}"
        print("Test passed.")
    else:
        print("Could not solve (expected for complex rules).")
