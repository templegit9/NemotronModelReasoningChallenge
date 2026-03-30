"""Solver for UNIT_CONVERSION category: infer linear conversion factor and apply."""

import re
import statistics


def solve(prompt: str) -> str:
    """Extract examples, compute conversion factor, apply to target."""
    examples = re.findall(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt)
    target_match = re.search(r'convert the following measurement:\s*([\d.]+)\s*m', prompt, re.IGNORECASE)

    if not examples or not target_match:
        return None

    ratios = [float(out) / float(inp) for inp, out in examples]
    factor = statistics.mean(ratios)
    target = float(target_match.group(1))
    result = target * factor

    return f"{result:.2f}"


def generate_cot(prompt: str, answer: str) -> str:
    """Generate chain-of-thought reasoning."""
    examples = re.findall(r'([\d.]+)\s*m\s+becomes\s+([\d.]+)', prompt)
    target_match = re.search(r'convert the following measurement:\s*([\d.]+)\s*m', prompt, re.IGNORECASE)

    if not examples or not target_match:
        return f"\\boxed{{{answer}}}"

    target = target_match.group(1)
    lines = []
    lines.append("I need to find the conversion factor from the examples:")
    ratios = []
    for inp, out in examples:
        ratio = float(out) / float(inp)
        ratios.append(ratio)
        lines.append(f"- {out} / {inp} = {ratio:.4f}")

    avg = statistics.mean(ratios)
    lines.append(f"\nThe average conversion factor is {avg:.4f}")
    lines.append(f"\nApplying to {target}: {target} × {avg:.4f} = {float(target) * avg:.2f}")

    return '\n'.join(lines) + f"\n\n\\boxed{{{answer}}}"


if __name__ == '__main__':
    test_prompt = """In Alice's Wonderland, a secret unit conversion is applied to measurements. For example:
18.75 m becomes 19.91
46.61 m becomes 49.48
29.36 m becomes 31.17
19.62 m becomes 20.83
Now, convert the following measurement: 15.19 m"""
    result = solve(test_prompt)
    print(f"Result: {result}, Expected: 16.13")
    # Check within tolerance
    assert abs(float(result) - 16.13) < 0.2, f"Got {result}"
    print("Test passed.")
