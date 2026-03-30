"""Solver for PHYSICS category: solve for altered gravitational constant, compute distance."""

import re
import statistics


def solve(prompt: str) -> str:
    """Extract time/distance examples, solve for g, compute target distance."""
    examples = re.findall(r'For t\s*=\s*([\d.]+)s?,\s*distance\s*=\s*([\d.]+)\s*m', prompt)
    target_match = re.search(r'falling distance for t\s*=\s*([\d.]+)s?', prompt, re.IGNORECASE)

    if not examples or not target_match:
        return None

    # d = 0.5 * g * t^2  =>  g = 2d / t^2
    g_values = [2 * float(d) / (float(t) ** 2) for t, d in examples]
    g = statistics.mean(g_values)
    target_t = float(target_match.group(1))
    result = 0.5 * g * target_t ** 2

    return f"{result:.2f}"


def generate_cot(prompt: str, answer: str) -> str:
    """Generate chain-of-thought reasoning."""
    examples = re.findall(r'For t\s*=\s*([\d.]+)s?,\s*distance\s*=\s*([\d.]+)\s*m', prompt)
    target_match = re.search(r'falling distance for t\s*=\s*([\d.]+)s?', prompt, re.IGNORECASE)

    if not examples or not target_match:
        return f"\\boxed{{{answer}}}"

    target_t = target_match.group(1)
    lines = []
    lines.append("Using d = 0.5 × g × t², I can solve for g from each example:")
    g_values = []
    for t, d in examples:
        g = 2 * float(d) / (float(t) ** 2)
        g_values.append(g)
        lines.append(f"- t={t}s, d={d}m: g = 2×{d}/{t}² = {g:.2f}")

    avg_g = statistics.mean(g_values)
    lines.append(f"\nAverage g = {avg_g:.2f}")
    result = 0.5 * avg_g * float(target_t) ** 2
    lines.append(f"\nFor t = {target_t}s:")
    lines.append(f"d = 0.5 × {avg_g:.2f} × {target_t}² = {result:.2f}")

    return '\n'.join(lines) + f"\n\n\\boxed{{{answer}}}"


if __name__ == '__main__':
    test_prompt = """In Alice's Wonderland, the gravitational constant has been secretly changed. Here are some example observations:
For t = 1.86s, distance = 14.58 m
For t = 2.33s, distance = 22.87 m
For t = 2.95s, distance = 36.67 m
Now, determine the falling distance for t = 4.67s given d = 0.5*g*t^2."""
    result = solve(test_prompt)
    print(f"Result: {result}, Expected: 91.89")
    assert abs(float(result) - 91.89) < 0.5, f"Got {result}"
    print("Test passed.")
