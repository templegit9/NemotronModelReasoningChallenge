"""Solver for NUMBER_SYSTEM category: decimal to Roman numeral conversion."""

import re

ROMAN_MAP = [
    (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'),
    (100, 'C'), (90, 'XC'), (50, 'L'), (40, 'XL'),
    (10, 'X'), (9, 'IX'), (5, 'V'), (4, 'IV'), (1, 'I'),
]


def decimal_to_roman(n: int) -> str:
    result = []
    for value, numeral in ROMAN_MAP:
        while n >= value:
            result.append(numeral)
            n -= value
    return ''.join(result)


def solve(prompt: str) -> str:
    """Extract the target number from prompt and convert to Roman numeral."""
    match = re.search(r'write the number (\d+)', prompt, re.IGNORECASE)
    if not match:
        return None
    return decimal_to_roman(int(match.group(1)))


def generate_cot(prompt: str, answer: str) -> str:
    """Generate chain-of-thought reasoning for this problem."""
    match = re.search(r'write the number (\d+)', prompt, re.IGNORECASE)
    if not match:
        return f"The answer is \\boxed{{{answer}}}"

    n = int(match.group(1))
    steps = []
    remaining = n
    parts = []
    for value, numeral in ROMAN_MAP:
        if remaining >= value:
            count = remaining // value
            parts.append(f"{remaining} >= {value}, so I write '{numeral}' {'× ' + str(count) if count > 1 else ''}")
            remaining %= value
            if remaining == 0:
                break

    steps_text = '\n'.join(f"- {s}" for s in parts)
    return (
        f"I need to convert {n} to Roman numerals.\n"
        f"{steps_text}\n"
        f"So {n} = {answer}\n\n"
        f"\\boxed{{{answer}}}"
    )


if __name__ == '__main__':
    # Test
    assert decimal_to_roman(38) == 'XXXVIII'
    assert decimal_to_roman(94) == 'XCIV'
    assert decimal_to_roman(9) == 'IX'
    assert decimal_to_roman(18) == 'XVIII'
    print("All tests passed.")
