"""Run all solvers against the training data and report accuracy."""

import csv
import math
import re
from collections import defaultdict

from solvers import roman_numeral, unit_conversion, physics, caesar_cipher, bit_manipulation, symbol_transform

SOLVER_MAP = {
    'NUMBER_SYSTEM': roman_numeral,
    'UNIT_CONVERSION': unit_conversion,
    'PHYSICS': physics,
    'TEXT_ENCRYPTION': caesar_cipher,
    'BIT_MANIPULATION': bit_manipulation,
    'SYMBOL_TRANSFORM': symbol_transform,
}


def verify(stored_answer, predicted):
    """Same verification logic as the competition metric."""
    stored_answer = stored_answer.strip()
    predicted = predicted.strip()

    # Binary strings: exact match
    if re.fullmatch(r'[01]+', stored_answer):
        return predicted.lower() == stored_answer.lower()

    try:
        stored_num = float(stored_answer)
        predicted_num = float(predicted)
        return math.isclose(stored_num, predicted_num, rel_tol=1e-2, abs_tol=1e-5)
    except (ValueError, TypeError):
        pass

    return predicted.lower() == stored_answer.lower()


def main():
    with open('data/train_split.csv') as f:
        rows = list(csv.DictReader(f))

    results = defaultdict(lambda: {'total': 0, 'solved': 0, 'correct': 0, 'failed': 0})

    for row in rows:
        cat = row['category']
        answer = row['answer']
        solver = SOLVER_MAP.get(cat)
        r = results[cat]
        r['total'] += 1

        if solver is None:
            continue

        predicted = solver.solve(row['prompt'])
        if predicted is None:
            r['failed'] += 1
            continue

        r['solved'] += 1
        # For text encryption, ignore '?' characters in comparison
        if cat == 'TEXT_ENCRYPTION' and '?' in predicted:
            # Check known characters match
            match = all(
                p == a for p, a in zip(predicted, answer.lower()) if p != '?'
            ) and len(predicted) == len(answer)
            if match:
                r['correct'] += 1
        elif verify(answer, predicted):
            r['correct'] += 1

    print(f"{'Category':<20} {'Total':>6} {'Solved':>7} {'Correct':>8} {'Failed':>7} {'Solve%':>7} {'Accuracy':>9}")
    print("-" * 75)
    total_all = 0
    correct_all = 0
    for cat in sorted(results):
        r = results[cat]
        solve_pct = f"{r['solved']/r['total']*100:.1f}%" if r['total'] else "N/A"
        acc = f"{r['correct']/r['total']*100:.1f}%" if r['total'] else "N/A"
        print(f"{cat:<20} {r['total']:>6} {r['solved']:>7} {r['correct']:>8} {r['failed']:>7} {solve_pct:>7} {acc:>9}")
        total_all += r['total']
        correct_all += r['correct']

    print("-" * 75)
    print(f"{'OVERALL':<20} {total_all:>6} {'':>7} {correct_all:>8} {'':>7} {'':>7} {correct_all/total_all*100:.1f}%")


if __name__ == '__main__':
    main()
