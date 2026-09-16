"""Evaluation runner against the full supplied historical dataset
(data/tickets.csv, ~214 rows), using resolved_action as ground truth.

This is a stronger accuracy signal than the 5 sample_test_cases.json rows
alone. Per DATA_NOTES.md, this is legitimate as long as the decision
pipeline reasons from the current ticket + policy docs rather than looking
up the historical row - which is exactly what it does; resolved_action is
only used here, in the eval script, to score the output afterward.

The free Gemini tier caps requests at 20/day, so this supports --limit and
--seed to run a random subset per invocation (accumulate coverage across
multiple days), rather than requiring all 214 rows in one run.

Usage:
    python -m eval.run_full_eval --limit 20 --seed 1
    python -m eval.run_full_eval              # attempt all rows
"""

import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.decision import make_decision  # noqa: E402

TICKETS_CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "tickets.csv"


def _parse_optional_int(value: str) -> int | None:
    return int(value) if value.strip() else None


def _parse_optional_str(value: str) -> str | None:
    value = value.strip()
    return value if value and value != "unknown" else (value or None)


def load_rows() -> list[dict]:
    with TICKETS_CSV_PATH.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader if row.get("ticket_id")]


def row_to_ticket(row: dict) -> dict:
    return {
        "message": row["message"],
        "order_value_inr": float(row["order_value_inr"]) if row["order_value_inr"] else None,
        "days_since_delivery": _parse_optional_int(row["days_since_delivery"]),
        "days_since_dispatch": _parse_optional_int(row["days_since_dispatch"]),
        "product_type": _parse_optional_str(row["product_type"]),
        "opened_status": _parse_optional_str(row["opened_status"]),
        "order_status": _parse_optional_str(row["order_status"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--limit", type=int, default=None, help="Max rows to evaluate (default: all)"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for row sampling when --limit is set"
    )
    args = parser.parse_args()

    rows = load_rows()
    if args.limit is not None and args.limit < len(rows):
        random.Random(args.seed).shuffle(rows)
        rows = rows[: args.limit]

    correct = 0
    per_action = defaultdict(lambda: {"correct": 0, "total": 0})
    mismatches = []

    for i, row in enumerate(rows, start=1):
        ticket = row_to_ticket(row)
        decision, _ = make_decision(ticket)
        expected = row["resolved_action"]
        actual = decision.action.value
        is_correct = actual == expected

        correct += int(is_correct)
        per_action[expected]["total"] += 1
        per_action[expected]["correct"] += int(is_correct)

        status = "PASS" if is_correct else "FAIL"
        print(f"[{i}/{len(rows)}] [{status}] ticket_id={row['ticket_id']} "
              f"expected={expected} actual={actual}")
        if not is_correct:
            mismatches.append(
                {"ticket_id": row["ticket_id"], "message": row["message"],
                 "expected": expected, "actual": actual}
            )

    total = len(rows)
    accuracy = correct / total * 100 if total else 0.0

    print("\n--- Per-action breakdown ---")
    for action, stats in sorted(per_action.items()):
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] else 0.0
        print(f"{action:35s} {stats['correct']:3d}/{stats['total']:<3d} ({acc:5.1f}%)")

    if mismatches:
        print("\n--- Mismatches ---")
        for m in mismatches:
            print(f"  ticket_id={m['ticket_id']}: expected={m['expected']} "
                  f"actual={m['actual']} | \"{m['message']}\"")

    print("\n--- Summary ---")
    print(f"{total} tickets evaluated")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy:.1f}%")


if __name__ == "__main__":
    main()
