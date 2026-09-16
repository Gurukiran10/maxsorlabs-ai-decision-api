"""Evaluation runner: runs the decision pipeline against the supplied sample
test cases and reports accuracy against the expected_action field.

Usage:
    python -m eval.run_eval
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.decision import make_decision  # noqa: E402

TEST_CASES_PATH = Path(__file__).resolve().parent / "sample_test_cases.json"


def main() -> None:
    cases = json.loads(TEST_CASES_PATH.read_text(encoding="utf-8"))

    correct = 0
    results = []

    for case in cases:
        ticket = {
            "message": case["message"],
            "order_value_inr": case.get("order_value_inr"),
            "days_since_delivery": case.get("days_since_delivery"),
            "days_since_dispatch": case.get("days_since_dispatch"),
            "product_type": case.get("product_type"),
            "opened_status": case.get("opened_status"),
            "order_status": case.get("order_status"),
        }
        decision, _ = make_decision(ticket)
        expected = case["expected_action"]
        actual = decision.action.value
        is_correct = actual == expected
        correct += int(is_correct)
        results.append(
            {
                "case_id": case["case_id"],
                "expected": expected,
                "actual": actual,
                "correct": is_correct,
                "confidence": decision.confidence,
            }
        )
        status = "PASS" if is_correct else "FAIL"
        print(f"[{status}] {case['case_id']}: expected={expected} actual={actual}")

    total = len(cases)
    accuracy = correct / total * 100 if total else 0.0
    print("\n--- Summary ---")
    print(f"{total} test cases")
    print(f"Correct: {correct}")
    print(f"Incorrect: {total - correct}")
    print(f"Accuracy: {accuracy:.0f}%")


if __name__ == "__main__":
    main()
