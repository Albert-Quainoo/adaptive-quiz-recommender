"""Run the automated-review calibration harness and print/report its metrics.

Fakes only, by default: the reference reviewer encodes the judgment an accurate
reviewer should reach for each labeled case (see
evaluation/review_calibration.py's module docstring), so this never calls Modal or
any real model unless --live is passed with an authorized reviewer provider wired
in separately. This is an evaluation and threshold-calibration run, never training.

    python -m scripts.run_review_calibration
    python -m scripts.run_review_calibration --output outputs/review_calibration.json
"""

import argparse
import json
from pathlib import Path

from evaluation.review_calibration import (
    build_negative_cases,
    build_positive_cases,
    build_reference_reviewer,
    run_calibration,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--report-store",
        type=Path,
        default=Path("outputs/review_calibration_reports.json"),
        help="Scratch AutomatedReviewReport store for this run (overwritten each run).",
    )
    arguments = parser.parse_args(argv)

    positive_cases = build_positive_cases()
    negative_cases = build_negative_cases()
    cases = positive_cases + negative_cases
    reviewer_factory = lambda: build_reference_reviewer(cases)  # noqa: E731

    arguments.report_store.unlink(missing_ok=True)
    report = run_calibration(
        cases, reviewer_factory=reviewer_factory, report_store_path=arguments.report_store
    )

    print(f"Cases: {report.total_cases} ({report.positive_cases} positive, {report.negative_cases} negative)")
    print(f"Approval precision: {report.approval_precision:.3f}")
    print(f"Approval recall:    {report.approval_recall:.3f}")
    print(f"Critical-error detection rate: {report.critical_error_detection_rate:.3f}")
    print(f"False-low-risk rate:           {report.false_low_risk_rate:.3f}  (optimize this toward 0)")
    print(f"Disagreement rate:  {report.disagreement_rate:.3f}")
    print(f"Parser-failure rate: {report.parser_failure_rate:.3f}")
    print(f"Reviewer calls: {report.reviewer_calls}  (estimated cost: ${report.estimated_cost_usd:.2f})")

    misclassified = [
        result
        for result in report.results
        if (result.label == "positive") != (result.recommendation == "recommend_human_approval")
    ]
    if misclassified:
        print("\nMisclassified cases:")
        for result in misclassified:
            print(f"  {result.case_id} ({result.label}): {result.recommendation}/{result.risk_level}")

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"\nFull report written to {arguments.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
