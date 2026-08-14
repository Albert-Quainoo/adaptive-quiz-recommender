"""Review, then approve, the reviewed v3 pilot and export its versioned approved bank."""

import argparse
from datetime import datetime, timezone
from pathlib import Path

from authoring.grounded_review import GroundedReviewStore, assert_immutable_source
from authoring.pilot_curation_v3 import (
    approve_all,
    approved_bank_items,
    load_sources,
    run_automated_review,
    write_bank,
)
from authoring.replenishment.cli import _reviewer_factory
from authoring.review.config import load_review_policy_config
from authoring.review.reports import review_report_path


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--batch", type=Path, required=True)
    root.add_argument("--store", type=Path, required=True)
    root.add_argument("--output", type=Path, required=True)
    root.add_argument("--reviewer", required=True)
    return root


def main(argv=None) -> int:
    arguments = parser().parse_args(argv)
    store = GroundedReviewStore(arguments.store)
    review = store.load()
    assert_immutable_source(arguments.batch, review)

    config = load_review_policy_config()
    reports = run_automated_review(
        arguments.batch,
        review,
        reviewer=_reviewer_factory()(),
        reports_path=review_report_path(arguments.store.parent, review.batch_id, "pilot-v3"),
        config=config,
    )
    blocking = {
        intent_id: report
        for intent_id, report in reports.items()
        if report.recommendation == "reject" or report.risk_level == "critical"
    }
    if blocking:
        for intent_id, report in sorted(blocking.items()):
            reasons = "; ".join(report.blocking_reasons) or "no reasons recorded"
            print(f"{intent_id}\t{report.recommendation}/{report.risk_level}\t{reasons}")
        raise SystemExit(
            f"automated review flagged {len(blocking)} item(s) as reject/critical; "
            "resolve before approving"
        )

    approved = approve_all(
        review,
        reviewer=arguments.reviewer,
        reviewed_at=datetime.now(timezone.utc),
    )
    items = approved_bank_items(approved, load_sources(arguments.batch))
    write_bank(arguments.output, items)
    store.save(approved)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
