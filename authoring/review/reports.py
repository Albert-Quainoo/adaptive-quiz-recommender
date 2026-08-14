"""On-disk store for AutomatedReviewReport records, parallel to grounded_review.py's
GroundedReviewStore.

Reports are immutable and append-only, keyed by reviewed_content_hash so an unchanged
candidate is never re-reviewed: calling latest_for_hash() before invoking the reviewer is
what implements the "do not create another reviewer call for an unchanged content hash
with an existing completed report" cost control.

Reports live in a fixed "automated_review_reports" subdirectory under a course's
review_store_path, so a non-recursive glob("*.json") over review_store_path itself (as
authoring/replenishment/inventory.py's pending-question scan already does) never
encounters a reports file.
"""

import json
from pathlib import Path

from authoring.grounded_review import GroundedReview
from authoring.review.models import AutomatedReviewReport

REPORTS_SUBDIRECTORY = "automated_review_reports"


def review_report_path(review_store_path: Path, batch_id: str, skill_id: str) -> Path:
    return review_store_path / REPORTS_SUBDIRECTORY / f"{batch_id}__{skill_id}.json"


class AutomatedReviewReportStore:
    def __init__(self, path: Path) -> None:
        self.path = path

    def load_all(self) -> list[AutomatedReviewReport]:
        if not self.path.is_file():
            return []
        return [
            AutomatedReviewReport.model_validate(record)
            for record in json.loads(self.path.read_text(encoding="utf-8"))
        ]

    def append(self, report: AutomatedReviewReport) -> None:
        reports = self.load_all()
        reports.append(report)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps([r.model_dump(mode="json") for r in reports], indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def latest_for_hash(self, content_hash: str) -> AutomatedReviewReport | None:
        matching = [
            report for report in self.load_all() if report.reviewed_content_hash == content_hash
        ]
        if not matching:
            return None
        return max(matching, key=lambda report: report.created_at)

    def for_hashes(self, content_hashes: set[str]) -> dict[str, AutomatedReviewReport]:
        """The latest report per hash, for the hashes actually asked for."""
        latest: dict[str, AutomatedReviewReport] = {}
        for report in self.load_all():
            if report.reviewed_content_hash not in content_hashes:
                continue
            existing = latest.get(report.reviewed_content_hash)
            if existing is None or report.created_at > existing.created_at:
                latest[report.reviewed_content_hash] = report
        return latest


def count_pending_review_items(review_store_path: Path) -> int:
    """Total items still awaiting a human decision across every review file in a
    course's review store -- the global backlog figure for backpressure."""
    if not review_store_path.is_dir():
        return 0
    total = 0
    for path in sorted(review_store_path.glob("*.json")):
        try:
            review = GroundedReview.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        total += sum(1 for item in review.items if item.final_review_status == "pending")
    return total
