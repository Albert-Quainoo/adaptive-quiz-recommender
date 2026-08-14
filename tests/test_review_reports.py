"""AutomatedReviewReportStore tests: caching by content hash, atomic writes, backlog
counting -- mirrors GroundedReviewStore's own test conventions."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from authoring.grounded_review import CurationItem, GroundedReview
from authoring.review.deterministic import run_deterministic_checks
from authoring.review.models import AutomatedReviewReport
from authoring.review.reports import (
    AutomatedReviewReportStore,
    count_pending_review_items,
    review_report_path,
)
from authoring.review.risk import score_risk
from authoring.review.config import ReviewPolicyConfig
from tests.review_fixtures import APPROVED_REFERENCES, CORRECTED_CANDIDATE, INTENT, SKILL

FIXED_TIME = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def _report(content_hash: str = "hash-1") -> AutomatedReviewReport:
    checks = run_deterministic_checks(CORRECTED_CANDIDATE, SKILL, INTENT, APPROVED_REFERENCES)
    decision = score_risk(checks, [], config=ReviewPolicyConfig())
    return AutomatedReviewReport(
        review_id=str(uuid4()),
        candidate_id=CORRECTED_CANDIDATE.question_id,
        skill_id="AI-SRC-08",
        intent_id=INTENT.intent_id,
        review_policy_version="review-policy-v1",
        reviewer_model_id="n/a",
        reviewer_model_revision="n/a",
        reviewer_prompt_version="n/a",
        reviewer_prompt_template_hash="n/a",
        rendered_review_request_hash="n/a",
        reviewed_content_hash=content_hash,
        created_at=FIXED_TIME,
        deterministic_checks=checks,
        risk_score=decision.risk_score,
        risk_level=decision.risk_level,
        recommendation=decision.recommendation,
        blocking_reasons=decision.blocking_reasons,
        warnings=decision.warnings,
    )


def test_append_and_load_all_round_trips(tmp_path: Path):
    store = AutomatedReviewReportStore(tmp_path / "reports.json")
    report = _report()
    store.append(report)
    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].reviewed_content_hash == "hash-1"


def test_latest_for_hash_returns_none_when_absent(tmp_path: Path):
    store = AutomatedReviewReportStore(tmp_path / "reports.json")
    assert store.latest_for_hash("missing") is None


def test_latest_for_hash_returns_most_recent(tmp_path: Path):
    store = AutomatedReviewReportStore(tmp_path / "reports.json")
    older = _report("hash-1").model_copy(update={"created_at": FIXED_TIME})
    newer = _report("hash-1").model_copy(
        update={"created_at": FIXED_TIME.replace(hour=13), "review_id": str(uuid4())}
    )
    store.append(older)
    store.append(newer)
    latest = store.latest_for_hash("hash-1")
    assert latest.review_id == newer.review_id


def test_append_is_atomic_write_no_partial_file(tmp_path: Path):
    path = tmp_path / "reports.json"
    store = AutomatedReviewReportStore(path)
    store.append(_report())
    assert not (tmp_path / "reports.json.tmp").exists()
    json.loads(path.read_text(encoding="utf-8"))


def test_review_report_path_is_nested_under_automated_review_reports_subdirectory():
    path = review_report_path(Path("outputs/replenishment/ai/reviews"), "batch-1", "AI-SRC-08")
    assert path == Path(
        "outputs/replenishment/ai/reviews/automated_review_reports/batch-1__AI-SRC-08.json"
    )


def test_count_pending_review_items_counts_only_pending(tmp_path: Path):
    review_store = tmp_path / "reviews"
    review_store.mkdir()
    review = GroundedReview(
        batch_id="batch-1",
        source_hashes={},
        items=[
            CurationItem(
                original_question_id="q-1",
                skill_id="AI-SRC-08",
                intent_id="AI-SRC-08-INT-02",
                recommendation="propose_revision",
                recommendation_reason="pending",
            ),
            CurationItem(
                original_question_id="q-2",
                skill_id="AI-SRC-08",
                intent_id="AI-SRC-08-INT-01",
                recommendation="approve_as_written",
                recommendation_reason="pending",
            ),
        ],
    )
    (review_store / "batch-1__AI-SRC-08.json").write_text(
        review.model_dump_json(), encoding="utf-8"
    )
    assert count_pending_review_items(review_store) == 2


def test_count_pending_review_items_ignores_the_reports_subdirectory(tmp_path: Path):
    review_store = tmp_path / "reviews"
    review_store.mkdir()
    reports_path = review_report_path(review_store, "batch-1", "AI-SRC-08")
    AutomatedReviewReportStore(reports_path).append(_report())
    # No review.json exists at all -- only the nested reports file -- so the
    # non-recursive glob("*.json") at review_store must see nothing to parse.
    assert count_pending_review_items(review_store) == 0
