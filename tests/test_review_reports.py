"""AutomatedReviewReportStore tests: caching by content hash, atomic writes, backlog
counting -- mirrors GroundedReviewStore's own test conventions."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from authoring.grounded_review import CurationItem, GroundedReview
from authoring.review.deterministic import run_deterministic_checks
from authoring.review.models import AnswerAssessment, AutomatedReviewReport, EquivalenceAssessment, OptionPairEvidence
from authoring.review.reports import (
    AutomatedReviewReportStore,
    count_pending_review_items,
    review_report_path,
)
from authoring.review.risk import score_risk
from authoring.review.config import ReviewPolicyConfig
from tests.review_fixtures import (
    APPROVED_REFERENCES,
    CORRECTED_CANDIDATE,
    CORRECTED_REVIEW_RESULT,
    INTENT,
    SKILL,
)

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


def test_historical_report_without_equivalence_assessment_loads_with_none(tmp_path: Path):
    """Every real report written before the hybrid option-equivalence gate existed has
    no equivalence_assessment field at all -- reading it back must default to None, not
    error, exactly like option_assessments' own precedent below."""
    store = AutomatedReviewReportStore(tmp_path / "reports.json")
    report = _report()
    assert report.equivalence_assessment is None
    store.append(report)
    reloaded = store.load_all()
    assert reloaded[0].equivalence_assessment is None


def test_report_with_equivalence_assessment_round_trips(tmp_path: Path):
    report = _report().model_copy(
        update={
            "equivalence_assessment": EquivalenceAssessment(
                gate_version="equivalence-gate-v1",
                nli_model_repository="cross-encoder/nli-deberta-v3-xsmall",
                nli_model_revision="a150876415327c80daeff35ca6f68f5ed8cf5c24",
                nli_threshold=0.25,
                threshold_version="equivalence-threshold-v1-2026-08-17",
                evidence=[
                    OptionPairEvidence(
                        option_index_a=0, option_index_b=1, detector="unit_conversion",
                        verdict="equivalent", score_or_normalized_form="0.75 cup vs 0.75 cup",
                        reason="both options convert to the same canonical quantity",
                    )
                ],
                escalated=True,
            )
        }
    )
    store = AutomatedReviewReportStore(tmp_path / "reports.json")
    store.append(report)
    reloaded = store.load_all()[0]
    assert reloaded.equivalence_assessment.escalated is True
    assert reloaded.equivalence_assessment.nli_threshold == 0.25
    assert len(reloaded.equivalence_assessment.evidence) == 1
    assert reloaded.equivalence_assessment.evidence[0].detector == "unit_conversion"


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


def test_historical_report_with_empty_option_assessments_still_round_trips(tmp_path: Path):
    """A stored report produced before CompactReviewResult.option_assessments existed
    (every real report written before this milestone) carries
    AnswerAssessment.option_assessments={} -- the schema must keep accepting that on
    read; only the live parsing path (authoring/review/response_parser.py's
    validate_compact_reviewer_output, exercised only for a freshly-parsed reviewer
    response) requires a complete assessment. This never rewrites historical data --
    it round-trips a report shaped exactly like historical ones through the same
    store every report (old or new) goes through."""
    checks = run_deterministic_checks(CORRECTED_CANDIDATE, SKILL, INTENT, APPROVED_REFERENCES)
    assert CORRECTED_REVIEW_RESULT.answer_assessment.option_assessments == {}
    report = AutomatedReviewReport(
        review_id=str(uuid4()),
        candidate_id=CORRECTED_CANDIDATE.question_id,
        skill_id="AI-SRC-08",
        intent_id=INTENT.intent_id,
        review_policy_version="review-policy-v1",
        reviewer_model_id=CORRECTED_REVIEW_RESULT.reviewer_model_id,
        reviewer_model_revision=CORRECTED_REVIEW_RESULT.reviewer_model_revision,
        reviewer_prompt_version=CORRECTED_REVIEW_RESULT.reviewer_prompt_version,
        reviewer_prompt_template_hash=CORRECTED_REVIEW_RESULT.reviewer_prompt_template_hash,
        rendered_review_request_hash=CORRECTED_REVIEW_RESULT.rendered_review_request_hash,
        reviewed_content_hash="historical-hash-1",
        created_at=FIXED_TIME,
        deterministic_checks=checks,
        grounding_assessment=CORRECTED_REVIEW_RESULT.grounding_assessment,
        answer_assessment=CORRECTED_REVIEW_RESULT.answer_assessment,
        objective_assessment=CORRECTED_REVIEW_RESULT.objective_assessment,
        difficulty_assessment=CORRECTED_REVIEW_RESULT.difficulty_assessment,
        duplicate_assessment=CORRECTED_REVIEW_RESULT.duplicate_assessment,
        risk_score=0.1,
        risk_level="low",
        recommendation="recommend_human_approval",
    )

    store = AutomatedReviewReportStore(tmp_path / "reports.json")
    store.append(report)

    reloaded = AutomatedReviewReportStore(tmp_path / "reports.json").load_all()
    assert len(reloaded) == 1
    assert reloaded[0].answer_assessment.option_assessments == {}
    assert reloaded[0].reviewed_content_hash == "historical-hash-1"


def test_real_historical_ai_fnd_04_report_with_empty_option_assessments_loads(tmp_path: Path):
    """Same compatibility guarantee, proven against the actual committed production
    artifact this whole regression is about -- not a synthetic stand-in."""
    real_report_path = (
        Path(__file__).resolve().parent.parent
        / "outputs/replenishment/ai/reviews/automated_review_reports"
        / "grounded-ai-fnd-release-v1__AI-FND-04.json"
    )
    store = AutomatedReviewReportStore(real_report_path)
    reports = store.load_all()
    assert reports, "expected the real historical AI-FND-04 report file to be present"
    for report in reports:
        assert report.answer_assessment is not None
        assert report.answer_assessment.option_assessments == {}


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
