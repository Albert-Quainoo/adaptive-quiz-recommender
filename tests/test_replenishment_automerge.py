"""Unit tests for the risk-tiered auto-merge eligibility check
(authoring/replenishment/automerge.py) -- pure logic over a GroundedReview
and its matching AutomatedReviewReportStore, no live pipeline required.
"""

import json
from datetime import datetime, timezone

import pytest

from api.schemas import QuizQuestion
from authoring.grounded_batch import IntentQuestion, PendingQuestion
from authoring.grounded_review import (
    CurationItem,
    GroundedReview,
    GroundedReviewStore,
    RevisionProvenance,
    approve_as_written,
    approve_revision,
    propose_revision,
    question_content_hash,
)
from authoring.replenishment.automerge import (
    AutoMergeEvaluation,
    combine_evaluations,
    evaluate_promotion_job,
)
from authoring.replenishment.jobs import ReplenishmentJob
from authoring.replenishment.manifest import CourseManifest
from authoring.review.models import AutomatedReviewReport, DeterministicChecks
from authoring.review.reports import AutomatedReviewReportStore, review_report_path

FIXED_TIME = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
SKILL_ID = "AI-SRC-08"
BATCH_ID = "auto-merge-batch-01"


@pytest.fixture
def manifest(tmp_path):
    return CourseManifest(
        course_id="ai",
        title="test",
        version="1",
        taxonomy_path=tmp_path / "taxonomy",
        approved_bank_path=tmp_path / "bank" / "ai-bank-v0.jsonl",
        bkt_model_path=tmp_path / "model.pkl",
        candidate_store_path=tmp_path / "reference_candidates.json",
        review_store_path=tmp_path / "reviews",
        allowed_domains=("example.edu",),
        low_supply_threshold=3,
        target_supply=6,
        default_bkt_model_version="test-v1",
        status="active",
    )


def _job(review_path) -> ReplenishmentJob:
    return ReplenishmentJob(
        job_id="promote-job-1",
        course_id="ai",
        skill_id=SKILL_ID,
        job_type="promote_approved_items",
        status="completed",
        requested_count=1,
        attempts=1,
        created_at=FIXED_TIME,
        metadata={"batch_id": BATCH_ID, "review_path": str(review_path)},
    )


def _question(stem="Which statement is correct?", correct="Correct") -> QuizQuestion:
    return QuizQuestion(
        question=stem,
        options=[correct, "Wrong A", "Wrong B", "Wrong C"],
        correct_answer=correct,
        explanation=f"{correct} is the supported choice.",
        concept="Grounded concept",
        difficulty="intermediate",
    )


def _as_written_item(question_id: str, intent_id: str) -> CurationItem:
    return CurationItem(
        original_question_id=question_id,
        skill_id=SKILL_ID,
        intent_id=intent_id,
        recommendation="approve_as_written",
        recommendation_reason="Candidate needed no changes.",
    )


def _pending_question(question_id: str, intent_id: str, q: QuizQuestion) -> PendingQuestion:
    base = q.model_dump()
    return PendingQuestion(
        batch_id=BATCH_ID,
        question_id=question_id,
        skill_id=SKILL_ID,
        question_index=0,
        intent_id=intent_id,
        seed=1,
        reference_ids=["AI-SRC-08-reference"],
        prompt_version="v3.3",
        prompt_hash="a" * 64,
        model_id="model",
        model_revision="revision",
        generation_parameters={},
        generated_at=FIXED_TIME,
        git_commit="deadbeef",
        raw_response="{}",
        question=IntentQuestion(**base, intent_id=intent_id),
    )


def _write_batch_output_dir(manifest, question_id, intent_id, q):
    output_dir = manifest.review_store_path.parent / "batches" / f"{BATCH_ID}__{SKILL_ID}"
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = _pending_question(question_id, intent_id, q)
    (output_dir / "pending_questions.jsonl").write_text(
        json.dumps(pending.model_dump(mode="json")) + "\n", encoding="utf-8"
    )
    return pending


def _report(content_hash: str, *, risk_level: str, intent_id: str) -> AutomatedReviewReport:
    return AutomatedReviewReport(
        review_id=f"report-{content_hash[:12]}",
        candidate_id="candidate",
        skill_id=SKILL_ID,
        intent_id=intent_id,
        review_policy_version="review-v1",
        reviewer_model_id="fake-reviewer",
        reviewer_model_revision="fake-rev-1",
        reviewer_prompt_version="review-v1",
        reviewer_prompt_template_hash="d" * 64,
        rendered_review_request_hash="d" * 64,
        reviewed_content_hash=content_hash,
        created_at=FIXED_TIME,
        deterministic_checks=DeterministicChecks(checks=[]),
        risk_score=0.1 if risk_level == "low" else 0.7,
        risk_level=risk_level,
        recommendation="recommend_human_approval" if risk_level == "low" else "propose_revision",
    )


def _save_review(manifest, review: GroundedReview):
    review_path = manifest.review_store_path / f"{BATCH_ID}__{SKILL_ID}.json"
    GroundedReviewStore(review_path).save(review)
    return review_path


def _report_store(manifest) -> AutomatedReviewReportStore:
    return AutomatedReviewReportStore(
        review_report_path(manifest.review_store_path, BATCH_ID, SKILL_ID)
    )


def test_all_low_risk_approve_as_written_batch_is_eligible(manifest):
    question_id, intent_id = "AI-SRC-08-q1", "AI-SRC-08-INT-01"
    source = _write_batch_output_dir(manifest, question_id, intent_id, _question())
    approved = approve_as_written(
        _as_written_item(question_id, intent_id), "reviewer", reviewed_at=FIXED_TIME
    )
    _report_store(manifest).append(
        _report(question_content_hash(source.question), risk_level="low", intent_id=intent_id)
    )
    review_path = _save_review(manifest, GroundedReview(batch_id=BATCH_ID, source_hashes={}, items=[approved]))

    result = evaluate_promotion_job(_job(review_path), manifest)

    assert result.eligible is True
    assert result.reasons == []
    assert len(result.items) == 1
    assert result.items[0].low_risk is True
    assert result.items[0].risk_level == "low"


@pytest.mark.parametrize("risk_level", ["medium", "high", "critical"])
def test_non_low_risk_item_is_not_eligible(manifest, risk_level):
    question_id, intent_id = "AI-SRC-08-q1", "AI-SRC-08-INT-01"
    source = _write_batch_output_dir(manifest, question_id, intent_id, _question())
    approved = approve_as_written(
        _as_written_item(question_id, intent_id), "reviewer", reviewed_at=FIXED_TIME
    )
    _report_store(manifest).append(
        _report(question_content_hash(source.question), risk_level=risk_level, intent_id=intent_id)
    )
    review_path = _save_review(manifest, GroundedReview(batch_id=BATCH_ID, source_hashes={}, items=[approved]))

    result = evaluate_promotion_job(_job(review_path), manifest)

    assert result.eligible is False
    assert any(question_id in reason for reason in result.reasons)


def test_approved_revision_with_low_risk_report_is_eligible(manifest):
    base_item = CurationItem(
        original_question_id="AI-SRC-08-q2",
        skill_id=SKILL_ID,
        intent_id="AI-SRC-08-INT-02",
        recommendation="propose_revision",
        recommendation_reason="Needs a concrete calculation.",
    )
    revised_question = _question("Which revised statement is correct?")
    proposed = propose_revision(
        base_item,
        _question(),
        revised_question,
        "editor",
        "Clarified the stem.",
        edited_at=FIXED_TIME,
        provenance=RevisionProvenance(
            source_batch_id=BATCH_ID, intent_id="AI-SRC-08-INT-02", skill_id=SKILL_ID,
            reference_ids=["AI-SRC-08-reference"], model_id="model", model_revision="revision",
            prompt_version="v3.3", prompt_hash="a" * 64,
        ),
    )
    approved = approve_revision(
        proposed, proposed.revisions[0].revision_id, "reviewer", reviewed_at=FIXED_TIME
    )
    revision = approved.revisions[0]

    _report_store(manifest).append(
        _report(revision.content_hash, risk_level="low", intent_id="AI-SRC-08-INT-02")
    )
    review_path = _save_review(manifest, GroundedReview(batch_id=BATCH_ID, source_hashes={}, items=[approved]))

    result = evaluate_promotion_job(_job(review_path), manifest)

    assert result.eligible is True


def test_approved_revision_with_high_risk_report_is_not_eligible(manifest):
    base_item = CurationItem(
        original_question_id="AI-SRC-08-q2",
        skill_id=SKILL_ID,
        intent_id="AI-SRC-08-INT-02",
        recommendation="propose_revision",
        recommendation_reason="Needs a concrete calculation.",
    )
    revised_question = _question("Which revised statement is correct?")
    proposed = propose_revision(
        base_item,
        _question(),
        revised_question,
        "editor",
        "Clarified the stem.",
        edited_at=FIXED_TIME,
        provenance=RevisionProvenance(
            source_batch_id=BATCH_ID, intent_id="AI-SRC-08-INT-02", skill_id=SKILL_ID,
            reference_ids=["AI-SRC-08-reference"], model_id="model", model_revision="revision",
            prompt_version="v3.3", prompt_hash="a" * 64,
        ),
    )
    approved = approve_revision(
        proposed, proposed.revisions[0].revision_id, "reviewer", reviewed_at=FIXED_TIME
    )
    revision = approved.revisions[0]

    _report_store(manifest).append(
        _report(revision.content_hash, risk_level="high", intent_id="AI-SRC-08-INT-02")
    )
    review_path = _save_review(manifest, GroundedReview(batch_id=BATCH_ID, source_hashes={}, items=[approved]))

    result = evaluate_promotion_job(_job(review_path), manifest)

    assert result.eligible is False


def test_approved_item_with_no_matching_report_is_not_eligible(manifest):
    question_id, intent_id = "AI-SRC-08-q1", "AI-SRC-08-INT-01"
    _write_batch_output_dir(manifest, question_id, intent_id, _question())
    approved = approve_as_written(
        _as_written_item(question_id, intent_id), "reviewer", reviewed_at=FIXED_TIME
    )
    review_path = _save_review(manifest, GroundedReview(batch_id=BATCH_ID, source_hashes={}, items=[approved]))

    result = evaluate_promotion_job(_job(review_path), manifest)

    assert result.eligible is False
    assert any("no automated review report on file" in reason for reason in result.reasons)


def test_no_approved_items_is_not_eligible(manifest):
    pending = _as_written_item("AI-SRC-08-q1", "AI-SRC-08-INT-01")
    review_path = _save_review(manifest, GroundedReview(batch_id=BATCH_ID, source_hashes={}, items=[pending]))

    result = evaluate_promotion_job(_job(review_path), manifest)

    assert result.eligible is False
    assert result.reasons == ["no approved items in this job's review file"]


def test_job_without_review_path_is_not_eligible(manifest):
    job = ReplenishmentJob(
        job_id="promote-job-2", course_id="ai", skill_id=SKILL_ID,
        job_type="promote_approved_items", status="completed",
        requested_count=1, attempts=1, created_at=FIXED_TIME, metadata={},
    )

    result = evaluate_promotion_job(job, manifest)

    assert result.eligible is False
    assert result.reasons == ["job has no recorded review_path"]


def test_combine_evaluations_empty_list_is_not_eligible():
    eligible, reasons = combine_evaluations([])
    assert eligible is False
    assert reasons == ["no promote_approved_items job completed this run"]


def test_combine_evaluations_all_eligible_is_true():
    evaluations = [
        AutoMergeEvaluation(job_id="job-1", eligible=True),
        AutoMergeEvaluation(job_id="job-2", eligible=True),
    ]
    eligible, reasons = combine_evaluations(evaluations)
    assert eligible is True
    assert reasons == []


def test_combine_evaluations_one_ineligible_job_blocks_the_run():
    evaluations = [
        AutoMergeEvaluation(job_id="job-1", eligible=True),
        AutoMergeEvaluation(job_id="job-2", eligible=False, reasons=["x: risk_level=medium"]),
    ]
    eligible, reasons = combine_evaluations(evaluations)
    assert eligible is False
    assert reasons == ["x: risk_level=medium"]
