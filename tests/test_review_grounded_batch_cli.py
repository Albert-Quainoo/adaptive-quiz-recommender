"""scripts/review_grounded_batch.py's review-card integration: list/inspect must
show automated review's recommendation, risk, and evidence alongside the existing
human curation view, without requiring an AutomatedReviewReport to exist."""

from datetime import datetime, timezone
from uuid import uuid4

import scripts.review_grounded_batch as review_cli
from authoring.grounded_review import GroundedReviewStore, load_source_questions, question_content_hash
from authoring.review.models import (
    AnswerAssessment,
    AutomatedReviewReport,
    DeterministicChecks,
    DifficultyAssessment,
    DuplicateAssessment,
    GroundingAssessment,
    ObjectiveAssessment,
)
from authoring.review.reports import AutomatedReviewReportStore
from tests.test_grounded_review import fake_source_batch

FIXED_TIME = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def _report(question) -> AutomatedReviewReport:
    return AutomatedReviewReport(
        review_id=str(uuid4()),
        candidate_id=question.question_id,
        skill_id=question.skill_id,
        intent_id=question.intent_id,
        review_policy_version="review-policy-v1",
        reviewer_model_id="fake-reviewer",
        reviewer_model_revision="fake-rev-1",
        reviewer_prompt_version="review-v1",
        reviewer_prompt_template_hash="d" * 64,
        rendered_review_request_hash="d" * 64,
        reviewed_content_hash=question_content_hash(question.question),
        created_at=FIXED_TIME,
        deterministic_checks=DeterministicChecks(checks=[]),
        grounding_assessment=GroundingAssessment(
            grounded=True, independently_supported_answer=True, grounding_confidence=0.9
        ),
        answer_assessment=AnswerAssessment(
            selected_option_text=question.question.correct_answer,
            matches_declared_answer=True,
            multiple_defensible_answers=False,
            obviously_signalled_answer=False,
            answer_confidence=0.9,
        ),
        objective_assessment=ObjectiveAssessment(
            measures_declared_skill=True,
            satisfies_intent_blueprint=True,
            matches_objective_verb=True,
            cognitive_demand="understand",
            duplicates_another_intent=False,
        ),
        difficulty_assessment=DifficultyAssessment(
            difficulty_justified=True,
            explanation_depth_matches_difficulty=True,
            is_definition_recall_only=False,
        ),
        duplicate_assessment=DuplicateAssessment(),
        risk_score=0.1,
        risk_level="low",
        recommendation="recommend_human_approval",
        supporting_reference_ids=[],
    )


def _write_review_and_report(tmp_path):
    batch, review = fake_source_batch(tmp_path)
    store_path = tmp_path / f"{review.batch_id}__AI-SRC-08.json"
    GroundedReviewStore(store_path).save(review)

    source = load_source_questions(batch)[0]
    report = _report(source)
    AutomatedReviewReportStore(review_cli._reports_path(store_path)).append(report)
    return batch, store_path, source, report


def test_inspect_prints_the_review_card_when_a_report_exists(tmp_path, capsys):
    batch, store_path, source, report = _write_review_and_report(tmp_path)

    exit_code = review_cli.main(
        ["--batch", str(batch), "--store", str(store_path), "inspect", source.question_id]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Automated recommendation: recommend_human_approval" in output
    assert "Risk: low" in output
    assert source.question.question in output


def test_inspect_still_works_when_no_report_exists_yet(tmp_path, capsys):
    batch, review = fake_source_batch(tmp_path)
    store_path = tmp_path / f"{review.batch_id}__AI-SRC-08.json"
    GroundedReviewStore(store_path).save(review)
    question_id = review.items[0].original_question_id

    exit_code = review_cli.main(
        ["--batch", str(batch), "--store", str(store_path), "inspect", question_id]
    )
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Automated review: not yet available." in output


def test_list_shows_the_automated_recommendation_and_risk(tmp_path, capsys):
    batch, store_path, source, report = _write_review_and_report(tmp_path)

    exit_code = review_cli.main(["--batch", str(batch), "--store", str(store_path), "list"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert f"auto=recommend_human_approval/low" in output
    assert source.question_id in output
