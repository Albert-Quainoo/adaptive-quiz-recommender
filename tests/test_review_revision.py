"""Automated revision proposal tests: reuse of propose_revision(), and corrective
generation reusing the grounded-generation prompt/parse path."""

import json
from datetime import datetime, timezone
from uuid import uuid4

import pytest

from authoring.grounded_review import CurationItem, RevisionProvenance
from authoring.review.deterministic import run_deterministic_checks
from authoring.review.config import ReviewPolicyConfig
from authoring.review.models import AutomatedReviewReport
from authoring.review.revision import (
    generate_revision_candidate,
    is_localized_issue,
    propose_automated_revision,
)
from authoring.review.risk import score_risk
from tests.review_fixtures import (
    APPROVED_REFERENCES,
    CORRECTED_QUESTION,
    INTENT,
    ORIGINAL_INACCURATE_CANDIDATE,
    SKILL,
)

FIXED_TIME = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def _report(blocking_reasons: list[str], risk_level: str = "high") -> AutomatedReviewReport:
    checks = run_deterministic_checks(
        ORIGINAL_INACCURATE_CANDIDATE, SKILL, INTENT, APPROVED_REFERENCES
    )
    return AutomatedReviewReport(
        review_id=str(uuid4()),
        candidate_id=ORIGINAL_INACCURATE_CANDIDATE.question_id,
        skill_id="AI-SRC-08",
        intent_id=INTENT.intent_id,
        review_policy_version="review-policy-v1",
        reviewer_model_id="fake-reviewer",
        reviewer_model_revision="fake-rev-1",
        reviewer_prompt_version="review-v1",
        reviewer_prompt_template_hash="c" * 64,
        rendered_review_request_hash="c" * 64,
        reviewed_content_hash="hash-1",
        created_at=FIXED_TIME,
        deterministic_checks=checks,
        risk_score=0.65,
        risk_level=risk_level,
        recommendation="propose_revision",
        blocking_reasons=blocking_reasons,
    )


def test_is_localized_issue_true_for_a_small_named_gap():
    report = _report(["explanation contains an unsupported claim: foo"])
    assert is_localized_issue(report)


def test_is_localized_issue_false_when_grounding_is_entirely_missing():
    report = _report(
        ["answer is not independently supported by an approved reference"], risk_level="critical"
    )
    assert not is_localized_issue(report)


def test_is_localized_issue_false_for_many_critical_findings():
    report = _report(
        ["reason one", "reason two", "reason three"],
        risk_level="critical",
    )
    assert not is_localized_issue(report)


class _FakeRevisionModel:
    model_id = "fake-generator"
    model_revision = "fake-generator-rev"

    def __init__(self, question):
        self._question = question

    def generate(self, messages, seed, generation_parameters):
        return json.dumps({"questions": [self._question.model_dump()]})


def test_generate_revision_candidate_reuses_grounded_generation_path():
    model = _FakeRevisionModel(CORRECTED_QUESTION)
    report = _report(["declared answer conflates path cost with the heuristic"])
    revised = generate_revision_candidate(
        ORIGINAL_INACCURATE_CANDIDATE,
        report,
        SKILL,
        INTENT,
        APPROVED_REFERENCES,
        model=model,
        seed=1,
    )
    assert revised.correct_answer == CORRECTED_QUESTION.correct_answer


def test_generate_revision_candidate_raises_on_wrong_question_count():
    class _EmptyModel:
        model_id = "fake-generator"
        model_revision = "fake-generator-rev"

        def generate(self, messages, seed, generation_parameters):
            return json.dumps({"questions": []})

    report = _report(["some issue"])
    with pytest.raises(ValueError):
        generate_revision_candidate(
            ORIGINAL_INACCURATE_CANDIDATE,
            report,
            SKILL,
            INTENT,
            APPROVED_REFERENCES,
            model=_EmptyModel(),
            seed=1,
        )


def test_propose_automated_revision_attaches_a_pending_revision_with_distinct_editor():
    item = CurationItem(
        original_question_id=ORIGINAL_INACCURATE_CANDIDATE.question_id,
        skill_id="AI-SRC-08",
        intent_id=INTENT.intent_id,
        recommendation="propose_revision",
        recommendation_reason="pending automated review",
    )
    report = _report(["declared answer conflates path cost with the heuristic"])
    provenance = RevisionProvenance.from_source(ORIGINAL_INACCURATE_CANDIDATE)

    updated = propose_automated_revision(
        item,
        ORIGINAL_INACCURATE_CANDIDATE.question,
        CORRECTED_QUESTION,
        report,
        provenance,
        clock=lambda: FIXED_TIME,
    )

    assert len(updated.revisions) == 1
    revision = updated.revisions[0]
    assert revision.editor == "automated-review-v1"
    assert revision.final_review_status == "pending"
    assert revision.question.correct_answer == CORRECTED_QUESTION.correct_answer
    # Automated review never sets final_review_status on the item itself.
    assert updated.final_review_status == "pending"


def test_propose_automated_revision_never_auto_approves():
    item = CurationItem(
        original_question_id=ORIGINAL_INACCURATE_CANDIDATE.question_id,
        skill_id="AI-SRC-08",
        intent_id=INTENT.intent_id,
        recommendation="propose_revision",
        recommendation_reason="pending automated review",
    )
    report = _report(["declared answer conflates path cost with the heuristic"])
    provenance = RevisionProvenance.from_source(ORIGINAL_INACCURATE_CANDIDATE)
    updated = propose_automated_revision(
        item, ORIGINAL_INACCURATE_CANDIDATE.question, CORRECTED_QUESTION, report, provenance
    )
    assert updated.revisions[0].reviewed_by is None
    assert updated.revisions[0].reviewed_at is None
