"""Focused tests for real per-option assessment (authoring/review/models.py's
CompactReviewResult.option_assessments and authoring/review/response_parser.py's
handling of it), which replaced the old parser shortcut that hardcoded
AnswerAssessment.option_assessments to {} regardless of what, if anything, the model
reported about individual options (authoring/review/response_parser.py:derive_assessments,
before this change).

Proves, end to end through the real ModelBackedContentReviewer -> parse_reviewer_output
-> derive_assessments -> score_risk pipeline, that a candidate whose options are
independently judged option-by-option as multiply defensible (the real
AI-FND-04-b4cd5c51a8cab3c4 semantic-overlap case -- see
tests/test_review_ai_fnd_04_semantic_overlap_regression.py) is now blocked, while its
approved revision and a known-good control both still clear review at low risk. Also
covers the new validation surface directly: missing/duplicate/unknown option
assessments, a selected-option/assessment contradiction, and the bounded repair retry
succeeding or failing closed against these new failure shapes.
"""

import json

import pytest

from authoring.grounded_batch import GenerationOutcome
from authoring.review.config import ReviewPolicyConfig
from authoring.review.reports import AutomatedReviewReportStore
from authoring.review.response_parser import ReviewerOutputError, parse_reviewer_output
from authoring.review.reviewer import ModelBackedContentReviewer
from authoring.review.service import review_candidate
from tests.review_fnd_fixtures import (
    FND03_APPROVED_REFERENCES,
    FND03_INTENT,
    FND03_KNOWN_GOOD_CANDIDATE,
    FND03_KNOWN_GOOD_QUESTION,
    FND03_SKILL,
    FND04_APPROVED_REFERENCES,
    FND04_INTENT,
    FND04_ORIGINAL_CANDIDATE,
    FND04_ORIGINAL_QUESTION,
    FND04_REVISED_CANDIDATE,
    FND04_REVISED_QUESTION,
    FND04_SKILL,
)
from tests.review_fixtures import APPROVED_REFERENCES, CORRECTED_QUESTION, INTENT, SKILL
from tests.review_generalized_fixtures import (
    GEN_CLOSE_CANDIDATE,
    GEN_CLOSE_INTENT,
    GEN_CLOSE_QUESTION,
    GEN_CLOSE_REFERENCE,
    GEN_CLOSE_SKILL,
    GEN_MATH_CANDIDATE,
    GEN_MATH_INTENT,
    GEN_MATH_QUESTION,
    GEN_MATH_REFERENCE,
    GEN_MATH_SKILL,
    GEN_NEGATION_CANDIDATE,
    GEN_NEGATION_INTENT,
    GEN_NEGATION_QUESTION,
    GEN_NEGATION_REFERENCE,
    GEN_NEGATION_SKILL,
    GEN_PARAPHRASE_CANDIDATE,
    GEN_PARAPHRASE_INTENT,
    GEN_PARAPHRASE_QUESTION,
    GEN_PARAPHRASE_REFERENCE,
    GEN_PARAPHRASE_SKILL,
)


class _FakeBatchModel:
    """Returns one GenerationOutcome per call, in order -- the last one repeats if
    called more times than there are responses configured. Mirrors
    tests/test_review_reviewer.py's helper of the same shape."""

    def __init__(self, responses: list[str]):
        self.model_id = "fake-model"
        self.model_revision = "fake-model-rev"
        self._responses = list(responses)
        self.request_count = 0

    def generate_with_metadata(self, messages, seed, generation_parameters):
        index = min(self.request_count, len(self._responses) - 1)
        text = self._responses[index]
        self.request_count += 1
        return GenerationOutcome(
            text=text, finish_reason="stop", input_tokens=100, output_tokens=50,
            max_new_tokens=generation_parameters.get("max_new_tokens", 1000),
        )


def _compact_payload(
    *,
    selected_option_index: int,
    independent_answer_text: str,
    option_assessments: list[list],
    consulted_reference_ids: list[str],
    supporting_reference_ids: list[str],
    multiple_defensible_answers: bool = False,
) -> dict:
    return {
        "grounded": True,
        "consulted_reference_ids": consulted_reference_ids,
        "supporting_reference_ids": supporting_reference_ids,
        "selected_option_index": selected_option_index,
        "independent_answer_text": independent_answer_text,
        "no_defensible_option": False,
        "declared_answer_matches": True,
        "multiple_defensible_answers": multiple_defensible_answers,
        "option_assessments": option_assessments,
        "unsupported_claims": [],
        "contradictions": [],
        "objective_aligned": True,
        "intent_aligned": True,
        "difficulty_appropriate": True,
        "duplicate_option_pairs": [],
        "confidence": 0.9,
        "blocking_reasons": [],
        "warnings": [],
    }


# --- 1: the original four-equivalent-option AI-FND-04 candidate is blocked ----------


def test_original_semantic_overlap_candidate_is_blocked_by_real_option_assessments(tmp_path):
    """The fix: even though the reviewer's own top-level multiple_defensible_answers
    flag is (deliberately, realistically) left False here -- exactly what the real
    live-Modal run did -- judging three of the four options "defensible" restatements
    of the same proposition, one per option as the new contract requires, is now on
    its own enough for derive_assessments to derive multiple_defensible_answers=True
    and for risk.py to block the candidate as critical."""
    selected_index = FND04_ORIGINAL_QUESTION.options.index(FND04_ORIGINAL_QUESTION.correct_answer)
    payload = _compact_payload(
        selected_option_index=selected_index,
        independent_answer_text=FND04_ORIGINAL_QUESTION.correct_answer,
        option_assessments=[
            [index, "correct" if index == selected_index else "defensible"]
            for index in range(4)
        ],
        consulted_reference_ids=[r.reference_id for r in FND04_APPROVED_REFERENCES],
        supporting_reference_ids=[FND04_APPROVED_REFERENCES[1].reference_id],
        multiple_defensible_answers=False,
    )
    model = _FakeBatchModel([json.dumps(payload)])
    reviewer = ModelBackedContentReviewer(model)

    report = review_candidate(
        FND04_ORIGINAL_CANDIDATE,
        FND04_SKILL,
        FND04_INTENT,
        FND04_APPROVED_REFERENCES,
        reviewer=reviewer,
        config=ReviewPolicyConfig(reviewer_passes=1),
        report_store=AutomatedReviewReportStore(tmp_path / "reports.json"),
    )

    assert report.answer_assessment.multiple_defensible_answers is True
    assert report.risk_level == "critical"
    assert report.recommendation == "reject"
    assert report.recommendation != "recommend_human_approval"
    assert any("more than one option is defensible" in reason for reason in report.blocking_reasons)


# --- 2 & 3: the good controls remain non-critical -------------------------------------


def test_approved_revised_fnd04_candidate_clears_review_at_low_risk(tmp_path):
    """Albert's real approved revision (now textually and semantically distinct
    options) must not be penalized by the new per-option requirement -- exactly one
    option judged "correct", the rest "incorrect", is the ordinary, expected shape."""
    selected_index = FND04_REVISED_QUESTION.options.index(FND04_REVISED_QUESTION.correct_answer)
    payload = _compact_payload(
        selected_option_index=selected_index,
        independent_answer_text=FND04_REVISED_QUESTION.correct_answer,
        option_assessments=[
            [index, "correct" if index == selected_index else "incorrect"] for index in range(4)
        ],
        consulted_reference_ids=[r.reference_id for r in FND04_APPROVED_REFERENCES],
        supporting_reference_ids=[FND04_APPROVED_REFERENCES[1].reference_id],
    )
    model = _FakeBatchModel([json.dumps(payload)])
    reviewer = ModelBackedContentReviewer(model)

    report = review_candidate(
        FND04_REVISED_CANDIDATE,
        FND04_SKILL,
        FND04_INTENT,
        FND04_APPROVED_REFERENCES,
        reviewer=reviewer,
        config=ReviewPolicyConfig(reviewer_passes=1),
        report_store=AutomatedReviewReportStore(tmp_path / "reports.json"),
    )

    assert report.answer_assessment.multiple_defensible_answers is False
    assert report.risk_level == "low"
    assert report.recommendation == "recommend_human_approval"


def test_known_good_fnd03_candidate_clears_review_at_low_risk(tmp_path):
    """A real, currently-approved AI-FND-03 candidate (unrelated skill, unrelated
    failure mode) must also clear review cleanly under the new requirement."""
    selected_index = FND03_KNOWN_GOOD_QUESTION.options.index(FND03_KNOWN_GOOD_QUESTION.correct_answer)
    payload = _compact_payload(
        selected_option_index=selected_index,
        independent_answer_text=FND03_KNOWN_GOOD_QUESTION.correct_answer,
        option_assessments=[
            [index, "correct" if index == selected_index else "incorrect"] for index in range(4)
        ],
        consulted_reference_ids=[r.reference_id for r in FND03_APPROVED_REFERENCES],
        supporting_reference_ids=[r.reference_id for r in FND03_APPROVED_REFERENCES],
    )
    model = _FakeBatchModel([json.dumps(payload)])
    reviewer = ModelBackedContentReviewer(model)

    report = review_candidate(
        FND03_KNOWN_GOOD_CANDIDATE,
        FND03_SKILL,
        FND03_INTENT,
        FND03_APPROVED_REFERENCES,
        reviewer=reviewer,
        config=ReviewPolicyConfig(reviewer_passes=1),
        report_store=AutomatedReviewReportStore(tmp_path / "reports.json"),
    )

    assert report.answer_assessment.multiple_defensible_answers is False
    assert report.risk_level == "low"
    assert report.recommendation == "recommend_human_approval"


# --- 4-7: validation of the new option_assessments contract --------------------------


def test_missing_assessment_fails_closed():
    """CORRECTED_QUESTION has four options -- an option_assessments entry for only
    three of them must fail, not silently default the missing one."""
    payload = _compact_payload(
        selected_option_index=0,
        independent_answer_text=CORRECTED_QUESTION.correct_answer,
        option_assessments=[[0, "correct"], [1, "incorrect"], [2, "incorrect"]],
        consulted_reference_ids=["AI-SRC-08-REF-01"],
        supporting_reference_ids=["AI-SRC-08-REF-01"],
    )
    with pytest.raises(ReviewerOutputError, match="missing assessment"):
        parse_reviewer_output(
            json.dumps(payload), question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES
        )


def test_duplicate_assessment_fails_closed():
    """Two entries naming the same option index -- a structural contradiction, caught
    at the model-validator level (mirrors duplicate_option_pairs's own duplicate
    check) rather than in validate_compact_reviewer_output, which needs the real
    option list for everything else."""
    payload = _compact_payload(
        selected_option_index=0,
        independent_answer_text=CORRECTED_QUESTION.correct_answer,
        option_assessments=[[0, "correct"], [0, "incorrect"], [1, "incorrect"], [2, "incorrect"]],
        consulted_reference_ids=["AI-SRC-08-REF-01"],
        supporting_reference_ids=["AI-SRC-08-REF-01"],
    )
    with pytest.raises(ReviewerOutputError, match="schema validation"):
        parse_reviewer_output(
            json.dumps(payload), question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES
        )


def test_unknown_option_fails_closed():
    """An option_assessments entry naming an option index that does not exist on a
    four-option candidate (index 4) must fail, not be silently dropped."""
    payload = _compact_payload(
        selected_option_index=0,
        independent_answer_text=CORRECTED_QUESTION.correct_answer,
        option_assessments=[
            [0, "correct"], [1, "incorrect"], [2, "incorrect"], [3, "incorrect"], [4, "incorrect"],
        ],
        consulted_reference_ids=["AI-SRC-08-REF-01"],
        supporting_reference_ids=["AI-SRC-08-REF-01"],
    )
    with pytest.raises(ReviewerOutputError, match="unknown option"):
        parse_reviewer_output(
            json.dumps(payload), question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES
        )


def test_selected_option_contradiction_fails_closed():
    """selected_option_index names option 0 as the reviewer's own independent answer,
    but option_assessments judges that same option "incorrect" -- an internal
    contradiction between the selected option and its own declared-answer judgment."""
    payload = _compact_payload(
        selected_option_index=0,
        independent_answer_text=CORRECTED_QUESTION.correct_answer,
        option_assessments=[[0, "incorrect"], [1, "incorrect"], [2, "incorrect"], [3, "correct"]],
        consulted_reference_ids=["AI-SRC-08-REF-01"],
        supporting_reference_ids=["AI-SRC-08-REF-01"],
    )
    with pytest.raises(ReviewerOutputError, match="inconsistent"):
        parse_reviewer_output(
            json.dumps(payload), question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES
        )


# --- 8 & 9: the bounded repair retry against these new failure shapes ----------------


def _valid_payload() -> dict:
    return _compact_payload(
        selected_option_index=0,
        independent_answer_text=CORRECTED_QUESTION.correct_answer,
        option_assessments=[[0, "correct"], [1, "incorrect"], [2, "incorrect"], [3, "incorrect"]],
        consulted_reference_ids=["AI-SRC-08-REF-01"],
        supporting_reference_ids=["AI-SRC-08-REF-01"],
    )


def test_malformed_response_then_successful_repair():
    """A first response missing option_assessments entirely triggers the existing
    bounded repair retry; a fully-compliant second response succeeds."""
    incomplete_payload = _valid_payload()
    del incomplete_payload["option_assessments"]
    model = _FakeBatchModel([json.dumps(incomplete_payload), json.dumps(_valid_payload())])
    reviewer = ModelBackedContentReviewer(model)

    result = reviewer.review(CORRECTED_QUESTION, SKILL, INTENT, APPROVED_REFERENCES)

    assert result.request_count == 2
    assert model.request_count == 2
    assert result.answer_assessment.option_assessments == {
        CORRECTED_QUESTION.options[0]: "correct",
        CORRECTED_QUESTION.options[1]: "incorrect",
        CORRECTED_QUESTION.options[2]: "incorrect",
        CORRECTED_QUESTION.options[3]: "incorrect",
    }


def test_malformed_response_after_repair_exhaustion_fails_closed():
    """Both the original and the one bounded repair attempt have an incomplete
    option_assessments -- the repair retry is exhausted (never a second, unbounded
    retry), and the failure fails closed with both calls counted."""
    first_incomplete = _valid_payload()
    del first_incomplete["option_assessments"]
    second_incomplete = _valid_payload()
    second_incomplete["option_assessments"] = [[0, "correct"], [1, "incorrect"]]
    model = _FakeBatchModel([json.dumps(first_incomplete), json.dumps(second_incomplete)])
    reviewer = ModelBackedContentReviewer(model)

    with pytest.raises(ReviewerOutputError) as excinfo:
        reviewer.review(CORRECTED_QUESTION, SKILL, INTENT, APPROVED_REFERENCES)

    assert model.request_count == 2
    context = excinfo.value.context
    assert context is not None
    assert context.request_count == 2
    assert "missing assessment" in str(excinfo.value)


# --- Generalized (non-AI-FND-04) captured fixtures -----------------------------------
#
# The AI-FND-04 case above is one real instance of "multiple options express the same
# claim." These four prove the fix generalizes across unrelated domains and failure
# shapes -- see tests/review_generalized_fixtures.py for the full rationale. Each
# drives a hand-built compact payload (representing what a reviewer that correctly
# follows the new review-v7 ANSWERING METHODOLOGY would report) through the real
# ModelBackedContentReviewer -> review_candidate() pipeline, exactly like tests 1-3.


def _gen_report(tmp_path, candidate, skill, intent, references, question, option_assessments, *, multiple_defensible_answers=False):
    selected_index = question.options.index(question.correct_answer)
    payload = _compact_payload(
        selected_option_index=selected_index,
        independent_answer_text=question.correct_answer,
        option_assessments=option_assessments,
        consulted_reference_ids=[r.reference_id for r in references],
        supporting_reference_ids=[r.reference_id for r in references],
        multiple_defensible_answers=multiple_defensible_answers,
    )
    model = _FakeBatchModel([json.dumps(payload)])
    reviewer = ModelBackedContentReviewer(model)
    return review_candidate(
        candidate, skill, intent, references,
        reviewer=reviewer,
        config=ReviewPolicyConfig(reviewer_passes=1),
        report_store=AutomatedReviewReportStore(tmp_path / "reports.json"),
    )


def test_generalized_paraphrase_candidate_is_blocked(tmp_path):
    """Three of four options (firewall purpose) restate the same claim in different
    words -- an unrelated domain from AI-FND-04, same shape of defect."""
    report = _gen_report(
        tmp_path, GEN_PARAPHRASE_CANDIDATE, GEN_PARAPHRASE_SKILL, GEN_PARAPHRASE_INTENT,
        [GEN_PARAPHRASE_REFERENCE], GEN_PARAPHRASE_QUESTION,
        option_assessments=[[0, "correct"], [1, "defensible"], [2, "defensible"], [3, "incorrect"]],
    )
    assert report.answer_assessment.multiple_defensible_answers is True
    assert report.risk_level == "critical"
    assert report.recommendation == "reject"


def test_generalized_math_equivalent_candidate_is_blocked(tmp_path):
    """0.75 cups and 75/100 cups are the same value in different notation --
    mathematical, not textual, equivalence.

    recommendation is require_full_human_review, not reject: the hybrid option-
    equivalence gate's unit_conversion detector (authoring/review/equivalence_units.py)
    independently confirms these two option texts are the same canonical quantity, and
    per authoring/review/risk.py's policy, any credible equivalence signal always
    escalates to require_full_human_review and never auto-rejects -- this overrides
    what critical severity alone would otherwise map to. risk_level stays "critical"
    (the escalation only ever raises the level floor, never lowers it)."""
    report = _gen_report(
        tmp_path, GEN_MATH_CANDIDATE, GEN_MATH_SKILL, GEN_MATH_INTENT,
        [GEN_MATH_REFERENCE], GEN_MATH_QUESTION,
        option_assessments=[[0, "correct"], [1, "defensible"], [2, "incorrect"], [3, "incorrect"]],
    )
    assert report.answer_assessment.multiple_defensible_answers is True
    assert report.risk_level == "critical"
    assert report.recommendation == "require_full_human_review"


def test_generalized_close_distractor_candidate_is_not_falsely_blocked(tmp_path):
    """All four options are atmospheric gases (topically/lexically close), but only
    carbon dioxide is correct -- surface similarity must not trigger a false block."""
    report = _gen_report(
        tmp_path, GEN_CLOSE_CANDIDATE, GEN_CLOSE_SKILL, GEN_CLOSE_INTENT,
        [GEN_CLOSE_REFERENCE], GEN_CLOSE_QUESTION,
        option_assessments=[[0, "correct"], [1, "incorrect"], [2, "incorrect"], [3, "incorrect"]],
    )
    assert report.answer_assessment.multiple_defensible_answers is False
    assert report.risk_level == "low"
    assert report.recommendation == "recommend_human_approval"


def test_generalized_negation_candidate_is_not_falsely_blocked(tmp_path):
    """Two options share a "Yes" prefix and two share a "No" prefix, but only one of
    the four is actually true -- a negation/structural pattern is not semantic
    equivalence and must not trigger a false block."""
    report = _gen_report(
        tmp_path, GEN_NEGATION_CANDIDATE, GEN_NEGATION_SKILL, GEN_NEGATION_INTENT,
        [GEN_NEGATION_REFERENCE], GEN_NEGATION_QUESTION,
        option_assessments=[[0, "correct"], [1, "incorrect"], [2, "incorrect"], [3, "incorrect"]],
    )
    assert report.answer_assessment.multiple_defensible_answers is False
    assert report.risk_level == "low"
    assert report.recommendation == "recommend_human_approval"
