"""Regression fixture for the AI-FND-04-b4cd5c51a8cab3c4 semantic-overlap case.

A live Modal run (batch grounded-ai-fnd-release-v1, 2026-08-17) generated a
four-option Chinese Room question where every option restated the same underlying
proposition ("behavioral success without genuine understanding") in different words.
Albert's own review caught this; the live automated semantic reviewer did not -- its
real response (reconstructed verbatim below from
outputs/replenishment/ai/reviews/automated_review_reports/grounded-ai-fnd-release-v1__AI-FND-04.json)
reported multiple_defensible_answers=False, duplicate_or_rephrased_distractors=[], and
recommended human approval at "low" risk.

REAL_CAPTURED_REVIEW_RESULT below is that real response, preserved verbatim as a
historical record -- it predates authoring/review/models.py's CompactReviewResult.
option_assessments field (the compact reviewer output contract had no way to ask for a
per-option judgment at all yet, and authoring/review/response_parser.py's
derive_assessments() hardcoded AnswerAssessment.option_assessments to {} regardless of
what, if anything, the model said about individual options). It is not fed through the
current pipeline: replaying pre-fix output through post-fix code would not tell us
anything about whether the fix works, only that historical data is still historical.

This was not a scoring-logic defect: authoring/review/risk.py already blocked on both
multiple_defensible_answers (test_review_risk.py::test_multiple_defensible_answers_is_blocking)
and a non-empty duplicate_or_rephrased_distractors. The gap was that the reviewer
*model* was never asked to judge each option individually, so nothing forced it to
notice three of the four options were restatements of the same claim. Fixed by adding
CompactReviewResult.option_assessments (authoring/review/models.py) -- one
correctness/defensibility judgment per option, required on every live review -- and
deriving multiple_defensible_answers from it directly in derive_assessments()
(authoring/review/response_parser.py), independent of the model's own top-level flag.

test_original_candidate_is_blocked_once_every_option_is_independently_assessed below
replays the exact same candidate/skill/intent/references through the real,
post-fix pipeline (ModelBackedContentReviewer -> parse_reviewer_output ->
derive_assessments -> score_risk) with a plausible *post-fix* reviewer response -- one
that gives every option its own judgment, as the contract now requires -- and proves it
is blocked. No longer marked xfail: the live behavior this regression exists to catch
is now genuinely fixed, not merely documented as a known gap.
"""

import json

from authoring.grounded_batch import GenerationOutcome
from authoring.review.config import ReviewPolicyConfig
from authoring.review.models import (
    AnswerAssessment,
    DifficultyAssessment,
    DuplicateAssessment,
    GroundingAssessment,
    ObjectiveAssessment,
    SemanticReviewResult,
)
from authoring.review.reports import AutomatedReviewReportStore
from authoring.review.reviewer import ModelBackedContentReviewer
from authoring.review.service import review_candidate
from tests.review_fnd_fixtures import (
    FND04_APPROVED_REFERENCES as APPROVED_REFERENCES,
    FND04_INTENT as INTENT,
    FND04_ORIGINAL_CANDIDATE as ORIGINAL_SEMANTIC_OVERLAP_CANDIDATE,
    FND04_ORIGINAL_QUESTION as ORIGINAL_SEMANTIC_OVERLAP_QUESTION,
    FND04_SKILL as SKILL,
)

# Reconstructed verbatim from the real live-Modal report (not fabricated): the reviewer
# independently selected the declared answer, reported it grounded and matching, and did
# NOT flag multiple_defensible_answers or duplicate_or_rephrased_distractors -- this is
# genuinely what the reviewer returned for this exact candidate, before the
# option_assessments fix existed. Preserved for the historical record; see module
# docstring for why it is not replayed through the current pipeline.
REAL_CAPTURED_REVIEW_RESULT = SemanticReviewResult(
    grounding_assessment=GroundingAssessment(
        grounded=True,
        independently_supported_answer=True,
        consulted_reference_ids=[
            "AI-FND-04-1d9f2a7b336c",
            "AI-FND-04-0a36a4f50e30",
        ],
        supporting_reference_ids=["AI-FND-04-0a36a4f50e30"],
        supported_claims=[],
        unsupported_claims=[],
        contradictions=[],
        grounding_confidence=1.0,
    ),
    answer_assessment=AnswerAssessment(
        selected_option_text=ORIGINAL_SEMANTIC_OVERLAP_QUESTION.correct_answer,
        matches_declared_answer=True,
        option_assessments={},
        multiple_defensible_answers=False,
        obviously_signalled_answer=False,
        duplicate_or_rephrased_distractors=[],
        answer_confidence=1.0,
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
    reviewer_model_id="meta-llama/Llama-3.1-8B-Instruct",
    reviewer_model_revision="unknown",
    reviewer_prompt_version="review-v5",
    reviewer_prompt_template_hash="real-modal-run-2026-08-17",
    rendered_review_request_hash="real-modal-run-2026-08-17",
)


class _FakeBatchModel:
    """Single-response fake BatchModel -- returns the same canned compact-JSON text on
    every call. Mirrors tests/test_review_reviewer.py's helper of the same shape."""

    def __init__(self, response: str):
        self.model_id = "fake-model"
        self.model_revision = "fake-model-rev"
        self._response = response
        self.request_count = 0

    def generate_with_metadata(self, messages, seed, generation_parameters):
        self.request_count += 1
        return GenerationOutcome(
            text=self._response, finish_reason="stop", input_tokens=100, output_tokens=50,
            max_new_tokens=generation_parameters.get("max_new_tokens", 1000),
        )


def test_original_candidate_is_blocked_once_every_option_is_independently_assessed(tmp_path):
    """The genuine fix: a post-fix reviewer response for this exact candidate, judging
    each option individually as the contract now requires -- three of the four
    options (including the declared answer) judged "correct"/"defensible" restatements
    of the same claim, matching what a careful independent read of these four options
    actually shows -- is on its own enough to derive multiple_defensible_answers=True
    and block the candidate, with no reliance on the model also separately
    remembering to set the top-level flag (which, per REAL_CAPTURED_REVIEW_RESULT
    above, it did not)."""
    selected_index = ORIGINAL_SEMANTIC_OVERLAP_QUESTION.options.index(
        ORIGINAL_SEMANTIC_OVERLAP_QUESTION.correct_answer
    )
    payload = {
        "grounded": True,
        "consulted_reference_ids": [r.reference_id for r in APPROVED_REFERENCES],
        "supporting_reference_ids": [APPROVED_REFERENCES[1].reference_id],
        "selected_option_index": selected_index,
        "independent_answer_text": ORIGINAL_SEMANTIC_OVERLAP_QUESTION.correct_answer,
        "no_defensible_option": False,
        "declared_answer_matches": True,
        # Deliberately left False, matching REAL_CAPTURED_REVIEW_RESULT above -- the
        # fix must not depend on the model also getting this top-level flag right.
        "multiple_defensible_answers": False,
        "option_assessments": [
            [index, "correct" if index == selected_index else "defensible"]
            for index in range(len(ORIGINAL_SEMANTIC_OVERLAP_QUESTION.options))
        ],
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
    reviewer = ModelBackedContentReviewer(_FakeBatchModel(json.dumps(payload)))

    report = review_candidate(
        ORIGINAL_SEMANTIC_OVERLAP_CANDIDATE,
        SKILL,
        INTENT,
        APPROVED_REFERENCES,
        reviewer=reviewer,
        config=ReviewPolicyConfig(reviewer_passes=1),
        report_store=AutomatedReviewReportStore(tmp_path / "reports.json"),
    )

    assert report.recommendation != "recommend_human_approval", (
        "a candidate whose four options restate the same proposition in different "
        "words must never be recommended for human approval as clean"
    )
    assert report.answer_assessment.multiple_defensible_answers is True
    assert report.risk_level == "critical"
