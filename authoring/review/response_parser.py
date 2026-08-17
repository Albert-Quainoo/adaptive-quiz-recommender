"""Strict structured-output parsing for reviewer responses.

Mirrors api/response_parser.py's no-leniency style exactly: json.loads followed by
pydantic model_validate, no partial-parse fallback -- markdown fences, trailing
prose, and trailing commas are all parse failures, never silently stripped or
extracted from. A malformed-output repair retry (authoring/review/reviewer.py) asks
the model to regenerate corrected JSON instead; this module never guesses at intent
from a broken payload.

validate_reviewer_assessments() is exported separately from parse_reviewer_output()
so the same nonexistent-reference/option check applies uniformly to every reviewer,
including FakeContentReviewer in tests, which never goes through raw JSON.

derive_assessments() deterministically expands a CompactReviewResult (the model's
actual, minimal output contract) into the richer per-dimension assessment shapes
(GroundingAssessment, AnswerAssessment, ObjectiveAssessment, DifficultyAssessment,
DuplicateAssessment) that authoring/review/risk.py already scores against -- risk
scoring needed no changes when the model-facing contract was simplified, because it
still receives the same shapes, just built by a fixed mapping instead of asked for
directly.
"""

import json
from dataclasses import dataclass
from typing import NamedTuple

from pydantic import ValidationError

from api.schemas import QuizQuestion
from authoring.question_intents import QuestionIntent
from authoring.review.models import (
    AnswerAssessment,
    CompactReviewResult,
    DifficultyAssessment,
    DuplicateAssessment,
    GroundingAssessment,
    ObjectiveAssessment,
)
from taxonomy.schemas import ReferenceProvenance


@dataclass(frozen=True)
class ReviewerCallContext:
    """Everything a caller needs to preserve provenance and diagnose a failure, even
    though parsing itself failed. Travels on ReviewerOutputError so service.py can
    still record real reviewer identity and call metadata on the final report instead
    of falling back to placeholder "n/a"/"unknown" values."""

    reviewer_model_id: str
    reviewer_model_revision: str
    reviewer_prompt_version: str
    reviewer_prompt_template_hash: str
    rendered_review_request_hash: str
    finish_reason: str | None
    input_token_count: int | None
    output_token_count: int | None
    configured_max_new_tokens: int | None
    request_count: int
    raw_responses: tuple[str, ...]


class ReviewerOutputError(ValueError):
    def __init__(self, message: str, *, context: ReviewerCallContext | None = None) -> None:
        super().__init__(message)
        self.context = context


class DerivedAssessments(NamedTuple):
    grounding_assessment: GroundingAssessment
    answer_assessment: AnswerAssessment
    objective_assessment: ObjectiveAssessment
    difficulty_assessment: DifficultyAssessment
    duplicate_assessment: DuplicateAssessment


def diagnose_truncation(raw_response: str) -> str:
    """Best-effort classification of a raw reviewer response's ending, inspecting
    only string/bracket balance -- never used to alter parsing, only to explain a
    ReviewerOutputError. One of:

    - "empty_response": nothing came back.
    - "ended_mid_string_value": cut off inside an open JSON string (mid-field).
    - "ended_without_closing_json_structures": an object/array was opened but never
      closed -- classic max-tokens truncation.
    - "trailing_content_after_closed_structure": a complete top-level structure is
      followed by more text (markdown fence, extra prose, ...), not truncation.
    - "ended_mid_token": stops right after a bare word/number with no closed
      structure and no open string -- looks cut off outside any string.
    - "well_formed_structure_other_parse_error": brackets and strings both balance;
      whatever broke parsing (e.g. a trailing comma) isn't a truncation.
    """
    text = raw_response.strip()
    if not text:
        return "empty_response"

    depth = 0
    in_string = False
    escaped = False
    closed_at: int | None = None
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "{[":
            depth += 1
        elif char in "}]":
            depth -= 1
            if depth == 0 and closed_at is None:
                closed_at = index

    if in_string:
        return "ended_mid_string_value"
    if depth > 0:
        return "ended_without_closing_json_structures"
    if closed_at is not None and closed_at < len(text) - 1:
        return "trailing_content_after_closed_structure"
    if text[-1].isalnum():
        return "ended_mid_token"
    return "well_formed_structure_other_parse_error"


def validate_compact_reviewer_output(
    compact: CompactReviewResult,
    *,
    question: QuizQuestion,
    approved_references: list[ReferenceProvenance],
) -> list[str]:
    """Reasons a freshly-parsed compact reviewer output refers to something that does
    not exist. Empty means every reference id and option text the reviewer named is
    real. Used only while parsing raw JSON (parse_reviewer_output below) -- see
    validate_reviewer_assessments for the equivalent check on the derived, rich
    assessment shape every ContentReviewer (real or fake) actually returns."""
    errors: list[str] = []
    approved_reference_ids = {reference.reference_id for reference in approved_references}
    unknown_references = [
        reference_id
        for reference_id in {*compact.consulted_reference_ids, *compact.supporting_reference_ids}
        if reference_id not in approved_reference_ids
    ]
    if unknown_references:
        errors.append(
            "reviewer output cites nonexistent reference id(s): " + ", ".join(sorted(unknown_references))
        )

    # CompactReviewResult's own model_validator already guarantees
    # selected_option_index is non-negative whenever no_defensible_option is false --
    # this only needs to check that the index actually identifies an existing option.
    if not compact.no_defensible_option and not (
        0 <= compact.selected_option_index < len(question.options)
    ):
        errors.append(
            f"reviewer output's selected_option_index {compact.selected_option_index} does "
            f"not identify an existing option (candidate has {len(question.options)} options)"
        )

    out_of_range_pairs = [
        pair
        for pair in compact.duplicate_option_pairs
        if not (0 <= pair[0] < len(question.options) and 0 <= pair[1] < len(question.options))
    ]
    if out_of_range_pairs:
        errors.append(
            "reviewer output's duplicate_option_pairs reference out-of-range option "
            f"index(es) {out_of_range_pairs} (candidate has {len(question.options)} options)"
        )

    # option_assessments must cover every real option exactly once -- no fewer
    # (incomplete), no more (an unknown/hallucinated option index). Duplicated indices
    # are already rejected at the model-validator level (no real option list needed for
    # that check); this is the one part that does need it.
    expected_indices = set(range(len(question.options)))
    judgments_by_index = dict(compact.option_assessments)
    actual_indices = set(judgments_by_index)
    missing_indices = sorted(expected_indices - actual_indices)
    unknown_indices = sorted(actual_indices - expected_indices)
    if missing_indices:
        errors.append(
            "reviewer output's option_assessments is missing assessment(s) for option "
            f"index(es) {missing_indices} (candidate has {len(question.options)} options)"
        )
    if unknown_indices:
        errors.append(
            "reviewer output's option_assessments references unknown option "
            f"index(es) {unknown_indices} (candidate has {len(question.options)} options)"
        )

    if compact.no_defensible_option:
        # "None of the four options is a defensible answer" (see prompt.py) means
        # exactly that -- not merely "none is my primary pick." A "correct" or
        # "defensible" judgment anywhere here directly contradicts the claim.
        not_incorrect = sorted(
            index for index, judgment in judgments_by_index.items() if judgment != "incorrect"
        )
        if not_incorrect:
            errors.append(
                "reviewer output's no_defensible_option is true but option_assessments "
                f"judges option index(es) {not_incorrect} as not 'incorrect'"
            )
    elif compact.selected_option_index is not None and compact.selected_option_index in judgments_by_index:
        # Bounds on selected_option_index itself are checked above; only compare
        # against option_assessments when it actually names a real option, so this
        # never masks the out-of-range error with an unrelated KeyError.
        selected_judgment = judgments_by_index[compact.selected_option_index]
        if selected_judgment != "correct":
            errors.append(
                f"reviewer output selected option {compact.selected_option_index} as its "
                f"independent answer but judged it {selected_judgment!r} in "
                "option_assessments, not 'correct' -- the selected option and its own "
                "declared-answer judgment are inconsistent"
            )

    return errors


def validate_reviewer_assessments(
    grounding_assessment: GroundingAssessment,
    answer_assessment: AnswerAssessment,
    *,
    question: QuizQuestion,
    approved_references: list[ReferenceProvenance],
) -> list[str]:
    """Reasons a reviewer's *derived* output refers to something that does not exist.
    Empty means every reference id and option text the reviewer named is real. Applies
    uniformly to every ContentReviewer's returned SemanticReviewResult, including
    FakeContentReviewer in tests (which never goes through parse_reviewer_output/
    validate_compact_reviewer_output at all) -- called by
    authoring/review/service.py's review_candidate() as a reviewer-agnostic safety
    net, independent of and in addition to the compact-output check above."""
    errors: list[str] = []
    approved_reference_ids = {reference.reference_id for reference in approved_references}
    unknown_references = [
        reference_id
        for reference_id in {
            *grounding_assessment.consulted_reference_ids,
            *grounding_assessment.supporting_reference_ids,
        }
        if reference_id not in approved_reference_ids
    ]
    if unknown_references:
        errors.append(
            "reviewer output cites nonexistent reference id(s): " + ", ".join(sorted(unknown_references))
        )

    # no_defensible_option is a valid outcome where selected_option_text is
    # deliberately not one of the options -- see CompactReviewResult/AnswerAssessment.
    if (
        not answer_assessment.no_defensible_option
        and answer_assessment.selected_option_text not in question.options
    ):
        errors.append(
            "reviewer output's selected_option_text is not one of the candidate's options"
        )

    return errors


def parse_reviewer_output(
    raw_response: str,
    *,
    question: QuizQuestion,
    approved_references: list[ReferenceProvenance],
) -> CompactReviewResult:
    try:
        data = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        ending = diagnose_truncation(raw_response)
        raise ReviewerOutputError(
            f"reviewer output is not valid JSON ({ending}): {exc}"
        ) from exc
    try:
        compact = CompactReviewResult.model_validate(data)
    except ValidationError as exc:
        raise ReviewerOutputError(f"reviewer output failed schema validation: {exc}") from exc

    errors = validate_compact_reviewer_output(
        compact, question=question, approved_references=approved_references
    )
    if errors:
        raise ReviewerOutputError("; ".join(errors))

    return compact


# Bloom's-taxonomy tiers a question testing only recall/interpretation of a
# definition is expected to satisfy; anything above this (apply/analyse/evaluate/
# create) demands more than recall. Used only to classify DifficultyAssessment.
# is_definition_recall_only for reporting -- see derive_assessments's docstring for
# why this no longer feeds a risk-scoring rule.
_RECALL_LEVEL_COGNITIVE_DEMANDS = frozenset({"remember", "understand"})


def derive_assessments(
    compact: CompactReviewResult, *, question: QuizQuestion, intent: QuestionIntent
) -> DerivedAssessments:
    """Fixed, deterministic mapping from the compact contract to the detailed
    per-dimension assessments authoring/review/risk.py scores. Fields the compact
    contract does not collect at all (intent-duplication) get a fixed, conservative
    default rather than being left to the model. option_assessments and
    duplicate_or_rephrased_distractors are both derived from real model-reported
    fields (compact.option_assessments and compact.duplicate_option_pairs
    respectively) rather than defaulted: exact-text duplicates are still caught
    deterministically by authoring/review/deterministic.py, but a semantic/rephrased
    duplicate, or a semantically-equivalent-but-textually-distinct correct option,
    needs the reviewer's own per-option judgment, which is exactly what these two
    fields supply. multiple_defensible_answers is likewise strengthened, not just
    passed through: it is true if either the model's own top-level flag says so, or
    more than one option was independently judged "correct"/"defensible" -- the
    second signal exists because a model can flag one without the other (see
    tests/test_review_ai_fnd_04_semantic_overlap_regression.py).

    is_definition_recall_only is derived from intent.cognitive_demand (known, local
    intent metadata, never asked of the reviewer), not independently judged from the
    question's content -- there is no longer a model-reported signal to compare it
    against, so authoring/review/risk.py's old "recall content but an apply/analyse/
    evaluate intent" mismatch rule was removed as dead code (a demand this function
    classifies as recall-level can never simultaneously be in the mismatch rule's
    apply/analyse/evaluate set). This field is retained purely as accurate, human-
    visible reporting metadata: a recall/interpretation-level intent (e.g.
    "understand") is correctly flagged, and -- per design, not by omission -- never
    penalized, since nothing downstream keys off it.
    """
    grounding_assessment = GroundingAssessment(
        grounded=compact.grounded,
        independently_supported_answer=compact.grounded,
        consulted_reference_ids=compact.consulted_reference_ids,
        supporting_reference_ids=compact.supporting_reference_ids,
        supported_claims=[],
        unsupported_claims=compact.unsupported_claims,
        contradictions=compact.contradictions,
        grounding_confidence=compact.confidence,
    )
    if compact.no_defensible_option:
        selected_option_text = compact.independent_answer_text
        # Deterministic, not model-trusted: an answer that matches none of the four
        # options structurally cannot match the declared answer, which is itself one
        # of those options.
        matches_declared_answer = False
    else:
        selected_option_text = question.options[compact.selected_option_index]
        matches_declared_answer = compact.declared_answer_matches
    # Every option index named in any validated duplicate_option_pairs entry, mapped
    # to its text and deduplicated -- both members of a pair are implicated, not just
    # one, since duplicate_option_pairs carries no "original vs. rephrase" direction.
    implicated_option_indices = sorted(
        {index for pair in compact.duplicate_option_pairs for index in pair}
    )
    duplicate_or_rephrased_distractors = [
        question.options[index] for index in implicated_option_indices
    ]
    # The existing option_assessments schema (AnswerAssessment.option_assessments,
    # dict[str, str]) keyed by option text, populated for real now instead of the old
    # hardcoded {} -- validate_compact_reviewer_output already guarantees every option
    # index appears in compact.option_assessments exactly once.
    option_assessments = {
        question.options[index]: judgment for index, judgment in compact.option_assessments
    }
    # A candidate where the reviewer independently judged more than one option
    # "correct" or "defensible" (e.g. semantically equivalent restatements of the same
    # proposition) has multiple defensible answers regardless of whether the model's
    # own top-level multiple_defensible_answers flag caught it -- this is the fix for
    # the AI-FND-04-b4cd5c51a8cab3c4 semantic-overlap gap: the old parser shortcut
    # (option_assessments hardcoded to {}) meant there was never a second, independent
    # signal to catch a case where the model forgot to set the top-level flag.
    defensible_count = sum(
        1 for judgment in option_assessments.values() if judgment in ("correct", "defensible")
    )
    multiple_defensible_answers = compact.multiple_defensible_answers or defensible_count > 1
    answer_assessment = AnswerAssessment(
        selected_option_text=selected_option_text,
        matches_declared_answer=matches_declared_answer,
        option_assessments=option_assessments,
        multiple_defensible_answers=multiple_defensible_answers,
        obviously_signalled_answer=False,
        duplicate_or_rephrased_distractors=duplicate_or_rephrased_distractors,
        answer_confidence=compact.confidence,
        no_defensible_option=compact.no_defensible_option,
    )
    objective_assessment = ObjectiveAssessment(
        measures_declared_skill=compact.objective_aligned,
        satisfies_intent_blueprint=compact.intent_aligned,
        matches_objective_verb=compact.objective_aligned,
        # cognitive_demand is known, local intent metadata -- never asked of the
        # reviewer -- but QuestionIntent.cognitive_demand is optional while
        # ObjectiveAssessment.cognitive_demand is required, so an intent that never
        # set one still needs a deterministic, non-empty value here.
        cognitive_demand=intent.cognitive_demand or "unspecified",
        duplicates_another_intent=False,
    )
    difficulty_assessment = DifficultyAssessment(
        difficulty_justified=compact.difficulty_appropriate,
        explanation_depth_matches_difficulty=compact.difficulty_appropriate,
        is_definition_recall_only=(
            (intent.cognitive_demand or "").strip().lower() in _RECALL_LEVEL_COGNITIVE_DEMANDS
        ),
    )
    duplicate_assessment = DuplicateAssessment()

    return DerivedAssessments(
        grounding_assessment=grounding_assessment,
        answer_assessment=answer_assessment,
        objective_assessment=objective_assessment,
        difficulty_assessment=difficulty_assessment,
        duplicate_assessment=duplicate_assessment,
    )
