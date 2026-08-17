"""Pure parser-level tests for the compact reviewer output contract -- no reviewer,
no model, no I/O. Covers the malformed-response shapes that motivated the live-Modal
smoke test (truncation) plus the other common ways a model's raw text fails strict
JSON parsing, diagnose_truncation()'s classification of each, the
selected_option_index/no_defensible_option representation of "none of the options is
defensible", and derive_assessments()'s deterministic mapping (including the
difficulty/recall classifier)."""

import json

import pytest

from authoring.review.config import ReviewPolicyConfig
from authoring.review.deterministic import run_deterministic_checks
from authoring.review.models import CompactReviewResult, SemanticReviewResult
from authoring.review.response_parser import (
    ReviewerOutputError,
    derive_assessments,
    diagnose_truncation,
    parse_reviewer_output,
)
from authoring.review.risk import score_risk
from tests.review_fixtures import (
    APPROVED_REFERENCES,
    CORRECTED_CANDIDATE,
    CORRECTED_QUESTION,
    INTENT,
    ORIGINAL_INACCURATE_QUESTION,
    SKILL,
)


# CORRECTED_QUESTION.options[0] is CORRECTED_QUESTION.correct_answer -- see
# tests/review_fixtures.py.
def _option_assessments_correct_at(index: int) -> list[list]:
    return [[i, "correct" if i == index else "incorrect"] for i in range(4)]


def _all_incorrect_option_assessments() -> list[list]:
    return [[i, "incorrect"] for i in range(4)]


def _compact_payload(**overrides) -> dict:
    payload = {
        "grounded": True,
        "supporting_reference_ids": ["AI-SRC-08-REF-01"],
        "selected_option_index": 0,
        "independent_answer_text": CORRECTED_QUESTION.correct_answer,
        "no_defensible_option": False,
        "declared_answer_matches": True,
        "multiple_defensible_answers": False,
        "option_assessments": _option_assessments_correct_at(0),
        "unsupported_claims": [],
        "contradictions": [],
        "objective_aligned": True,
        "intent_aligned": True,
        "difficulty_appropriate": True,
        "confidence": 0.9,
        "blocking_reasons": [],
        "warnings": [],
    }
    payload.update(overrides)
    return payload


def _valid_raw_response(**overrides) -> str:
    return json.dumps(_compact_payload(**overrides))


def test_valid_compact_json_parses():
    compact = parse_reviewer_output(
        _valid_raw_response(), question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES
    )
    assert isinstance(compact, CompactReviewResult)
    assert compact.grounded is True
    assert compact.selected_option_index == 0
    assert compact.no_defensible_option is False


def test_response_truncated_before_closing_brace_fails_and_is_diagnosed():
    raw = _valid_raw_response()[:-1]  # drop the final closing brace
    with pytest.raises(ReviewerOutputError, match="ended_without_closing_json_structures"):
        parse_reviewer_output(raw, question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES)
    assert diagnose_truncation(raw) == "ended_without_closing_json_structures"


def test_response_truncated_mid_string_value_fails_and_is_diagnosed():
    raw = _valid_raw_response()
    cut = raw.index('"independent_answer_text": "') + len('"independent_answer_text": "') + 5
    raw = raw[:cut]
    with pytest.raises(ReviewerOutputError, match="ended_mid_string_value"):
        parse_reviewer_output(raw, question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES)
    assert diagnose_truncation(raw) == "ended_mid_string_value"


def test_trailing_comma_fails_and_is_not_classified_as_truncation():
    raw = _valid_raw_response()
    raw = raw[:-1] + ",}"  # inject a trailing comma before the final brace
    with pytest.raises(ReviewerOutputError):
        parse_reviewer_output(raw, question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES)
    assert diagnose_truncation(raw) == "well_formed_structure_other_parse_error"


def test_markdown_fenced_json_fails_without_any_lenient_stripping():
    raw = f"```json\n{_valid_raw_response()}\n```"
    with pytest.raises(ReviewerOutputError):
        parse_reviewer_output(raw, question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES)
    assert diagnose_truncation(raw) == "trailing_content_after_closed_structure"


def test_extra_prose_after_json_fails_without_any_lenient_extraction():
    raw = _valid_raw_response() + " Hope this helps!"
    with pytest.raises(ReviewerOutputError):
        parse_reviewer_output(raw, question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES)
    assert diagnose_truncation(raw) == "trailing_content_after_closed_structure"


def test_empty_response_is_diagnosed_as_empty():
    assert diagnose_truncation("") == "empty_response"
    assert diagnose_truncation("   ") == "empty_response"
    with pytest.raises(ReviewerOutputError):
        parse_reviewer_output("", question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES)


def test_missing_required_field_fails_schema_validation():
    raw = json.dumps({"grounded": True})
    with pytest.raises(ReviewerOutputError, match="schema validation"):
        parse_reviewer_output(raw, question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES)


# --- reference provenance: consulted vs. supporting ---------------------------------


def test_consulted_and_supporting_reference_ids_are_independent_fields():
    compact = parse_reviewer_output(
        _valid_raw_response(
            consulted_reference_ids=["AI-SRC-08-REF-01"],
            supporting_reference_ids=["AI-SRC-08-REF-01"],
        ),
        question=CORRECTED_QUESTION,
        approved_references=APPROVED_REFERENCES,
    )
    assert compact.consulted_reference_ids == ["AI-SRC-08-REF-01"]
    assert compact.supporting_reference_ids == ["AI-SRC-08-REF-01"]


def test_consulted_reference_id_need_not_be_supporting():
    """A reference can be consulted without supporting the answer -- e.g. the
    reviewer read it and concluded it was irrelevant or contradictory."""
    compact = parse_reviewer_output(
        _valid_raw_response(
            consulted_reference_ids=["AI-SRC-08-REF-01"],
            supporting_reference_ids=[],
        ),
        question=CORRECTED_QUESTION,
        approved_references=APPROVED_REFERENCES,
    )
    assert compact.consulted_reference_ids == ["AI-SRC-08-REF-01"]
    assert compact.supporting_reference_ids == []


def test_supporting_reference_ids_forbidden_when_not_grounded():
    """Item 1, second bullet: this is the exact contradiction a live-Modal run
    produced (grounded=false alongside a non-empty supporting_reference_ids) --
    now rejected as a schema failure rather than silently accepted."""
    raw = _valid_raw_response(
        grounded=False,
        supporting_reference_ids=["AI-SRC-08-REF-01"],
        declared_answer_matches=False,
    )
    with pytest.raises(ReviewerOutputError, match="schema validation"):
        parse_reviewer_output(raw, question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES)


def test_supporting_reference_ids_empty_when_not_grounded_is_valid():
    compact = parse_reviewer_output(
        _valid_raw_response(
            grounded=False,
            consulted_reference_ids=["AI-SRC-08-REF-01"],
            supporting_reference_ids=[],
            declared_answer_matches=False,
        ),
        question=CORRECTED_QUESTION,
        approved_references=APPROVED_REFERENCES,
    )
    assert compact.grounded is False
    assert compact.supporting_reference_ids == []


def test_nonexistent_consulted_reference_id_fails_like_supporting_does():
    raw = _valid_raw_response(consulted_reference_ids=["bogus-reference-id"])
    with pytest.raises(ReviewerOutputError, match="nonexistent reference"):
        parse_reviewer_output(raw, question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES)


def test_derive_assessments_maps_consulted_reference_ids():
    compact = parse_reviewer_output(
        _valid_raw_response(
            consulted_reference_ids=["AI-SRC-08-REF-01"], supporting_reference_ids=[]
        ),
        question=CORRECTED_QUESTION,
        approved_references=APPROVED_REFERENCES,
    )
    derived = derive_assessments(compact, question=CORRECTED_QUESTION, intent=INTENT)
    assert derived.grounding_assessment.consulted_reference_ids == ["AI-SRC-08-REF-01"]
    assert derived.grounding_assessment.supporting_reference_ids == []


# --- no_defensible_option representation ---------------------------------------


def test_no_defensible_option_true_with_null_index_is_a_valid_assessment():
    """Item 1: none of the four options being correct is a valid assessment, not a
    parser/schema failure."""
    compact = parse_reviewer_output(
        _valid_raw_response(
            selected_option_index=None,
            no_defensible_option=True,
            independent_answer_text="an answer not among the options",
            declared_answer_matches=False,
            option_assessments=_all_incorrect_option_assessments(),
        ),
        question=CORRECTED_QUESTION,
        approved_references=APPROVED_REFERENCES,
    )
    assert compact.no_defensible_option is True
    assert compact.selected_option_index is None


def test_no_defensible_option_true_with_non_null_index_fails_schema_validation():
    raw = _valid_raw_response(selected_option_index=0, no_defensible_option=True)
    with pytest.raises(ReviewerOutputError, match="schema validation"):
        parse_reviewer_output(raw, question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES)


def test_no_defensible_option_false_with_null_index_fails_schema_validation():
    raw = _valid_raw_response(selected_option_index=None, no_defensible_option=False)
    with pytest.raises(ReviewerOutputError, match="schema validation"):
        parse_reviewer_output(raw, question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES)


def test_out_of_range_selected_option_index_fails_schema_validation():
    """Item 4, third bullet: an arbitrary out-of-option answer that is *not*
    explicitly marked no_defensible_option still fails -- there is no silent
    fallback."""
    raw = _valid_raw_response(selected_option_index=99, no_defensible_option=False)
    with pytest.raises(ReviewerOutputError, match="does not identify an existing option"):
        parse_reviewer_output(raw, question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES)


def test_negative_selected_option_index_fails_schema_validation():
    raw = _valid_raw_response(selected_option_index=-1, no_defensible_option=False)
    with pytest.raises(ReviewerOutputError, match="schema validation"):
        parse_reviewer_output(raw, question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES)


# --- derive_assessments -------------------------------------------------------------


def test_derive_assessments_maps_compact_fields_deterministically():
    compact = parse_reviewer_output(
        _valid_raw_response(
            unsupported_claims=["a claim"],
            contradictions=["a contradiction"],
            multiple_defensible_answers=True,
        ),
        question=CORRECTED_QUESTION,
        approved_references=APPROVED_REFERENCES,
    )
    derived = derive_assessments(compact, question=CORRECTED_QUESTION, intent=INTENT)

    assert derived.grounding_assessment.grounded is True
    assert derived.grounding_assessment.independently_supported_answer is True
    assert derived.grounding_assessment.supporting_reference_ids == ["AI-SRC-08-REF-01"]
    assert derived.grounding_assessment.unsupported_claims == ["a claim"]
    assert derived.grounding_assessment.contradictions == ["a contradiction"]
    assert derived.grounding_assessment.grounding_confidence == 0.9

    assert derived.answer_assessment.selected_option_text == CORRECTED_QUESTION.correct_answer
    assert derived.answer_assessment.matches_declared_answer is True
    assert derived.answer_assessment.multiple_defensible_answers is True
    assert derived.answer_assessment.no_defensible_option is False
    assert derived.answer_assessment.answer_confidence == 0.9

    assert derived.objective_assessment.measures_declared_skill is True
    assert derived.objective_assessment.satisfies_intent_blueprint is True
    assert derived.objective_assessment.cognitive_demand == INTENT.cognitive_demand

    assert derived.difficulty_assessment.difficulty_justified is True

    assert derived.duplicate_assessment.exact_duplicate_of is None
    assert derived.duplicate_assessment.near_duplicate_of == []


def test_derive_assessments_falls_back_to_unspecified_cognitive_demand():
    compact = parse_reviewer_output(
        _valid_raw_response(), question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES
    )
    intent_without_cognitive_demand = INTENT.model_copy(update={"cognitive_demand": None})
    derived = derive_assessments(
        compact, question=CORRECTED_QUESTION, intent=intent_without_cognitive_demand
    )
    assert derived.objective_assessment.cognitive_demand == "unspecified"


# --- duplicate_option_pairs: semantic/rephrased distractor duplication --------------


def test_no_duplicates_is_the_default_and_valid():
    compact = parse_reviewer_output(
        _valid_raw_response(), question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES
    )
    assert compact.duplicate_option_pairs == []
    derived = derive_assessments(compact, question=CORRECTED_QUESTION, intent=INTENT)
    assert derived.answer_assessment.duplicate_or_rephrased_distractors == []


def test_rephrased_duplicate_pair_is_caught_semantically():
    """The case an offline calibration batch found missing: two options that say the
    same thing in different words, which normalization-based exact matching cannot
    catch, now surfaces through duplicate_option_pairs."""
    compact = parse_reviewer_output(
        _valid_raw_response(duplicate_option_pairs=[[1, 2]]),
        question=CORRECTED_QUESTION,
        approved_references=APPROVED_REFERENCES,
    )
    assert compact.duplicate_option_pairs == [(1, 2)]
    derived = derive_assessments(compact, question=CORRECTED_QUESTION, intent=INTENT)
    assert derived.answer_assessment.duplicate_or_rephrased_distractors == [
        CORRECTED_QUESTION.options[1],
        CORRECTED_QUESTION.options[2],
    ]


def test_reversed_pair_is_normalized_to_ascending_order():
    compact = parse_reviewer_output(
        _valid_raw_response(duplicate_option_pairs=[[2, 1]]),
        question=CORRECTED_QUESTION,
        approved_references=APPROVED_REFERENCES,
    )
    assert compact.duplicate_option_pairs == [(1, 2)]


def test_repeated_pair_after_normalization_fails_schema_validation():
    """[1, 2] and [2, 1] normalize to the same pair -- listing both is a repeat, not
    two distinct findings."""
    raw = _valid_raw_response(duplicate_option_pairs=[[1, 2], [2, 1]])
    with pytest.raises(ReviewerOutputError, match="schema validation"):
        parse_reviewer_output(raw, question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES)


def test_exact_repeated_pair_fails_schema_validation():
    raw = _valid_raw_response(duplicate_option_pairs=[[0, 1], [0, 1]])
    with pytest.raises(ReviewerOutputError, match="schema validation"):
        parse_reviewer_output(raw, question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES)


def test_self_pair_fails_schema_validation():
    raw = _valid_raw_response(duplicate_option_pairs=[[1, 1]])
    with pytest.raises(ReviewerOutputError, match="schema validation"):
        parse_reviewer_output(raw, question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES)


def test_out_of_range_pair_index_fails_and_is_not_silently_dropped():
    raw = _valid_raw_response(duplicate_option_pairs=[[1, 99]])
    with pytest.raises(ReviewerOutputError, match="out-of-range option"):
        parse_reviewer_output(raw, question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES)


def test_negative_pair_index_fails():
    raw = _valid_raw_response(duplicate_option_pairs=[[-1, 1]])
    with pytest.raises(ReviewerOutputError, match="out-of-range option"):
        parse_reviewer_output(raw, question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES)


@pytest.mark.parametrize(
    "malformed_pairs",
    [
        [[0, 1, 2]],  # three elements
        [[0]],  # one element
        [["a", "b"]],  # non-integer
        [0, 1],  # not nested -- a flat list of ints instead of pairs
    ],
)
def test_malformed_pair_structures_fail_schema_validation(malformed_pairs):
    raw = json.dumps(_compact_payload(duplicate_option_pairs=malformed_pairs))
    with pytest.raises(ReviewerOutputError, match="schema validation"):
        parse_reviewer_output(raw, question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES)


def test_duplicate_option_pairs_risk_policy_branch_fires():
    """The risk-policy rule this whole field exists to make reachable again."""
    compact = parse_reviewer_output(
        _valid_raw_response(duplicate_option_pairs=[[1, 2]]),
        question=CORRECTED_QUESTION,
        approved_references=APPROVED_REFERENCES,
    )
    derived = derive_assessments(compact, question=CORRECTED_QUESTION, intent=INTENT)
    result = SemanticReviewResult(
        grounding_assessment=derived.grounding_assessment,
        answer_assessment=derived.answer_assessment,
        objective_assessment=derived.objective_assessment,
        difficulty_assessment=derived.difficulty_assessment,
        duplicate_assessment=derived.duplicate_assessment,
        reviewer_model_id="fake-model",
        reviewer_model_revision="fake-rev",
        reviewer_prompt_version="review-v5",
        reviewer_prompt_template_hash="a" * 64,
        rendered_review_request_hash="a" * 64,
    )
    checks = run_deterministic_checks(CORRECTED_CANDIDATE, SKILL, INTENT, APPROVED_REFERENCES)
    decision = score_risk(checks, [result], config=ReviewPolicyConfig())
    assert decision.risk_level == "medium"
    assert decision.recommendation == "propose_revision"
    assert any("duplicate" in warning or "rephrase" in warning for warning in decision.warnings)


def test_derive_assessments_resolves_selected_option_index_to_option_text():
    compact = parse_reviewer_output(
        _valid_raw_response(
            selected_option_index=2,
            declared_answer_matches=False,
            option_assessments=_option_assessments_correct_at(2),
        ),
        question=CORRECTED_QUESTION,
        approved_references=APPROVED_REFERENCES,
    )
    derived = derive_assessments(compact, question=CORRECTED_QUESTION, intent=INTENT)
    assert derived.answer_assessment.selected_option_text == CORRECTED_QUESTION.options[2]
    assert derived.answer_assessment.no_defensible_option is False


def test_derive_assessments_forces_declared_answer_mismatch_when_no_option_is_defensible():
    """Item 3: a report where no option is defensible must never be marked as
    matching the declared answer, regardless of what the model itself reported --
    matches_declared_answer is deterministically overridden, not trusted."""
    compact = parse_reviewer_output(
        _valid_raw_response(
            selected_option_index=None,
            no_defensible_option=True,
            independent_answer_text="an answer not among the options",
            # Deliberately inconsistent model self-report -- must not leak through.
            declared_answer_matches=True,
            option_assessments=_all_incorrect_option_assessments(),
        ),
        question=ORIGINAL_INACCURATE_QUESTION,
        approved_references=APPROVED_REFERENCES,
    )
    derived = derive_assessments(compact, question=ORIGINAL_INACCURATE_QUESTION, intent=INTENT)
    assert derived.answer_assessment.no_defensible_option is True
    assert derived.answer_assessment.matches_declared_answer is False
    assert derived.answer_assessment.selected_option_text == "an answer not among the options"


# --- difficulty/recall classifier (item 5) ------------------------------------------


def test_is_definition_recall_only_true_for_an_understand_level_intent():
    """Item 5: the corrected heuristic question's intent is cognitive_demand
    "understand" -- genuine definition/interpretation recall -- and must be
    classified as such."""
    assert INTENT.cognitive_demand == "understand"
    compact = parse_reviewer_output(
        _valid_raw_response(), question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES
    )
    derived = derive_assessments(compact, question=CORRECTED_QUESTION, intent=INTENT)
    assert derived.difficulty_assessment.is_definition_recall_only is True


def test_is_definition_recall_only_false_for_an_apply_level_intent():
    apply_intent = INTENT.model_copy(update={"cognitive_demand": "apply"})
    compact = parse_reviewer_output(
        _valid_raw_response(), question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES
    )
    derived = derive_assessments(compact, question=CORRECTED_QUESTION, intent=apply_intent)
    assert derived.difficulty_assessment.is_definition_recall_only is False


def test_recall_classification_never_lowers_risk_or_blocks_when_intent_permits_it():
    """Item 5: definition recall must not be penalized when the intent explicitly
    permits it (an "understand"-level intent) -- a recall-classified, otherwise-clean
    pass must still reach low risk."""
    compact = parse_reviewer_output(
        _valid_raw_response(), question=CORRECTED_QUESTION, approved_references=APPROVED_REFERENCES
    )
    derived = derive_assessments(compact, question=CORRECTED_QUESTION, intent=INTENT)
    assert derived.difficulty_assessment.is_definition_recall_only is True

    result = SemanticReviewResult(
        grounding_assessment=derived.grounding_assessment,
        answer_assessment=derived.answer_assessment,
        objective_assessment=derived.objective_assessment,
        difficulty_assessment=derived.difficulty_assessment,
        duplicate_assessment=derived.duplicate_assessment,
        reviewer_model_id="fake-model",
        reviewer_model_revision="fake-rev",
        reviewer_prompt_version="review-v3",
        reviewer_prompt_template_hash="f" * 64,
        rendered_review_request_hash="f" * 64,
    )
    checks = run_deterministic_checks(CORRECTED_CANDIDATE, SKILL, INTENT, APPROVED_REFERENCES)
    decision = score_risk(checks, [result], config=ReviewPolicyConfig())
    assert decision.risk_level == "low"
    assert decision.recommendation == "recommend_human_approval"
