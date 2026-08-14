"""FakeContentReviewer and ModelBackedContentReviewer tests -- no network, no real
model. Covers the compact reviewer output contract (including the
selected_option_index/no_defensible_option representation of "none of the options is
defensible"), the single malformed-output repair retry, and provenance/metadata
preservation."""

import json

import pytest

from authoring.grounded_batch import GenerationOutcome
from authoring.review.reviewer import FakeContentReviewer, ModelBackedContentReviewer, ReviewerUnavailableError
from authoring.review.response_parser import ReviewerOutputError
from tests.review_fixtures import (
    APPROVED_REFERENCES,
    CORRECTED_QUESTION,
    CORRECTED_REVIEW_RESULT,
    INTENT,
    ORIGINAL_INACCURATE_QUESTION,
    SKILL,
)


def test_fake_reviewer_returns_configured_result():
    reviewer = FakeContentReviewer({CORRECTED_QUESTION.question: CORRECTED_REVIEW_RESULT})
    result = reviewer.review(CORRECTED_QUESTION, SKILL, INTENT, APPROVED_REFERENCES)
    assert result is CORRECTED_REVIEW_RESULT
    assert reviewer.calls == [CORRECTED_QUESTION.question]


def test_fake_reviewer_raises_configured_exception():
    reviewer = FakeContentReviewer({CORRECTED_QUESTION.question: RuntimeError("boom")})
    with pytest.raises(RuntimeError, match="boom"):
        reviewer.review(CORRECTED_QUESTION, SKILL, INTENT, APPROVED_REFERENCES)


def test_fake_reviewer_missing_configuration_raises_assertion():
    reviewer = FakeContentReviewer({})
    with pytest.raises(AssertionError):
        reviewer.review(CORRECTED_QUESTION, SKILL, INTENT, APPROVED_REFERENCES)


class _FakeBatchModel:
    """Returns one GenerationOutcome per call, in order, from `responses` -- the
    last one repeats if generate_with_metadata is called more times than there are
    responses configured. Lets a test drive the first-call/repair-call sequence
    exactly."""

    def __init__(self, responses: list[str] | None = None, *, raises: Exception | None = None):
        self.model_id = "fake-model"
        self.model_revision = "fake-model-rev"
        self._responses = list(responses or [])
        self._raises = raises
        self.request_count = 0
        self.requests: list[list[dict[str, str]]] = []
        self.generation_parameters: list[dict] = []

    def generate_with_metadata(self, messages, seed, generation_parameters):
        self.requests.append(messages)
        self.generation_parameters.append(generation_parameters)
        if self._raises is not None:
            raise self._raises
        index = min(self.request_count, len(self._responses) - 1)
        text = self._responses[index]
        self.request_count += 1
        return GenerationOutcome(
            text=text,
            finish_reason="stop",
            input_tokens=123,
            output_tokens=45,
            max_new_tokens=generation_parameters.get("max_new_tokens", 1000),
        )


# CORRECTED_QUESTION.options[0] is CORRECTED_QUESTION.correct_answer -- see
# tests/review_fixtures.py.
def _compact_payload(**overrides) -> dict:
    payload = {
        "grounded": True,
        "supporting_reference_ids": ["AI-SRC-08-REF-01"],
        "selected_option_index": 0,
        "independent_answer_text": CORRECTED_QUESTION.correct_answer,
        "no_defensible_option": False,
        "declared_answer_matches": True,
        "multiple_defensible_answers": False,
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


def test_model_backed_reviewer_parses_valid_compact_output_on_first_call():
    model = _FakeBatchModel([_valid_raw_response()])
    reviewer = ModelBackedContentReviewer(model)
    result = reviewer.review(CORRECTED_QUESTION, SKILL, INTENT, APPROVED_REFERENCES)

    assert result.reviewer_model_id == "fake-model"
    assert result.reviewer_model_revision == "fake-model-rev"
    assert result.answer_assessment.selected_option_text == CORRECTED_QUESTION.correct_answer
    assert result.answer_assessment.matches_declared_answer is True
    assert result.answer_assessment.no_defensible_option is False
    assert result.grounding_assessment.grounded is True
    assert result.request_count == 1
    assert result.finish_reason == "stop"
    assert result.input_token_count == 123
    assert result.output_token_count == 45
    assert result.configured_max_new_tokens == 1000
    assert model.request_count == 1


def test_model_backed_reviewer_respects_configured_max_new_tokens():
    model = _FakeBatchModel([_valid_raw_response()])
    reviewer = ModelBackedContentReviewer(model, max_new_tokens=250)
    result = reviewer.review(CORRECTED_QUESTION, SKILL, INTENT, APPROVED_REFERENCES)
    assert model.generation_parameters[0]["max_new_tokens"] == 250
    assert result.configured_max_new_tokens == 250


def test_model_backed_reviewer_outage_raises_reviewer_unavailable():
    model = _FakeBatchModel(raises=RuntimeError("endpoint down"))
    reviewer = ModelBackedContentReviewer(model)
    with pytest.raises(ReviewerUnavailableError):
        reviewer.review(CORRECTED_QUESTION, SKILL, INTENT, APPROVED_REFERENCES)


def test_model_backed_reviewer_outage_on_repair_call_also_raises_reviewer_unavailable():
    """An outage on the *second* (repair) call must still route as an outage, not be
    swallowed into a fail-closed content error."""

    class _OutageOnSecondCall:
        model_id = "fake-model"
        model_revision = "fake-model-rev"

        def __init__(self) -> None:
            self.calls = 0

        def generate_with_metadata(self, messages, seed, generation_parameters):
            self.calls += 1
            if self.calls == 1:
                return GenerationOutcome(
                    text="not json", finish_reason="stop", input_tokens=10, output_tokens=5,
                    max_new_tokens=1000,
                )
            raise RuntimeError("endpoint down mid-repair")

    reviewer = ModelBackedContentReviewer(_OutageOnSecondCall())
    with pytest.raises(ReviewerUnavailableError):
        reviewer.review(CORRECTED_QUESTION, SKILL, INTENT, APPROVED_REFERENCES)


def test_model_backed_reviewer_malformed_json_triggers_repair_and_succeeds():
    model = _FakeBatchModel(["not json at all", _valid_raw_response()])
    reviewer = ModelBackedContentReviewer(model)
    result = reviewer.review(CORRECTED_QUESTION, SKILL, INTENT, APPROVED_REFERENCES)

    assert result.request_count == 2
    assert model.request_count == 2
    assert result.answer_assessment.matches_declared_answer is True
    # The repair turn carries the original conversation plus the bad response and a
    # correction request referencing the parser error.
    repair_messages = model.requests[1]
    assert repair_messages[-2] == {"role": "assistant", "content": "not json at all"}
    assert "could not be parsed" in repair_messages[-1]["content"]


def test_model_backed_reviewer_repair_failure_fails_closed_and_counts_both_calls():
    model = _FakeBatchModel(["not json at all", "still not json"])
    reviewer = ModelBackedContentReviewer(model)

    with pytest.raises(ReviewerOutputError) as excinfo:
        reviewer.review(CORRECTED_QUESTION, SKILL, INTENT, APPROVED_REFERENCES)

    assert model.request_count == 2
    context = excinfo.value.context
    assert context is not None
    assert context.request_count == 2
    assert context.raw_responses == ("not json at all", "still not json")
    assert context.reviewer_model_id == "fake-model"
    assert context.reviewer_model_revision == "fake-model-rev"
    assert context.reviewer_prompt_version
    assert context.reviewer_prompt_template_hash
    assert context.rendered_review_request_hash
    assert context.finish_reason == "stop"


def test_model_backed_reviewer_repair_must_pass_the_full_strict_schema():
    """The repair response must satisfy the exact same schema as the first attempt --
    a response that is valid JSON but missing required fields still fails closed."""
    model = _FakeBatchModel(["not json at all", json.dumps({"grounded": True})])
    reviewer = ModelBackedContentReviewer(model)
    with pytest.raises(ReviewerOutputError):
        reviewer.review(CORRECTED_QUESTION, SKILL, INTENT, APPROVED_REFERENCES)


def test_model_backed_reviewer_nonexistent_reference_triggers_repair():
    model = _FakeBatchModel(
        [
            _valid_raw_response(supporting_reference_ids=["bogus-reference-id"]),
            _valid_raw_response(),
        ]
    )
    reviewer = ModelBackedContentReviewer(model)
    result = reviewer.review(CORRECTED_QUESTION, SKILL, INTENT, APPROVED_REFERENCES)
    assert result.request_count == 2


def test_model_backed_reviewer_out_of_range_option_index_raises_after_failed_repair():
    """An arbitrary out-of-option answer that is *not* explicitly marked
    no_defensible_option must still fail -- selected_option_index has to identify a
    real option or the response has to say so via no_defensible_option, there is no
    third, silently-accepted possibility."""
    model = _FakeBatchModel(
        [
            _valid_raw_response(selected_option_index=99),
            _valid_raw_response(selected_option_index=99),
        ]
    )
    reviewer = ModelBackedContentReviewer(model)
    with pytest.raises(ReviewerOutputError, match="does not identify an existing option"):
        reviewer.review(CORRECTED_QUESTION, SKILL, INTENT, APPROVED_REFERENCES)


def test_model_backed_reviewer_no_defensible_option_is_a_valid_outcome_not_an_error():
    """Item 1: the reviewer determining that none of the four options is correct must
    parse as a normal, successful assessment -- not a parser/schema failure."""
    model = _FakeBatchModel(
        [
            _valid_raw_response(
                selected_option_index=None,
                no_defensible_option=True,
                independent_answer_text="The remaining cost from the current state n to a goal state",
                declared_answer_matches=False,
            )
        ]
    )
    reviewer = ModelBackedContentReviewer(model)
    result = reviewer.review(ORIGINAL_INACCURATE_QUESTION, SKILL, INTENT, APPROVED_REFERENCES)

    assert result.request_count == 1  # no repair needed -- this parsed cleanly
    assert result.answer_assessment.no_defensible_option is True
    assert result.answer_assessment.matches_declared_answer is False
    assert (
        result.answer_assessment.selected_option_text
        == "The remaining cost from the current state n to a goal state"
    )


def test_model_backed_reviewer_corrected_candidate_selects_its_correct_option():
    """Item 4, second bullet: the corrected candidate selects its correct option."""
    model = _FakeBatchModel([_valid_raw_response(selected_option_index=0, no_defensible_option=False)])
    reviewer = ModelBackedContentReviewer(model)
    result = reviewer.review(CORRECTED_QUESTION, SKILL, INTENT, APPROVED_REFERENCES)

    assert result.answer_assessment.no_defensible_option is False
    assert result.answer_assessment.selected_option_text == CORRECTED_QUESTION.correct_answer
    assert result.answer_assessment.matches_declared_answer is True
