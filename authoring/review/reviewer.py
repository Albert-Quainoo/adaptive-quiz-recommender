"""Provider-neutral semantic content reviewer.

ModelBackedContentReviewer wraps anything satisfying authoring.grounded_batch.BatchModel
plus a generate_with_metadata() method -- both LlamaBatchModel (authoring/replenishment/
worker.py) and ModalBatchModel (authoring/replenishment/modal_inference.py) implement it
-- so the review service never couples directly to Modal's HTTP transport. Swapping
providers is choosing which BatchModel to construct, exactly like
authoring/replenishment/cli.py already does for generation.

This module never imports authoring.replenishment.worker: ReviewerUnavailableError is
its own type so authoring/replenishment/worker.py can import *this* module (to wire the
automated-review job stage) without a circular import back the other way.
"""

from typing import Protocol

from api.schemas import QuizQuestion
from authoring.grounded_batch import BatchModel, GenerationOutcome, prompt_hash
from authoring.question_intents import QuestionIntent
from authoring.review.models import SemanticReviewResult
from authoring.review.prompt import REVIEW_PROMPT_VERSION, build_repair_messages, build_review_messages
from authoring.review.response_parser import (
    ReviewerCallContext,
    ReviewerOutputError,
    derive_assessments,
    parse_reviewer_output,
)
from taxonomy.schemas import ReferenceProvenance, SkillDefinition

DEFAULT_REVIEWER_MAX_NEW_TOKENS = 1000


class ReviewerUnavailableError(RuntimeError):
    """Raised when the reviewer model cannot be reached or produces no usable output.

    Routes an in-flight automated_review job to waiting_for_model, exactly like
    ModelUnavailableError routes generation -- an outage is never treated as a content
    rejection. Applies equally to the original call and the single repair-retry call:
    the model becoming unavailable mid-repair is still an outage, not a content
    problem, and must not be silently swallowed into a fail-closed parse error.
    """


class ContentReviewer(Protocol):
    model_id: str
    model_revision: str

    def review(
        self,
        question: QuizQuestion,
        skill: SkillDefinition,
        intent: QuestionIntent,
        approved_references: list[ReferenceProvenance],
        *,
        seed: int = 0,
    ) -> SemanticReviewResult: ...


class FakeContentReviewer:
    """Deterministic test double. Configured with an explicit result (or exception) per
    question text, so one test can drive several fixtures through the same reviewer."""

    def __init__(
        self,
        results: dict[str, SemanticReviewResult | Exception] | None = None,
        *,
        default: SemanticReviewResult | Exception | None = None,
        model_id: str = "fake-reviewer",
        model_revision: str = "fake-rev-1",
    ) -> None:
        self._results = dict(results or {})
        self._default = default
        self.model_id = model_id
        self.model_revision = model_revision
        self.calls: list[str] = []

    def review(
        self,
        question: QuizQuestion,
        skill: SkillDefinition,
        intent: QuestionIntent,
        approved_references: list[ReferenceProvenance],
        *,
        seed: int = 0,
    ) -> SemanticReviewResult:
        self.calls.append(question.question)
        outcome = self._results.get(question.question, self._default)
        if outcome is None:
            raise AssertionError(
                f"FakeContentReviewer has no configured result for: {question.question!r}"
            )
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class ModelBackedContentReviewer:
    """Adapts any BatchModel (local or Modal-backed) with a generate_with_metadata()
    method into a ContentReviewer.

    On a malformed/unparseable response, makes exactly one repair-retry call asking
    the model to return corrected JSON for the same schema, with the parser error
    attached -- see review() below. Never more than two network calls per review().
    """

    def __init__(self, model: BatchModel, *, max_new_tokens: int = DEFAULT_REVIEWER_MAX_NEW_TOKENS) -> None:
        self.model = model
        self.max_new_tokens = max_new_tokens

    @property
    def model_id(self) -> str:
        return self.model.model_id

    @property
    def model_revision(self) -> str:
        return self.model.model_revision

    def _call(self, messages: list[dict[str, str]], seed: int) -> GenerationOutcome:
        # Review's output budget is tracked separately from generation's (see
        # authoring/review/config.py's reviewer_max_new_tokens) so raising or
        # lowering one never silently moves the other. Both LlamaBatchModel and
        # ModalBatchModel always decode greedily (do_sample=False), so this call --
        # first or repair -- is already deterministic without any extra parameter.
        try:
            return self.model.generate_with_metadata(
                messages, seed, {"max_new_tokens": self.max_new_tokens}
            )
        except Exception as exc:
            raise ReviewerUnavailableError(str(exc)) from exc

    def review(
        self,
        question: QuizQuestion,
        skill: SkillDefinition,
        intent: QuestionIntent,
        approved_references: list[ReferenceProvenance],
        *,
        seed: int = 0,
    ) -> SemanticReviewResult:
        model_id = self.model.model_id
        messages = build_review_messages(question, skill, intent, approved_references)
        # messages[0] is the system message, which never interpolates anything
        # candidate/skill/intent/reference-specific (see prompt.py) -- hashing it
        # alone gives a value that is constant across every call for a given
        # REVIEW_PROMPT_VERSION, the signal for "did the prompt's own instructions
        # change." rendered_review_request_hash hashes the full, as-sent request and
        # is unique per candidate.
        template_hash = prompt_hash([messages[0]])
        rendered_hash = prompt_hash(messages)

        first_outcome = self._call(messages, seed)
        try:
            compact = parse_reviewer_output(
                first_outcome.text, question=question, approved_references=approved_references
            )
        except ReviewerOutputError as first_error:
            repair_messages = build_repair_messages(messages, first_outcome.text, str(first_error))
            second_outcome = self._call(repair_messages, seed)
            try:
                compact = parse_reviewer_output(
                    second_outcome.text,
                    question=question,
                    approved_references=approved_references,
                )
            except ReviewerOutputError as second_error:
                context = ReviewerCallContext(
                    reviewer_model_id=model_id,
                    reviewer_model_revision=self.model.model_revision,
                    reviewer_prompt_version=REVIEW_PROMPT_VERSION,
                    reviewer_prompt_template_hash=template_hash,
                    rendered_review_request_hash=rendered_hash,
                    finish_reason=second_outcome.finish_reason,
                    input_token_count=second_outcome.input_tokens,
                    output_token_count=second_outcome.output_tokens,
                    configured_max_new_tokens=second_outcome.max_new_tokens,
                    request_count=2,
                    raw_responses=(first_outcome.text, second_outcome.text),
                )
                raise ReviewerOutputError(
                    f"reviewer output still invalid after one repair retry: {second_error}",
                    context=context,
                ) from second_error
            outcome = second_outcome
            request_count = 2
        else:
            outcome = first_outcome
            request_count = 1

        assessments = derive_assessments(compact, question=question, intent=intent)
        return SemanticReviewResult(
            grounding_assessment=assessments.grounding_assessment,
            answer_assessment=assessments.answer_assessment,
            objective_assessment=assessments.objective_assessment,
            difficulty_assessment=assessments.difficulty_assessment,
            duplicate_assessment=assessments.duplicate_assessment,
            reviewer_model_id=model_id,
            reviewer_model_revision=self.model.model_revision,
            reviewer_prompt_version=REVIEW_PROMPT_VERSION,
            reviewer_prompt_template_hash=template_hash,
            rendered_review_request_hash=rendered_hash,
            finish_reason=outcome.finish_reason,
            input_token_count=outcome.input_tokens,
            output_token_count=outcome.output_tokens,
            configured_max_new_tokens=outcome.max_new_tokens,
            request_count=request_count,
        )
