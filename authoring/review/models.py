"""Immutable, versioned automated-review reports for generated question candidates.

An AutomatedReviewReport never sets human-approval fields -- CurationItem's
final_review_status stays "pending" until a human calls approve_revision,
approve_as_written, or reject_item in authoring/grounded_review.py. This module only
carries a recommendation and risk assessment that feeds that existing human workflow.

reviewed_content_hash is always the output of question_content_hash() in
authoring/grounded_review.py, so a report is tied to the exact question fields it
reviewed; changing any field changes the hash and invalidates the report as a cache hit.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

RiskLevel = Literal["low", "medium", "high", "critical"]

# Per-option correctness/defensibility judgment the reviewer must give for every
# candidate option -- see CompactReviewResult.option_assessments below.
OptionJudgment = Literal["correct", "defensible", "incorrect"]

ReviewRecommendation = Literal[
    "recommend_human_approval",
    "propose_revision",
    "reject",
    "require_full_human_review",
]


class DeterministicCheckResult(BaseModel):
    code: str = Field(min_length=1)
    passed: bool
    message: str = Field(min_length=1)
    blocking: bool


class DeterministicChecks(BaseModel):
    checks: list[DeterministicCheckResult]

    @property
    def blocking_failures(self) -> list[DeterministicCheckResult]:
        return [check for check in self.checks if check.blocking and not check.passed]

    @property
    def all_passed(self) -> bool:
        return all(check.passed for check in self.checks)


class GroundingAssessment(BaseModel):
    grounded: bool
    independently_supported_answer: bool
    # consulted_reference_ids: every reference the reviewer looked at while judging
    # this candidate, whether or not it ended up supporting the answer.
    # supporting_reference_ids: the (necessarily narrower) subset that actually
    # supports the answer -- meaningless, and forbidden below, when grounded is
    # false: a live-Modal run once returned grounded=false alongside a non-empty
    # supporting_reference_ids for the same candidate, an internally contradictory
    # response this validator now rejects outright.
    consulted_reference_ids: list[str] = Field(default_factory=list)
    supporting_reference_ids: list[str] = Field(default_factory=list)
    supported_claims: list[str] = Field(default_factory=list)
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    grounding_confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _supporting_references_require_grounded(self) -> "GroundingAssessment":
        if not self.grounded and self.supporting_reference_ids:
            raise ValueError(
                "supporting_reference_ids must be empty when grounded is false -- a "
                "reference cannot support an answer that isn't grounded"
            )
        return self


class AnswerAssessment(BaseModel):
    # Renamed from independently_selected_answer: this is *just* which candidate
    # option (or, when no_defensible_option, which free-text phrasing) the reviewer
    # picked -- a selection, not a claim about whether it's grounded. Do not confuse
    # this with GroundingAssessment.independently_supported_answer (the grounding
    # conclusion, i.e. whether that selection actually holds up against the approved
    # references) -- the two answer near-identically-named fields on different
    # assessment objects were easy to conflate; this rename plus the docstring above
    # is the fix.
    selected_option_text: str = Field(min_length=1)
    matches_declared_answer: bool
    option_assessments: dict[str, str] = Field(default_factory=dict)
    multiple_defensible_answers: bool
    obviously_signalled_answer: bool
    duplicate_or_rephrased_distractors: list[str] = Field(default_factory=list)
    answer_confidence: float = Field(ge=0.0, le=1.0)
    # True when the reviewer independently determined that none of the candidate's
    # options is a defensible answer -- a distinct, stronger finding than merely
    # disagreeing with the declared answer (that disagreement could still name a
    # *different* real option). selected_option_text still carries the reviewer's own
    # phrasing in this case (see CompactReviewResult.independent_answer_text below),
    # it just isn't one of the four options.
    # Defaults to False so existing fixtures/tests built before this field existed
    # keep constructing valid AnswerAssessment objects unchanged.
    no_defensible_option: bool = False


class ObjectiveAssessment(BaseModel):
    measures_declared_skill: bool
    satisfies_intent_blueprint: bool
    matches_objective_verb: bool
    cognitive_demand: str = Field(min_length=1)
    duplicates_another_intent: bool


class DifficultyAssessment(BaseModel):
    difficulty_justified: bool
    explanation_depth_matches_difficulty: bool
    is_definition_recall_only: bool


class DuplicateAssessment(BaseModel):
    exact_duplicate_of: str | None = None
    near_duplicate_of: list[str] = Field(default_factory=list)


class CompactReviewResult(BaseModel):
    """The reviewer model's actual output contract: only the judgments a human
    approval decision needs, never a repetition of the candidate, references,
    options, or verbose reasoning. authoring/review/response_parser.py's
    derive_assessments() deterministically expands this into the richer
    GroundingAssessment/AnswerAssessment/ObjectiveAssessment/DifficultyAssessment/
    DuplicateAssessment shapes the rest of the review layer (risk.py in particular)
    already scores against, so that scoring logic needed no changes."""

    grounded: bool
    # consulted_reference_ids: every reference the reviewer looked at while judging
    # this candidate. supporting_reference_ids: the narrower subset that actually
    # supports the answer -- must be empty when grounded is false (see the validator
    # below); a live-Modal run once returned grounded=false with a non-empty
    # supporting_reference_ids for the same candidate, which this now rejects.
    consulted_reference_ids: list[str] = Field(default_factory=list)
    supporting_reference_ids: list[str] = Field(default_factory=list)
    # A reviewer determining that none of the four options is defensible is a valid,
    # first-class outcome -- not a schema failure -- so it is represented directly
    # rather than by forcing independent_answer to equal an option that doesn't exist:
    # selected_option_index identifies the reviewer's own choice (0-based, into the
    # candidate's options) when one is defensible, independent_answer_text is always
    # the reviewer's own phrasing of what it believes the answer is (used even when no
    # option is defensible), and no_defensible_option records which case this is.
    selected_option_index: int | None = None
    independent_answer_text: str = Field(min_length=1)
    no_defensible_option: bool
    declared_answer_matches: bool
    multiple_defensible_answers: bool
    # One judgment per candidate option, keyed by its 0-based index -- the real
    # per-option assessment the model must produce, replacing the old parser shortcut
    # that hardcoded AnswerAssessment.option_assessments to {} regardless of what, if
    # anything, the model said about individual options. "correct" = the reviewer's
    # own independently-derived correct answer; "defensible" = not the reviewer's
    # primary pick but could reasonably also be argued correct (e.g. it restates the
    # same underlying claim in different words); "incorrect" = clearly wrong. Every
    # option shown to the reviewer must get exactly one entry -- completeness and
    # index bounds are checked in validate_compact_reviewer_output, which has the real
    # option list; a duplicated index is rejected right here, the one thing checkable
    # without it (mirrors duplicate_option_pairs's own validator below).
    option_assessments: list[tuple[int, OptionJudgment]] = Field(min_length=1)
    unsupported_claims: list[str] = Field(default_factory=list)
    contradictions: list[str] = Field(default_factory=list)
    objective_aligned: bool
    intent_aligned: bool
    difficulty_appropriate: bool
    # Pairs of 0-based option indices the reviewer judges semantically equivalent or
    # merely rephrased versions of each other -- the complement to deterministic
    # exact-duplicate detection (authoring/question_quality.py's duplicate_options
    # check, applied after text normalization), which cannot catch two options that
    # say the same thing in different words. [] means no rephrased duplicates found.
    # Each pair is normalized to ascending order and pairs may not repeat or name the
    # same index twice -- see the validator below; index bounds are checked
    # separately in validate_compact_reviewer_output, which has the real option list.
    duplicate_option_pairs: list[tuple[int, int]] = Field(default_factory=list)
    confidence: float = Field(ge=0.0, le=1.0)
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _duplicate_option_pairs_are_well_formed(self) -> "CompactReviewResult":
        normalized: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for pair in self.duplicate_option_pairs:
            first, second = pair
            if first == second:
                raise ValueError(
                    f"duplicate_option_pairs entry {pair!r} must name two distinct option indices"
                )
            ordered = (min(first, second), max(first, second))
            if ordered in seen:
                raise ValueError(
                    f"duplicate_option_pairs contains a repeated pair: {ordered!r}"
                )
            seen.add(ordered)
            normalized.append(ordered)
        self.duplicate_option_pairs = normalized
        return self

    @model_validator(mode="after")
    def _option_assessments_have_no_duplicate_indices(self) -> "CompactReviewResult":
        indices = [index for index, _judgment in self.option_assessments]
        if len(indices) != len(set(indices)):
            raise ValueError("option_assessments contains a duplicate option index")
        return self

    @model_validator(mode="after")
    def _selected_option_index_matches_no_defensible_option(self) -> "CompactReviewResult":
        if self.no_defensible_option:
            if self.selected_option_index is not None:
                raise ValueError(
                    "selected_option_index must be null when no_defensible_option is true"
                )
        elif self.selected_option_index is None or self.selected_option_index < 0:
            raise ValueError(
                "selected_option_index must be a non-negative index when "
                "no_defensible_option is false"
            )
        return self

    @model_validator(mode="after")
    def _supporting_references_require_grounded(self) -> "CompactReviewResult":
        if not self.grounded and self.supporting_reference_ids:
            raise ValueError(
                "supporting_reference_ids must be empty when grounded is false -- a "
                "reference cannot support an answer that isn't grounded"
            )
        return self


class SemanticReviewResult(BaseModel):
    """One reviewer pass's structured output, plus the reviewer identity that produced
    it -- this identity, not the pass content, is what a reviewer-model outage or a
    version bump changes, so it travels with the assessments rather than the report.

    reviewer_prompt_template_hash and rendered_review_request_hash used to be one
    field (reviewer_prompt_hash): the hash of the fully-rendered messages, which
    changes on every distinct candidate and so could never answer "did the prompt's
    own instructions change" on its own. reviewer_prompt_template_hash hashes only the
    static system message (instructions + schema, no candidate/skill/intent/reference
    content) -- constant across every call for a given REVIEW_PROMPT_VERSION, and the
    signal to watch for an unintended prompt-template change. rendered_review_request_
    hash hashes the full messages actually sent (system + the per-candidate user
    turn) -- unique per candidate, and the signal for exact-request reproducibility/
    auditing.

    finish_reason/input_token_count/output_token_count/configured_max_new_tokens/
    request_count are optional because FakeContentReviewer fixtures (which never make
    a real model call) construct this without them; a real ModelBackedContentReviewer
    pass always sets them."""

    grounding_assessment: GroundingAssessment
    answer_assessment: AnswerAssessment
    objective_assessment: ObjectiveAssessment
    difficulty_assessment: DifficultyAssessment
    duplicate_assessment: DuplicateAssessment
    reviewer_model_id: str = Field(min_length=1)
    reviewer_model_revision: str = Field(min_length=1)
    reviewer_prompt_version: str = Field(min_length=1)
    reviewer_prompt_template_hash: str = Field(min_length=1)
    rendered_review_request_hash: str = Field(min_length=1)
    finish_reason: str | None = None
    input_token_count: int | None = None
    output_token_count: int | None = None
    configured_max_new_tokens: int | None = None
    request_count: int = 1


EquivalenceDetector = Literal["symbolic_math", "unit_conversion", "nli_semantic"]

# "not_applicable": the detector correctly determined it does not apply to this pair
# (e.g. neither option is a numeric expression) -- the normal, common, non-error case,
# never escalates. "equivalent": a credible equivalence signal -- always escalates.
# "not_equivalent": the detector ran and found no equivalence -- never escalates.
# "error": the detector could not reach a verdict (unexpected exception, model-load
# failure, timeout) -- authoring/review/risk.py escalates this exactly like
# "equivalent", since silently treating an error as "no equivalence" would be
# indistinguishable from a false negative it can't rule out. See
# authoring/review/equivalence_gate.py.
EquivalenceVerdict = Literal["not_applicable", "equivalent", "not_equivalent", "error"]


class OptionPairEvidence(BaseModel):
    """One detector's verdict on one option pair -- the structured evidence unit the
    hybrid option-equivalence gate (authoring/review/equivalence_gate.py) produces for
    every one of a 4-option candidate's 6 pairs x 3 detectors. A gate-level failure
    (the NLI scorer's warm-up itself failing, before any pair was evaluated) is never
    represented here -- see EquivalenceAssessment.initialization_error below."""

    option_index_a: int = Field(ge=0)
    option_index_b: int = Field(ge=0)
    detector: EquivalenceDetector
    verdict: EquivalenceVerdict
    score_or_normalized_form: str = Field(min_length=1)
    reason: str = Field(min_length=1)

    @model_validator(mode="after")
    def _distinct_options(self) -> "OptionPairEvidence":
        if self.option_index_a == self.option_index_b:
            raise ValueError("option_index_a and option_index_b must differ")
        return self


class EquivalenceAssessment(BaseModel):
    """Provenance-carrying record of one hybrid-gate pass over a candidate's options.
    Additive/optional on AutomatedReviewReport (see below) so historical reports
    written before this gate existed still load unchanged."""

    gate_version: str = Field(min_length=1)
    nli_model_repository: str = Field(min_length=1)
    nli_model_revision: str = Field(min_length=1)
    nli_threshold: float = Field(ge=0.0, le=1.0)
    threshold_version: str = Field(min_length=1)
    evidence: list[OptionPairEvidence] = Field(default_factory=list)
    escalated: bool
    # Additive/optional (default None) so a report from before this field existed still
    # loads unchanged. Set when the NLI scorer's warm-up itself failed (network error,
    # cache corruption, checksum mismatch, timeout -- see
    # authoring/review/equivalence_gate.py's module docstring) before any option pair
    # could be evaluated; `evidence` is empty in that case, never fabricated
    # pair-indexed entries. Sanitized: never a filesystem path, URL, environment value,
    # or other exception-message content an underlying library might embed.
    # authoring/review/risk.py escalates on this exactly like a per-pair "error"
    # verdict -- see authoring/review/service.py's equivalence_reasons construction.
    initialization_error: str | None = None


class AutomatedReviewReport(BaseModel):
    review_id: str = Field(min_length=1)
    candidate_id: str = Field(min_length=1)
    skill_id: str = Field(min_length=1)
    intent_id: str = Field(min_length=1)
    review_policy_version: str = Field(min_length=1)
    reviewer_model_id: str = Field(min_length=1)
    reviewer_model_revision: str = Field(min_length=1)
    reviewer_prompt_version: str = Field(min_length=1)
    reviewer_prompt_template_hash: str = Field(min_length=1)
    rendered_review_request_hash: str = Field(min_length=1)
    reviewed_content_hash: str = Field(min_length=1)
    created_at: datetime
    deterministic_checks: DeterministicChecks
    grounding_assessment: GroundingAssessment | None = None
    answer_assessment: AnswerAssessment | None = None
    objective_assessment: ObjectiveAssessment | None = None
    difficulty_assessment: DifficultyAssessment | None = None
    duplicate_assessment: DuplicateAssessment | None = None
    risk_score: float = Field(ge=0.0, le=1.0)
    risk_level: RiskLevel
    recommendation: ReviewRecommendation
    blocking_reasons: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    consulted_reference_ids: list[str] = Field(default_factory=list)
    supporting_reference_ids: list[str] = Field(default_factory=list)
    finish_reason: str | None = None
    input_token_count: int | None = None
    output_token_count: int | None = None
    configured_max_new_tokens: int | None = None
    reviewer_request_count: int = 0
    equivalence_assessment: EquivalenceAssessment | None = None

    @property
    def semantic_review_performed(self) -> bool:
        """False when review_candidate's deterministic checks short-circuited before
        any reviewer.review() call was attempted (authoring/review/service.py's
        `if deterministic.blocking_failures:` branch, which always leaves
        reviewer_prompt_version at its "n/a" sentinel) -- true whenever at least one
        real semantic pass was attempted, whether or not it ultimately succeeded."""
        return self.reviewer_prompt_version != "n/a"
