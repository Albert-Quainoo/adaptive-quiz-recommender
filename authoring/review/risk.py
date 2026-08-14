"""Deterministic risk-scoring policy over review results.

The reviewer LLM never assigns the final risk decision. This module turns deterministic
checks and one or more semantic reviewer passes into a risk level and recommendation
using fixed rules, so the same inputs always produce the same decision -- the property
that makes the risk policy itself unit-testable without any model.
"""

from dataclasses import dataclass, field

from authoring.review.config import ReviewPolicyConfig
from authoring.review.models import (
    DeterministicChecks,
    ReviewRecommendation,
    RiskLevel,
    SemanticReviewResult,
)

_LEVEL_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
_LEVEL_BASE_SCORE = {"low": 0.1, "medium": 0.4, "high": 0.65, "critical": 0.9}


@dataclass(frozen=True)
class RiskDecision:
    risk_score: float
    risk_level: RiskLevel
    recommendation: ReviewRecommendation
    blocking_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _max_level(a: str, b: str) -> str:
    return a if _LEVEL_ORDER[a] >= _LEVEL_ORDER[b] else b


def _score_single_pass(
    result: SemanticReviewResult,
    *,
    config: ReviewPolicyConfig,
    known_reviewer_versions: frozenset[tuple[str, str, str]] | None,
) -> tuple[str, list[str], list[str]]:
    """Risk level for exactly one reviewer pass, plus its blocking reasons/warnings."""
    blocking: list[str] = []
    warnings: list[str] = []
    level: str = "low"

    grounding = result.grounding_assessment
    answer = result.answer_assessment
    objective = result.objective_assessment
    difficulty = result.difficulty_assessment

    if grounding.contradictions:
        level = _max_level(level, "critical")
        blocking.append("answer or explanation is contradicted by an approved reference")
    if not grounding.grounded or not grounding.independently_supported_answer:
        level = _max_level(level, "critical")
        blocking.append("answer is not independently supported by an approved reference")
    if answer.no_defensible_option:
        level = _max_level(level, "critical")
        blocking.append(
            "reviewer found no defensible option among the candidate's four options -- "
            f"the reviewer's independently selected answer ({answer.selected_option_text!r}) "
            "is absent from the options"
        )
    if not answer.matches_declared_answer:
        level = _max_level(level, "critical")
        blocking.append(
            "reviewer's independently selected answer disagrees with the declared answer"
        )
    if answer.multiple_defensible_answers:
        level = _max_level(level, "critical")
        blocking.append("more than one option is defensible as correct")

    if grounding.unsupported_claims:
        level = _max_level(level, "high")
        warnings.append(
            "explanation contains an unsupported claim: " + "; ".join(grounding.unsupported_claims)
        )
    if not objective.measures_declared_skill or not objective.satisfies_intent_blueprint:
        level = _max_level(level, "high")
        blocking.append("question does not measure the declared skill or satisfy its intent")
    if not objective.matches_objective_verb:
        level = _max_level(level, "high")
        warnings.append("question does not exercise the learning objective's verb")
    if not difficulty.difficulty_justified or not difficulty.explanation_depth_matches_difficulty:
        level = _max_level(level, "high")
        warnings.append("difficulty is not justified by the question or its explanation")
    # is_definition_recall_only is deterministic, accurate reporting metadata (derived
    # from the intent's own cognitive_demand -- see
    # authoring/review/response_parser.py:derive_assessments) and deliberately never
    # penalized here: a recall/interpretation-level intent (e.g. "understand") is
    # exactly the case where recall-only content is expected and correct, so there is
    # no independent "content vs. intent" mismatch left to score once both signals
    # come from the same field.
    if objective.duplicates_another_intent:
        level = _max_level(level, "high")
        warnings.append("question duplicates another intent")
    if answer.duplicate_or_rephrased_distractors:
        level = _max_level(level, "medium")
        warnings.append("one or more distractors duplicate or rephrase another option")
    if answer.obviously_signalled_answer:
        level = _max_level(level, "medium")
        warnings.append("the correct answer is obviously signalled by the stem")

    if grounding.grounding_confidence < config.grounding_confidence_threshold:
        level = _max_level(level, "medium")
        warnings.append(f"grounding confidence {grounding.grounding_confidence:.2f} below threshold")
    if answer.answer_confidence < config.answer_confidence_threshold:
        level = _max_level(level, "medium")
        warnings.append(f"answer confidence {answer.answer_confidence:.2f} below threshold")

    if known_reviewer_versions is not None:
        identity = (result.reviewer_model_id, result.reviewer_model_revision, result.reviewer_prompt_version)
        if identity not in known_reviewer_versions:
            level = _max_level(level, "high")
            warnings.append("novel or uncalibrated reviewer model/prompt version")

    return level, blocking, warnings


def score_risk(
    deterministic: DeterministicChecks,
    passes: list[SemanticReviewResult],
    *,
    config: ReviewPolicyConfig,
    reviewer_output_errors: list[str] | None = None,
    is_calibrated_configuration: bool = True,
    known_reviewer_versions: frozenset[tuple[str, str, str]] | None = None,
) -> RiskDecision:
    reviewer_output_errors = reviewer_output_errors or []

    if deterministic.blocking_failures:
        reasons = [f"{check.code}: {check.message}" for check in deterministic.blocking_failures]
        return RiskDecision(
            risk_score=_LEVEL_BASE_SCORE["critical"],
            risk_level="critical",
            recommendation="reject",
            blocking_reasons=reasons,
        )

    if reviewer_output_errors:
        return RiskDecision(
            risk_score=_LEVEL_BASE_SCORE["critical"],
            risk_level="critical",
            recommendation="require_full_human_review",
            blocking_reasons=list(reviewer_output_errors),
        )

    if not passes:
        return RiskDecision(
            risk_score=_LEVEL_BASE_SCORE["critical"],
            risk_level="critical",
            recommendation="require_full_human_review",
            blocking_reasons=["no reviewer pass produced a usable result"],
        )

    level = "low"
    blocking: list[str] = []
    warnings: list[str] = []
    for result in passes:
        pass_level, pass_blocking, pass_warnings = _score_single_pass(
            result, config=config, known_reviewer_versions=known_reviewer_versions
        )
        level = _max_level(level, pass_level)
        blocking.extend(reason for reason in pass_blocking if reason not in blocking)
        warnings.extend(reason for reason in pass_warnings if reason not in warnings)

    disagreement_reasons: list[str] = []
    if len(passes) > 1:
        first, *rest = passes
        for other in rest:
            if (
                first.answer_assessment.selected_option_text
                != other.answer_assessment.selected_option_text
            ):
                disagreement_reasons.append("reviewer passes disagree on the correct answer")
            if first.grounding_assessment.grounded != other.grounding_assessment.grounded:
                disagreement_reasons.append("reviewer passes disagree on whether the item is grounded")
            if (
                first.answer_assessment.multiple_defensible_answers
                != other.answer_assessment.multiple_defensible_answers
            ):
                disagreement_reasons.append("reviewer passes disagree on ambiguity")
            if (
                first.objective_assessment.measures_declared_skill
                != other.objective_assessment.measures_declared_skill
                or first.objective_assessment.satisfies_intent_blueprint
                != other.objective_assessment.satisfies_intent_blueprint
            ):
                disagreement_reasons.append("reviewer passes disagree on objective alignment")
        disagreement_reasons = sorted(set(disagreement_reasons))

    if not is_calibrated_configuration:
        level = _max_level(level, "high")
        warnings.append(
            "generator/reviewer configuration is flagged for elevated rejection/revision rates"
        )

    if disagreement_reasons:
        level = _max_level(level, "high")
        blocking = sorted(set(blocking) | set(disagreement_reasons))
        recommendation: ReviewRecommendation = "require_full_human_review"
    elif level == "critical":
        recommendation = "reject"
    elif level in ("high", "medium"):
        recommendation = "propose_revision"
    else:
        recommendation = "recommend_human_approval"

    score = min(1.0, _LEVEL_BASE_SCORE[level] + 0.02 * len(warnings))
    return RiskDecision(
        risk_score=score,
        risk_level=level,
        recommendation=recommendation,
        blocking_reasons=blocking,
        warnings=warnings,
    )
