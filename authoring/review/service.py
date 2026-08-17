"""Automated review orchestration: deterministic checks, semantic reviewer passes, and
risk scoring, tying authoring/review/'s pure modules together into one
AutomatedReviewReport per candidate.

Why this is state-aware quality control, not consciousness: every decision here is a
fixed function of the current candidate, its deterministic checks, and however many
reviewer passes ran -- there is no persistent internal state, no self-directed goal, and
no memory beyond the immutable reports this module writes. "State-aware" means the
system tracks *pipeline* state (which stage a job is in, which content hash was already
reviewed) so it can act efficiently and avoid redundant work, not that it has any model
of itself.

Human vs. automated responsibilities: this module (and everything it calls) may only
ever produce a recommendation and risk assessment. Nothing here writes
CurationItem.final_review_status -- that field is set exclusively by
authoring.grounded_review.approve_revision/approve_as_written/reject_item, all of which
require an explicit human reviewer identity. A "low" risk_level and
"recommend_human_approval" recommendation still leaves the item pending; this milestone
does not auto-promote any conceptual item.

Risk levels: see authoring/review/risk.py. Low = every deterministic check passed, the
answer was independently verified, no ambiguity, every claim grounded, objective/intent
aligned, defensible difficulty, a known reviewer/prompt version, no duplicate, confidence
thresholds met. Medium/high = specific, named gaps. Critical = a factual, ambiguity, or
grounding failure, or a malformed/hallucinating reviewer output.

Backpressure: see authoring/review/backpressure.py and authoring/review/config.py's
ReviewPolicyConfig -- pending-per-skill and total-backlog caps stop new generation, they
never touch review or promotion of what's already pending. Reviewer calls are cached by
content hash (report_store.latest_for_hash) so an unchanged candidate is never reviewed
twice, and bounded per job tick by ReviewPolicyConfig.max_calls_per_job_tick (enforced by
the caller in authoring/replenishment/worker.py, not this module).

Modal usage: the default reviewer_provider is "fake" (FakeContentReviewer); a real
review call only happens when authoring/replenishment/cli.py is configured with
QUIZ_REVIEW_REVIEWER_PROVIDER=modal, using the same authenticated Modal endpoint
generation already uses (authoring/replenishment/modal_inference.py).

Calibration: authoring/review/reliability.py aggregates past reports into
rejection/revision rates by generator+reviewer+skill configuration; a flagged
configuration cannot reach "low" risk (is_calibrated_configuration below) until a human
recalibrates it. evaluation/review_calibration.py runs this against a held-out
positive/negative set to report precision/recall thresholds -- it is an evaluation
harness, not a training procedure, and never modifies the reviewer.

LLM-as-judge limitations: a reviewer model can hallucinate a reference or option that
does not exist (guarded by response_parser.validate_reviewer_assessments and treated as
critical/require_full_human_review), agree with itself twice, or share blind spots with
the generator model if they are the same model. Multi-pass consistency (risk.score_risk's
disagreement handling) catches instability, not shared blind spots -- a human remains the
final authority on every conceptual item.

Changing prompt/policy versions: bump REVIEW_PROMPT_VERSION in
authoring/review/prompt.py and/or ReviewPolicyConfig.review_policy_version explicitly,
add or update the corresponding regression test, and never let this module infer a new
version from its own output -- there is no online self-modification here.
"""

from collections.abc import Callable
from datetime import datetime, timezone
from uuid import uuid4

from authoring.grounded_batch import PendingQuestion
from authoring.grounded_review import question_content_hash
from authoring.question_intents import QuestionIntent
from authoring.review.config import ReviewPolicyConfig
from authoring.review.deterministic import run_deterministic_checks
from authoring.review.equivalence_gate import GATE_VERSION, escalation_reasons, evaluate_option_equivalence
from authoring.review.equivalence_nli import FakeNliScorer, NliScorer
from authoring.review.models import AutomatedReviewReport, EquivalenceAssessment, SemanticReviewResult
from authoring.review.reports import AutomatedReviewReportStore
from authoring.review.response_parser import (
    ReviewerCallContext,
    ReviewerOutputError,
    validate_reviewer_assessments,
)
from authoring.review.reviewer import ContentReviewer
from authoring.review.risk import score_risk
from taxonomy.schemas import ReferenceProvenance, SkillDefinition


def review_candidate(
    candidate: PendingQuestion,
    skill: SkillDefinition,
    intent: QuestionIntent,
    approved_references: list[ReferenceProvenance],
    *,
    reviewer: ContentReviewer,
    config: ReviewPolicyConfig,
    report_store: AutomatedReviewReportStore,
    existing_item_ids: set[str] = frozenset(),
    existing_question_texts: set[str] = frozenset(),
    accepted_answer_positions: list[int] = (),
    passes: int | None = None,
    is_calibrated_configuration: bool = True,
    known_reviewer_versions: frozenset[tuple[str, str, str]] | None = None,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    equivalence_nli_scorer: NliScorer | FakeNliScorer | None = None,
) -> AutomatedReviewReport:
    """Produce (or reuse, from cache) one AutomatedReviewReport for a candidate.

    A ReviewerUnavailableError raised by `reviewer` is left to propagate uncaught -- the
    caller (authoring/replenishment/worker.py) routes it to waiting_for_model. A malformed
    or hallucinating reviewer output (ReviewerOutputError) is caught here and folded into
    a critical, require_full_human_review report instead, since that is a content
    problem, not an outage.
    """
    content_hash = question_content_hash(candidate.question)

    cached = report_store.latest_for_hash(content_hash)
    if cached is not None:
        return cached

    deterministic = run_deterministic_checks(
        candidate,
        skill,
        intent,
        approved_references,
        existing_item_ids=existing_item_ids,
        existing_question_texts=existing_question_texts,
        accepted_answer_positions=accepted_answer_positions,
    )

    review_id = str(uuid4())
    created_at = clock()

    if deterministic.blocking_failures:
        decision = score_risk(deterministic, [], config=config)
        report = AutomatedReviewReport(
            review_id=review_id,
            candidate_id=candidate.question_id,
            skill_id=candidate.skill_id,
            intent_id=candidate.intent_id,
            review_policy_version=config.review_policy_version,
            reviewer_model_id="n/a",
            reviewer_model_revision="n/a",
            reviewer_prompt_version="n/a",
            reviewer_prompt_template_hash="n/a",
            rendered_review_request_hash="n/a",
            reviewed_content_hash=content_hash,
            created_at=created_at,
            deterministic_checks=deterministic,
            risk_score=decision.risk_score,
            risk_level=decision.risk_level,
            recommendation=decision.recommendation,
            blocking_reasons=decision.blocking_reasons,
            warnings=decision.warnings,
        )
        report_store.append(report)
        return report

    # Independent of the semantic reviewer entirely -- only needs the candidate's own
    # stem/options -- so it runs once per candidate regardless of reviewer_passes.
    # Never auto-rejects/approves/revises/promotes: risk.score_risk only ever escalates
    # to require_full_human_review on its evidence (see authoring/review/risk.py and
    # authoring/review/equivalence_gate.py's module docstrings).
    equivalence_evidence = evaluate_option_equivalence(
        candidate.question.question,
        candidate.question.options,
        nli_threshold=config.equivalence_nli_threshold,
        nli_scorer=equivalence_nli_scorer,
    )
    equivalence_reasons = escalation_reasons(equivalence_evidence)

    pass_count = passes if passes is not None else config.reviewer_passes
    results: list[SemanticReviewResult] = []
    reviewer_output_errors: list[str] = []
    # The most recent failed pass's provenance/call metadata -- a real reviewer call
    # was made and its identity, prompt hash, and token/finish_reason metadata are all
    # known even though parsing (or the hallucination check below) rejected it, so a
    # failure must not fall back to placeholder "n/a"/"unknown" values when this is
    # available. See ReviewerCallContext's docstring.
    last_error_context: ReviewerCallContext | None = None
    for pass_index in range(pass_count):
        try:
            result = reviewer.review(
                candidate.question, skill, intent, approved_references, seed=pass_index
            )
        except ReviewerOutputError as exc:
            reviewer_output_errors.append(str(exc))
            if exc.context is not None:
                last_error_context = exc.context
            continue

        errors = validate_reviewer_assessments(
            result.grounding_assessment,
            result.answer_assessment,
            question=candidate.question,
            approved_references=approved_references,
        )
        if errors:
            reviewer_output_errors.extend(errors)
            last_error_context = ReviewerCallContext(
                reviewer_model_id=result.reviewer_model_id,
                reviewer_model_revision=result.reviewer_model_revision,
                reviewer_prompt_version=result.reviewer_prompt_version,
                reviewer_prompt_template_hash=result.reviewer_prompt_template_hash,
                rendered_review_request_hash=result.rendered_review_request_hash,
                finish_reason=result.finish_reason,
                input_token_count=result.input_token_count,
                output_token_count=result.output_token_count,
                configured_max_new_tokens=result.configured_max_new_tokens,
                request_count=result.request_count,
                raw_responses=(),
            )
        else:
            results.append(result)

    decision = score_risk(
        deterministic,
        results,
        config=config,
        reviewer_output_errors=reviewer_output_errors,
        is_calibrated_configuration=is_calibrated_configuration,
        known_reviewer_versions=known_reviewer_versions,
        equivalence_escalation_reasons=equivalence_reasons,
    )

    latest = results[-1] if results else None
    # Prefer a successful pass's own provenance; failing that, a failed pass's
    # preserved provenance (a real call was still made); only fall back to
    # placeholders when no reviewer call happened at all (pass_count == 0).
    provenance_source = latest if latest is not None else last_error_context

    def _provenance(field: str, fallback):
        return getattr(provenance_source, field) if provenance_source is not None else fallback

    report = AutomatedReviewReport(
        review_id=review_id,
        candidate_id=candidate.question_id,
        skill_id=candidate.skill_id,
        intent_id=candidate.intent_id,
        review_policy_version=config.review_policy_version,
        reviewer_model_id=_provenance("reviewer_model_id", reviewer.model_id),
        reviewer_model_revision=_provenance("reviewer_model_revision", reviewer.model_revision),
        reviewer_prompt_version=_provenance("reviewer_prompt_version", "n/a"),
        reviewer_prompt_template_hash=_provenance("reviewer_prompt_template_hash", "n/a"),
        rendered_review_request_hash=_provenance("rendered_review_request_hash", "n/a"),
        reviewed_content_hash=content_hash,
        created_at=created_at,
        deterministic_checks=deterministic,
        grounding_assessment=latest.grounding_assessment if latest else None,
        answer_assessment=latest.answer_assessment if latest else None,
        objective_assessment=latest.objective_assessment if latest else None,
        difficulty_assessment=latest.difficulty_assessment if latest else None,
        duplicate_assessment=latest.duplicate_assessment if latest else None,
        risk_score=decision.risk_score,
        risk_level=decision.risk_level,
        recommendation=decision.recommendation,
        blocking_reasons=decision.blocking_reasons,
        warnings=decision.warnings,
        consulted_reference_ids=(
            latest.grounding_assessment.consulted_reference_ids if latest else []
        ),
        supporting_reference_ids=(
            latest.grounding_assessment.supporting_reference_ids if latest else []
        ),
        finish_reason=_provenance("finish_reason", None),
        input_token_count=_provenance("input_token_count", None),
        output_token_count=_provenance("output_token_count", None),
        configured_max_new_tokens=_provenance("configured_max_new_tokens", None),
        reviewer_request_count=_provenance("request_count", 0),
        equivalence_assessment=EquivalenceAssessment(
            gate_version=GATE_VERSION,
            nli_model_repository=config.equivalence_nli_model_repository,
            nli_model_revision=config.equivalence_nli_model_revision,
            nli_threshold=config.equivalence_nli_threshold,
            threshold_version=config.equivalence_threshold_version,
            evidence=equivalence_evidence,
            escalated=bool(equivalence_reasons),
        ),
    )
    report_store.append(report)
    return report
