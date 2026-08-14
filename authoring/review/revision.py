"""Automated revision proposals.

generate_revision_candidate() reuses the same corrective-generation path grounded_batch's
retry loop already uses -- api.prompt_builder.build_grounded_quiz_messages's
corrective_feedback parameter -- so an automated revision is one more corrective
generation attempt driven by the automated review's findings instead of a validator
exception, through the same BatchModel/grounding-brief machinery generation already uses.

propose_automated_revision() then reuses authoring.grounded_review.propose_revision() so
a machine-authored revision is an ordinary QuestionRevision: it coexists in
CurationItem.revisions alongside any human revision, remains pending, and can only be
approved by a human calling approve_revision. This is what makes "automated review
cannot set human approval fields" true without any extra guard -- the only functions that
ever write final_review_status are approve_revision/approve_as_written/reject_item, all
of which require an explicit human reviewer string, and none is called from this module.

A ModelUnavailableError raised by the wrapped model during revision generation is left
to propagate uncaught -- the caller (authoring/replenishment/worker.py) already knows
that type and routes it to waiting_for_model, exactly like ordinary generation.
"""

from collections.abc import Callable
from datetime import datetime, timezone

from api.prompt_builder import build_grounded_quiz_messages
from api.response_parser import parse_quiz_messages
from api.schemas import QuizGenerationRequest, QuizQuestion
from authoring.grounded_batch import BatchModel, PendingQuestion
from authoring.grounded_review import CurationItem, RevisionProvenance, propose_revision
from authoring.grounding_briefs import grounding_brief
from authoring.question_intents import QuestionIntent
from authoring.review.models import AutomatedReviewReport
from taxonomy.schemas import ReferenceProvenance, SkillDefinition

AUTOMATED_REVISION_EDITOR = "automated-review-v1"


def is_localized_issue(report: AutomatedReviewReport) -> bool:
    """Whether a report's findings look like a targeted rewrite could plausibly fix
    them, rather than a fundamental problem with the item's premise or grounding.

    A small number of blocking reasons, with no reference-availability or objective
    problem among them, is treated as localized -- an item entirely missing grounding,
    or one that doesn't measure its declared skill at all, is not.
    """
    if report.risk_level == "critical" and len(report.blocking_reasons) > 2:
        return False
    unfixable_markers = (
        "not independently supported",
        "does not measure the declared skill",
        "no reviewer pass produced a usable result",
    )
    return not any(
        marker in reason for reason in report.blocking_reasons for marker in unfixable_markers
    )


def generate_revision_candidate(
    original: PendingQuestion,
    report: AutomatedReviewReport,
    skill: SkillDefinition,
    intent: QuestionIntent,
    approved_references: list[ReferenceProvenance],
    *,
    model: BatchModel,
    seed: int,
) -> QuizQuestion:
    """Ask the generation model for one corrected question, grounded the same way the
    original candidate was, using the automated review's findings as corrective
    feedback."""
    feedback = "Automated review found: " + (
        "; ".join(report.blocking_reasons) if report.blocking_reasons else "issues requiring revision"
    )
    request = QuizGenerationRequest(
        topic=skill.name,
        difficulty=original.question.difficulty,
        learning_objective=skill.learning_objective,
        question_count=1,
        reference_material=[reference.reference_material for reference in approved_references],
    )
    messages = build_grounded_quiz_messages(
        request,
        question_intent=intent,
        grounding_brief=grounding_brief(skill.skill_id),
        avoid_stems=[original.question.question],
        corrective_feedback=feedback,
    )
    raw_response = model.generate(messages, seed, {})
    response = parse_quiz_messages(raw_response)
    if len(response.questions) != 1:
        raise ValueError(f"revision generation produced {len(response.questions)} questions, expected 1")
    return response.questions[0]


def propose_automated_revision(
    item: CurationItem,
    original_question: QuizQuestion,
    edited_question: QuizQuestion,
    report: AutomatedReviewReport,
    provenance: RevisionProvenance,
    *,
    editor: str = AUTOMATED_REVISION_EDITOR,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> CurationItem:
    reason = report.blocking_reasons[0] if report.blocking_reasons else "automated review flagged this candidate"
    review_note = f"Automated revision ({report.review_id}): {reason}"
    return propose_revision(
        item,
        original_question,
        edited_question,
        editor,
        review_note,
        edited_at=clock(),
        provenance=provenance,
    )
