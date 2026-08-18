"""Read-only content inventory for every active skill in one course.

Nothing here writes anything. It answers "what does this skill have right
now" by reading the same files and job rows the rest of the pipeline writes,
so the admin view, the policy, and the CLI `status` command all see one
consistent picture.
"""

from pydantic import BaseModel, Field
from typing import Literal, Protocol

from app.bootstrap import BootstrapError, load_approved_bank
from authoring.grounded_review import GroundedReview
from authoring.replenishment.jobs import ReplenishmentJob
from authoring.replenishment.manifest import CourseManifest, active_bank_path
from authoring.retrieval.store import CandidateStore
from taxonomy.loader import load_skills
from taxonomy.schemas import SkillDefinition
from templates.registry import TEMPLATES

ReadinessStatus = Literal[
    "taxonomy_only",
    "retrieval_pending",
    "reference_review",
    "generation_pending",
    "question_review",
    "ready",
    "template_ready",
    "content_exhausted",
    "replenishment_failed",
]

TemplateStatus = Literal["not_applicable", "implemented", "unimplemented"]


class JobRepositoryProtocol(Protocol):
    def latest_for_skill(
        self, course_id: str, skill_id: str
    ) -> ReplenishmentJob | None: ...


class SkillInventory(BaseModel):
    course_id: str
    skill_id: str
    generation_strategy: str
    total_approved_items: int = Field(ge=0)
    unseen_approved_items: int | None = Field(default=None, ge=0)
    pending_reference_candidates: int = Field(ge=0)
    approved_reference_candidates: int = Field(ge=0)
    pending_generated_questions: int = Field(ge=0)
    template_status: TemplateStatus
    latest_job: ReplenishmentJob | None = None
    readiness: ReadinessStatus


def derive_readiness(
    *,
    generation_strategy: str,
    template_status: TemplateStatus,
    total_approved_items: int,
    unseen_approved_items: int | None,
    low_supply_threshold: int,
    pending_reference_candidates: int,
    approved_reference_candidates: int,
    pending_generated_questions: int,
    latest_job: ReplenishmentJob | None,
) -> ReadinessStatus:
    if generation_strategy == "templated":
        return "template_ready" if template_status == "implemented" else "content_exhausted"

    if latest_job is not None:
        if latest_job.status in ("retryable_failure", "permanent_failure"):
            return "replenishment_failed"
        if latest_job.status == "waiting_for_reference_review":
            return "reference_review"
        if latest_job.status == "waiting_for_model":
            return "generation_pending"
        if latest_job.status in (
            "waiting_for_question_review",
            "waiting_for_full_human_review",
            "rejected_by_automated_review",
            "rejected_deterministically",
        ):
            # All four mean the same thing from a supply-policy perspective: a human
            # needs to act on already-generated output, so no new generation should be
            # enqueued. authoring/review/card.py's priority ordering is what
            # distinguishes them for the reviewer.
            return "question_review"
        if latest_job.status in ("queued", "running"):
            if latest_job.job_type in (
                "generate_questions",
                "validate_questions",
                "automated_review",
                "automated_revision",
                "promote_approved_items",
            ):
                return "generation_pending"
            return "retrieval_pending"
        # completed/cancelled/no_longer_needed fall through to a fresh
        # supply-based reading -- no_longer_needed deliberately does NOT
        # join the ("retryable_failure", "permanent_failure") branch above:
        # unlike a genuine error, it means this job's target demand was
        # already fully satisfied, and the skill must stay eligible for a
        # later scan to re-evaluate once the blueprint or bank changes,
        # never permanently "replenishment_failed".

    supply = (
        unseen_approved_items if unseen_approved_items is not None else total_approved_items
    )
    if supply >= low_supply_threshold:
        return "ready"

    touched = (
        total_approved_items > 0
        or pending_reference_candidates > 0
        or approved_reference_candidates > 0
        or pending_generated_questions > 0
        or latest_job is not None
    )
    return "content_exhausted" if touched else "taxonomy_only"


def _template_status(skill: SkillDefinition) -> TemplateStatus:
    if skill.generation_strategy != "templated":
        return "not_applicable"
    return "implemented" if skill.template_id in TEMPLATES else "unimplemented"


def _pending_questions_by_skill(review_store_path) -> dict[str, int]:
    counts: dict[str, int] = {}
    if not review_store_path.is_dir():
        return counts
    for path in sorted(review_store_path.glob("*.json")):
        try:
            review = GroundedReview.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for item in review.items:
            if item.final_review_status == "pending":
                counts[item.skill_id] = counts.get(item.skill_id, 0) + 1
    return counts


def compute_course_inventory(
    manifest: CourseManifest,
    job_repository: JobRepositoryProtocol,
    *,
    answered_item_ids: set[str] | None = None,
) -> dict[str, SkillInventory]:
    catalogue = load_skills(manifest.skills_path(), manifest.references_path())

    try:
        bank_items = load_approved_bank(active_bank_path(manifest))
    except (BootstrapError, FileNotFoundError):
        bank_items = []

    candidates = CandidateStore(manifest.candidate_store_path).load()
    pending_questions_by_skill = _pending_questions_by_skill(manifest.review_store_path)

    inventory: dict[str, SkillInventory] = {}

    for skill in catalogue.skills:
        approved_items = [item for item in bank_items if item.skill_id == skill.skill_id]
        total_approved = len(approved_items)
        unseen_approved = (
            None
            if answered_item_ids is None
            else sum(
                1
                for item in approved_items
                if item.item_id is not None and item.item_id not in answered_item_ids
            )
        )
        skill_candidates = [
            candidate for candidate in candidates if candidate.skill_id == skill.skill_id
        ]
        pending_references = sum(
            1 for candidate in skill_candidates if candidate.review_status == "pending"
        )
        approved_references = sum(
            1 for candidate in skill_candidates if candidate.review_status == "approved"
        )
        pending_questions = pending_questions_by_skill.get(skill.skill_id, 0)
        template_status = _template_status(skill)
        latest_job = job_repository.latest_for_skill(manifest.course_id, skill.skill_id)

        readiness = derive_readiness(
            generation_strategy=skill.generation_strategy,
            template_status=template_status,
            total_approved_items=total_approved,
            unseen_approved_items=unseen_approved,
            low_supply_threshold=manifest.low_supply_threshold,
            pending_reference_candidates=pending_references,
            approved_reference_candidates=approved_references,
            pending_generated_questions=pending_questions,
            latest_job=latest_job,
        )

        inventory[skill.skill_id] = SkillInventory(
            course_id=manifest.course_id,
            skill_id=skill.skill_id,
            generation_strategy=skill.generation_strategy,
            total_approved_items=total_approved,
            unseen_approved_items=unseen_approved,
            pending_reference_candidates=pending_references,
            approved_reference_candidates=approved_references,
            pending_generated_questions=pending_questions,
            template_status=template_status,
            latest_job=latest_job,
            readiness=readiness,
        )

    return inventory
