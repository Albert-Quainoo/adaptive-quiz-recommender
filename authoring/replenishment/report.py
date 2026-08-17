"""Structured report for one scheduled-replenishment run: the deficiency
scan, what the bounded worker loop did about it, the budget spent, and what
still needs an admin's attention. Pure data assembly and Markdown rendering
only -- no filesystem or network access, so it is trivially unit-testable
and reusable for both the GitHub step summary and the JSON artifact.
"""

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime

from authoring.replenishment.jobs import ReplenishmentJob
from authoring.replenishment.policy import ReplenishmentDecision

# Fixed, arbitrary namespace for deterministic_job_key() below -- any stable
# UUID works here since it only needs to be consistent across calls, never
# to collide with a real job_id (those are uuid4(), assigned only at actual
# enqueue time by SQLiteReplenishmentJobRepository.enqueue()).
_PROPOSED_JOB_NAMESPACE = uuid.UUID("6f1f5b3a-3b0a-4c9a-9c3e-2b6a7a9d5e10")


def deterministic_job_key(course_id: str, skill_id: str) -> str:
    """Stable identity for a job that would be enqueued for (course_id,
    skill_id), independent of any real job_id. Lets a dry-run report --
    which enqueues nothing -- still name the specific job a later live run
    would create for the same course/skill, so an admin can correlate them."""
    return str(uuid.uuid5(_PROPOSED_JOB_NAMESPACE, f"{course_id}:{skill_id}"))

# Maps a job's terminal-for-this-tick status onto the five outcome buckets
# requirement 5 asks the summary to distinguish, plus two operationally
# useful extras (permanent failure and cancellation) that would otherwise be
# silently folded into "rejected" and lose their distinct meaning.
_OUTCOME_BY_STATUS = {
    "queued": "in_progress",
    "running": "in_progress",
    "waiting_for_reference_review": "in_progress",
    "waiting_for_model": "in_progress",
    "waiting_for_question_review": "awaiting_approval",
    "waiting_for_full_human_review": "awaiting_approval",
    "rejected_by_automated_review": "rejected",
    "rejected_deterministically": "rejected",
    "retryable_failure": "retryable_failure",
    "permanent_failure": "permanent_failure",
    "completed": "generated",
    "cancelled": "cancelled",
}


def classify_outcome(status: str) -> str:
    return _OUTCOME_BY_STATUS.get(status, "in_progress")


@dataclass(frozen=True)
class DeficiencyRow:
    course_id: str
    skill_id: str
    difficulty: str
    # "enqueued" (live) | "proposed" (dry run) | "deferred" | "blocked"
    # | "manual_action_required" | "no_deficiency"
    decision: str
    requested_count: int
    reason: str
    proposed_job_key: str  # deterministic_job_key(course_id, skill_id), or "-" when not enqueuing


@dataclass(frozen=True)
class PlannedJobRow:
    """One of the (at most budget.max_new_candidates) deficiencies this run
    actually plans to work, in the same deterministic order enqueue()/the
    worker's FIFO claim would process them -- never a priority the policy
    itself hasn't already decided."""

    rank: int
    course_id: str
    skill_id: str
    difficulty: str
    requested_count: int
    proposed_job_key: str
    reason: str


@dataclass(frozen=True)
class JobOutcomeRow:
    job_id: str
    course_id: str
    skill_id: str
    job_type: str
    status: str
    outcome_category: str
    error_code: str | None
    is_new_candidate: bool


@dataclass(frozen=True)
class PendingApprovalRow:
    job_id: str
    course_id: str
    skill_id: str
    status: str
    waiting_since: str


@dataclass(frozen=True)
class CycleReport:
    generated_at: str
    dry_run: bool
    deficiencies: list[DeficiencyRow] = field(default_factory=list)
    execution_plan: list[PlannedJobRow] = field(default_factory=list)
    job_outcomes: list[JobOutcomeRow] = field(default_factory=list)
    pending_approvals: list[PendingApprovalRow] = field(default_factory=list)
    budget: dict = field(default_factory=dict)
    stop_reason: str | None = None
    archived_job_dirs: list[str] = field(default_factory=list)
    # Actual repository.enqueue() calls made this run -- always 0 for a dry
    # run (see deficiency_row's dry_run handling above); reported explicitly
    # so "did this run really write nothing" never has to be taken on faith.
    queue_rows_written: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


def deficiency_row(
    decision: ReplenishmentDecision,
    *,
    difficulty: str,
    dry_run: bool,
    decision_override: str | None = None,
    reason_override: str | None = None,
) -> DeficiencyRow:
    """decision_override/reason_override let the caller (run_cycle) downgrade
    an otherwise-enqueueable decision to "blocked" (unresolved difficulty) or
    "deferred" (beyond this run's max_new_candidates cap) without losing the
    original policy reason. Never used to *upgrade* a no_deficiency row."""
    if decision_override is not None:
        decision_label = decision_override
        reason = reason_override or decision.reason
        proposed_job_key = (
            "-"
            if decision_override == "blocked"
            else deterministic_job_key(decision.course_id, decision.skill_id)
        )
    elif decision.should_enqueue:
        # A dry run enqueues nothing (run_replenishment_cycle.py never calls
        # repository.enqueue() when dry_run is True) -- "enqueued" would be a
        # lie about job-queue state, so dry runs get "proposed" instead.
        decision_label = "enqueued" if not dry_run else "proposed"
        reason = decision.reason
        proposed_job_key = deterministic_job_key(decision.course_id, decision.skill_id)
    elif decision.manual_action_required:
        # A deficient hand_authored skill: never auto-enqueued, but not a
        # satisfied no_deficiency either -- stays visibly reported until a
        # person's own writing brings its supply to threshold.
        decision_label = "manual_action_required"
        reason = decision.reason
        proposed_job_key = "-"
    else:
        decision_label = "no_deficiency"
        reason = decision.reason
        proposed_job_key = "-"

    return DeficiencyRow(
        course_id=decision.course_id,
        skill_id=decision.skill_id,
        difficulty=difficulty,
        decision=decision_label,
        requested_count=decision.requested_count,
        reason=reason,
        proposed_job_key=proposed_job_key,
    )


def planned_job_row(
    *, rank: int, total_eligible: int, decision: ReplenishmentDecision, difficulty: str
) -> PlannedJobRow:
    return PlannedJobRow(
        rank=rank,
        course_id=decision.course_id,
        skill_id=decision.skill_id,
        difficulty=difficulty,
        requested_count=decision.requested_count,
        proposed_job_key=deterministic_job_key(decision.course_id, decision.skill_id),
        reason=(
            f"selected {rank} of {total_eligible} eligible deficiencies, in "
            "deterministic course-then-taxonomy scan order (COURSE_IDS order, "
            "then skills.csv row order within each course) -- the same "
            "creation order enqueue() assigns and the worker's FIFO claim "
            "(oldest created_at first) would later process"
        ),
    )


def job_outcome_row(job: ReplenishmentJob, *, is_new_candidate: bool) -> JobOutcomeRow:
    return JobOutcomeRow(
        job_id=job.job_id,
        course_id=job.course_id,
        skill_id=job.skill_id,
        job_type=job.job_type,
        status=job.status,
        outcome_category=classify_outcome(job.status),
        error_code=job.error_code,
        is_new_candidate=is_new_candidate,
    )


def pending_approval_row(job: ReplenishmentJob) -> PendingApprovalRow:
    return PendingApprovalRow(
        job_id=job.job_id,
        course_id=job.course_id,
        skill_id=job.skill_id,
        status=job.status,
        waiting_since=job.created_at.isoformat(),
    )


def build_report(
    *,
    dry_run: bool,
    deficiencies: list[DeficiencyRow],
    job_outcomes: list[JobOutcomeRow],
    pending_approvals: list[PendingApprovalRow],
    budget: dict,
    stop_reason: str | None,
    archived_job_dirs: list[str],
    clock,
    execution_plan: list[PlannedJobRow] | None = None,
    queue_rows_written: int = 0,
) -> CycleReport:
    now: datetime = clock()
    return CycleReport(
        generated_at=now.isoformat(),
        dry_run=dry_run,
        deficiencies=deficiencies,
        execution_plan=execution_plan or [],
        job_outcomes=job_outcomes,
        pending_approvals=pending_approvals,
        budget=budget,
        stop_reason=stop_reason,
        archived_job_dirs=archived_job_dirs,
        queue_rows_written=queue_rows_written,
    )


def _table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_none_\n"
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join(lines) + "\n"


def render_markdown(report: CycleReport) -> str:
    deferred_count = sum(1 for row in report.deficiencies if row.decision == "deferred")
    blocked_count = sum(1 for row in report.deficiencies if row.decision == "blocked")
    manual_count = sum(
        1 for row in report.deficiencies if row.decision == "manual_action_required"
    )
    parts = [
        "# Content Replenishment Cycle Report\n",
        f"Generated: `{report.generated_at}`  \n"
        f"Mode: **{'dry run (scan only, no external calls)' if report.dry_run else 'live'}**  \n"
        f"Stopped because: {report.stop_reason or 'ran out of claimable work'}  \n"
        f"Job-queue rows written this run: **{report.queue_rows_written}**\n",
        "## Deficiency scan (complete inventory)\n",
        _table(
            ["Course", "Skill", "Difficulty", "Decision", "Requested", "Reason", "Proposed job"],
            [
                [
                    row.course_id,
                    row.skill_id,
                    row.difficulty,
                    row.decision,
                    str(row.requested_count),
                    row.reason,
                    row.proposed_job_key[:8] if row.proposed_job_key != "-" else "-",
                ]
                for row in report.deficiencies
            ],
        ),
        "## Planned execution (this run)\n",
        _table(
            ["Rank", "Course", "Skill", "Difficulty", "Requested", "Proposed job", "Reason"],
            [
                [
                    str(row.rank),
                    row.course_id,
                    row.skill_id,
                    row.difficulty,
                    str(row.requested_count),
                    row.proposed_job_key[:8],
                    row.reason,
                ]
                for row in report.execution_plan
            ],
        ),
        f"Planned this run: **{len(report.execution_plan)}**  \n"
        f"Deferred (eligible, over cap, left for a later run): **{deferred_count}**  \n"
        f"Blocked (unresolved difficulty, needs a reviewed blueprint): **{blocked_count}**  \n"
        f"Manual action required (hand_authored, needs a person to write items): **{manual_count}**\n",
        "## Job processing outcomes this run\n",
        _table(
            ["Job", "Course", "Skill", "Stage", "Status", "Outcome", "New?", "Error"],
            [
                [
                    row.job_id[:8],
                    row.course_id,
                    row.skill_id,
                    row.job_type,
                    row.status,
                    row.outcome_category,
                    "yes" if row.is_new_candidate else "no",
                    row.error_code or "-",
                ]
                for row in report.job_outcomes
            ],
        ),
        "## Pending admin approvals\n",
        _table(
            ["Job", "Course", "Skill", "Status", "Waiting since"],
            [
                [row.job_id[:8], row.course_id, row.skill_id, row.status, row.waiting_since]
                for row in report.pending_approvals
            ],
        ),
        "## Budget\n",
        "\n".join(f"- **{key}**: {value}" for key, value in report.budget.items()) + "\n",
    ]
    if report.archived_job_dirs:
        parts.append(
            "## Archived (retention)\n"
            + "\n".join(f"- `{path}`" for path in report.archived_job_dirs)
            + "\n"
        )
    return "\n".join(parts)
