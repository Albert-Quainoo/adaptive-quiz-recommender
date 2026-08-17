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
    decision: str  # "enqueued" | "no_deficiency"
    requested_count: int
    reason: str
    proposed_job_key: str  # deterministic_job_key(course_id, skill_id), or "-" when not enqueuing


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
    job_outcomes: list[JobOutcomeRow] = field(default_factory=list)
    pending_approvals: list[PendingApprovalRow] = field(default_factory=list)
    budget: dict = field(default_factory=dict)
    stop_reason: str | None = None
    archived_job_dirs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def deficiency_row(
    decision: ReplenishmentDecision, *, difficulty: str
) -> DeficiencyRow:
    return DeficiencyRow(
        course_id=decision.course_id,
        skill_id=decision.skill_id,
        difficulty=difficulty,
        decision="enqueued" if decision.should_enqueue else "no_deficiency",
        requested_count=decision.requested_count,
        reason=decision.reason,
        proposed_job_key=(
            deterministic_job_key(decision.course_id, decision.skill_id)
            if decision.should_enqueue
            else "-"
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
) -> CycleReport:
    now: datetime = clock()
    return CycleReport(
        generated_at=now.isoformat(),
        dry_run=dry_run,
        deficiencies=deficiencies,
        job_outcomes=job_outcomes,
        pending_approvals=pending_approvals,
        budget=budget,
        stop_reason=stop_reason,
        archived_job_dirs=archived_job_dirs,
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
    parts = [
        "# Content Replenishment Cycle Report\n",
        f"Generated: `{report.generated_at}`  \n"
        f"Mode: **{'dry run (scan only, no external calls)' if report.dry_run else 'live'}**  \n"
        f"Stopped because: {report.stop_reason or 'ran out of claimable work'}\n",
        "## Deficiency scan\n",
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
