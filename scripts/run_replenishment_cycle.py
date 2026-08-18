"""Bounded, reportable replenishment run for the scheduled GitHub Actions
workflow (.github/workflows/replenishment.yml).

This is a thin orchestration layer over the existing, unmodified pipeline
(authoring/replenishment/{cli,worker,jobs,policy,inventory}.py): in live
mode it scans the four active course manifests, enqueues deficient skills
exactly the way `python -m authoring.replenishment.cli scan` already does,
then drives the worker loop for at most a few new skill episodes per run
under an explicit call/cost budget (authoring/replenishment/budget.py),
snapshotting every touched job's artifacts to a job-scoped, git-committable
directory (authoring/replenishment/snapshot.py) so a GitHub-hosted
(ephemeral) runner can resume later runs purely from PostgreSQL job state
plus whatever the caller checked out of the content-ops branch.

    python -m scripts.run_replenishment_cycle --dry-run
    python -m scripts.run_replenishment_cycle

--dry-run performs the same read-only inventory/deficiency scan and produces
the identical report, but writes nothing anywhere: no job is enqueued,
updated, claimed, or processed; no branch artifact or PR is created; no
Brave/Modal/reviewer call is made; no bank or lifecycle state changes. Each
would-be-enqueued row instead carries a deterministic_job_key() (report.py)
so the report can be read as "these are the jobs this would enqueue" without
ever calling repository.enqueue().
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import authoring.replenishment.cli as cli
import authoring.replenishment.report as report
from authoring.grounded_batch import BatchGenerationError
from authoring.question_intents import intents_by_skill
from authoring.replenishment.budget import CycleBudgetConfig, CycleBudgetTracker
from authoring.replenishment.inventory import compute_course_inventory
from authoring.replenishment.jobs import SchemaNotReadyError, open_repository
from authoring.replenishment.manifest import load_preparation_eligible_manifests
from authoring.replenishment.policy import ReplenishmentDecision, decide_replenishment
from authoring.replenishment.snapshot import archive_stale_jobs, snapshot_job_artifacts
from authoring.replenishment.worker import (
    _blueprint_generation_difficulty,
    blueprints_covering_skill,
    process_one,
    ready_to_resume,
)
from authoring.review.backpressure import apply_review_backlog_caps
from authoring.review.reports import count_pending_review_items

# The four courses this workflow monitors -- named explicitly rather than
# processing whatever load_preparation_eligible_manifests() happens to
# return, so a future course only becomes part of this workflow through a
# deliberate change here, never by silently picking up a new manifest file.
COURSE_IDS = ("intro-ai", "dsa", "linear-algebra", "database-systems")

DEFAULT_SNAPSHOT_ROOT = Path("outputs/replenishment")
DEFAULT_REPORT_DIR = Path("outputs/replenishment/_reports")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _report_relative_paths(report_dir: Path, snapshot_root: Path) -> list[str]:
    return [
        str((report_dir / "latest.json").relative_to(snapshot_root)),
        str((report_dir / "latest.md").relative_to(snapshot_root)),
    ]


def _run_manifest_paths(
    cycle_report: report.CycleReport, report_dir: Path, snapshot_root: Path
) -> list[str]:
    """Every path under snapshot_root that this specific run wrote or
    modified: the two report files (always), each job this run actually
    processed (job_outcomes -- empty for a dry run), and any job directory
    this run compacted via archive_stale_jobs() (also empty for a dry run).
    Deliberately excludes every other job-scoped directory already present
    under snapshot_root from an earlier run's commit to the content-ops
    branch -- those are historical artifacts this run neither wrote nor
    should re-upload."""
    paths = _report_relative_paths(report_dir, snapshot_root)
    for outcome in cycle_report.job_outcomes:
        paths.append(str(Path(outcome.course_id) / outcome.job_id))
    for archived in cycle_report.archived_job_dirs:
        paths.append(str(Path(archived).relative_to(snapshot_root)))
    return sorted(set(paths))


def _write_run_manifest(
    report_dir: Path, snapshot_root: Path, *, dry_run: bool, paths: list[str]
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_id": os.getenv("GITHUB_RUN_ID", "local"),
                "generated_at": utc_now().isoformat(),
                "dry_run": dry_run,
                "paths": paths,
            },
            indent=2, sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _resolve_difficulty(skill_id: str) -> str:
    """Best-effort difficulty for the deficiency report: only meaningful once
    exactly one reviewed blueprint unambiguously covers this skill with one
    consistent explicit difficulty (see worker.py's
    _blueprint_generation_difficulty). Anything else -- no blueprint yet,
    more than one covering blueprint, inconsistent difficulty -- reports
    "unknown" rather than guessing; generation itself will fail the same way
    a human would expect if this job actually reaches that stage."""
    covering = blueprints_covering_skill(skill_id)
    if len(covering) != 1:
        return "unknown"
    intents = intents_by_skill(covering[0]).get(skill_id, [])
    try:
        return _blueprint_generation_difficulty(skill_id, intents)
    except BatchGenerationError:
        return "unknown"


def _tick_budget_delta(job_type: str, config: CycleBudgetConfig) -> dict:
    from authoring.replenishment.budget import (
        GENERATION_JOB_TYPES,
        REVIEW_JOB_TYPES,
        SEARCH_JOB_TYPES,
    )

    if job_type in GENERATION_JOB_TYPES:
        return {
            "ticks": 1,
            "generation_calls": 1,
            "estimated_cost_usd": config.cost_per_generation_call_usd,
        }
    if job_type in REVIEW_JOB_TYPES:
        return {
            "ticks": 1,
            "review_calls": 1,
            "estimated_cost_usd": config.cost_per_review_call_usd,
        }
    if job_type in SEARCH_JOB_TYPES:
        return {
            "ticks": 1,
            "search_calls": 1,
            "estimated_cost_usd": config.cost_per_search_call_usd,
        }
    return {"ticks": 1, "estimated_cost_usd": 0.0}


def _next_claimable_job(repository, manifests, clock):
    """The job process_one() (worker.py) would claim next, without spending a
    tick -- mirrors its own resume-then-claim ordering exactly (same public
    calls: recover_expired_leases, requeue_retryable, ready_to_resume,
    mark_queued) so the new-candidate cap below can be enforced *before* a
    brand-new job is claimed, not after a call has already been spent on it.
    Idempotent: safe to call every loop iteration."""
    repository.recover_expired_leases(clock=clock)
    repository.requeue_retryable(clock=clock)
    for waiting_job in repository.list_waiting():
        manifest = manifests.get(waiting_job.course_id)
        if manifest is not None and ready_to_resume(waiting_job, manifest):
            next_job_type = (
                "promote_approved_items"
                if waiting_job.status
                in (
                    "waiting_for_question_review",
                    "waiting_for_full_human_review",
                    "rejected_by_automated_review",
                    "rejected_deterministically",
                )
                else waiting_job.job_type
            )
            repository.mark_queued(waiting_job.job_id, job_type=next_job_type)

    queued = sorted(
        (job for job in repository.list_active() if job.status == "queued"),
        key=lambda job: job.created_at,
    )
    return queued[0] if queued else None


def run_cycle(
    *,
    database: str | Path,
    snapshot_root: Path,
    dry_run: bool,
    budget_config: CycleBudgetConfig,
    retention_days: int,
    clock=utc_now,
) -> report.CycleReport:
    # dry_run=True opens a genuinely read-only repository: no initialize_schema()
    # (not even its idempotent CREATE-TABLE-IF-NOT-EXISTS/index-recreation), a
    # read-only-enforced PostgreSQL transaction (open_repository -> read_only=True ->
    # SQLiteReplenishmentJobRepository's postgresql_readonly execution option), and a
    # read-only schema-readiness check instead -- raises SchemaNotReadyError, caught
    # by main() below, rather than silently creating or repairing anything.
    repository = open_repository(database, read_only=dry_run)

    manifests = {
        manifest.course_id: manifest
        for manifest in load_preparation_eligible_manifests()
        if manifest.course_id in COURSE_IDS
    }
    missing = set(COURSE_IDS) - manifests.keys()
    if missing:
        print(
            f"warning: no preparation-eligible manifest for {sorted(missing)}",
            file=sys.stderr,
        )

    review_config = cli._review_config()

    # Two passes: first resolve every decision's difficulty and whether it is
    # blocked, in deterministic course-then-taxonomy scan order; only then
    # cap the first budget_config.max_new_candidates *eligible* (should-
    # enqueue, not blocked) ones as this run's execution plan, so "the first
    # N" is decided from the complete picture, not truncated mid-course.
    candidates: list[tuple[ReplenishmentDecision, str, str | None]] = []
    for course_id in COURSE_IDS:
        manifest = manifests.get(course_id)
        if manifest is None:
            continue
        inventory = compute_course_inventory(manifest, repository)
        decisions = decide_replenishment(inventory, cli._policy_config(manifest))
        decisions = apply_review_backlog_caps(
            decisions,
            pending_per_skill={
                skill_id: row.pending_generated_questions
                for skill_id, row in inventory.items()
            },
            total_pending_backlog=count_pending_review_items(manifest.review_store_path),
            config=review_config,
        )
        for decision in decisions:
            difficulty = (
                _resolve_difficulty(decision.skill_id) if decision.should_enqueue else "-"
            )
            blocked_reason = None
            if decision.should_enqueue and difficulty == "unknown":
                # No reviewed blueprint declares one consistent, explicit
                # difficulty for this skill yet (see worker.py's
                # _blueprint_generation_difficulty docstring: an automated
                # run must never guess introductory vs. intermediate). Not a
                # gate on the job itself -- retrieval/reference review need
                # no difficulty and still proceed in a live run; this only
                # keeps the report from calling it "proposed"/"enqueued" (as
                # if it were a ready-to-generate job) and excludes it from
                # this run's execution plan below.
                blocked_reason = (
                    f"blocked: {decision.skill_id} has no reviewed blueprint "
                    "declaring one consistent, explicit difficulty; automated "
                    "replenishment cannot guess introductory vs. intermediate "
                    "-- author and review a blueprint for this skill before "
                    "it can be planned as a generation-ready job"
                )
            candidates.append((decision, difficulty, blocked_reason))

    eligible_indices = [
        index
        for index, (decision, _difficulty, blocked_reason) in enumerate(candidates)
        if decision.should_enqueue and blocked_reason is None
    ]
    planned_indices = set(eligible_indices[: budget_config.max_new_candidates])
    total_eligible = len(eligible_indices)

    deficiency_rows: list[report.DeficiencyRow] = []
    planned_rows: list[report.PlannedJobRow] = []
    queue_rows_written = 0
    plan_rank = 0
    for index, (decision, difficulty, blocked_reason) in enumerate(candidates):
        is_planned = index in planned_indices
        is_deferred = (
            decision.should_enqueue and blocked_reason is None and not is_planned
        )
        deficiency_rows.append(
            report.deficiency_row(
                decision,
                difficulty=difficulty,
                dry_run=dry_run,
                decision_override=(
                    "blocked" if blocked_reason is not None
                    else "deferred" if is_deferred
                    else None
                ),
                reason_override=(
                    blocked_reason
                    if blocked_reason is not None
                    else (
                        f"{decision.reason}; deferred: exceeds this run's "
                        f"max_new_candidates cap ({budget_config.max_new_candidates})"
                    )
                    if is_deferred
                    else None
                ),
            )
        )
        if is_planned:
            plan_rank += 1
            planned_rows.append(
                report.planned_job_row(
                    rank=plan_rank,
                    total_eligible=total_eligible,
                    decision=decision,
                    difficulty=difficulty,
                )
            )

    if not dry_run:
        # blocked_reason only withholds the report's "proposed"/"enqueued"
        # label and excludes this skill from the execution plan above -- it
        # does not withhold the job itself. Retrieval and reference review
        # (this job's first stages) need no difficulty at all; only
        # generation does, and worker.py's own
        # _blueprint_generation_difficulty already refuses to guess one,
        # permanent-failing that stage with error_code
        # "generation_config_error" if it's still unresolved by then.
        #
        # Enqueued in two passes -- every eligible (non-blocked) deficiency
        # first, in scan order, then every blocked one -- rather than one
        # pass in raw scan order. enqueue()'s created_at is real wall-clock
        # time at call time (see authoring/replenishment/jobs.py), and the
        # worker's FIFO claim (_next_claimable_job() below) always takes the
        # oldest queued job -- so a blocked skill that merely sorts earlier
        # in the taxonomy than this run's planned target must never be
        # given an earlier created_at than that target, or it silently
        # claims this run's new-candidate slot instead.
        for decision, _difficulty, blocked_reason in candidates:
            if decision.should_enqueue and blocked_reason is None:
                repository.enqueue(
                    course_id=decision.course_id,
                    skill_id=decision.skill_id,
                    requested_count=decision.requested_count,
                )
                queue_rows_written += 1
        for decision, _difficulty, blocked_reason in candidates:
            if decision.should_enqueue and blocked_reason is not None:
                repository.enqueue(
                    course_id=decision.course_id,
                    skill_id=decision.skill_id,
                    requested_count=decision.requested_count,
                )
                queue_rows_written += 1

    job_outcomes: list[report.JobOutcomeRow] = []
    archived: list[str] = []
    stop_reason: str | None = None
    tracker = CycleBudgetTracker(config=budget_config)

    if dry_run:
        stop_reason = "dry run: scan only, no worker processing"
    else:
        while True:
            reason = tracker.exhausted()
            if reason:
                stop_reason = reason
                break

            next_job = _next_claimable_job(repository, manifests, clock)
            if next_job is None:
                stop_reason = "no claimable or resumable work"
                break

            # attempts == 0 means this job has never been claimed before --
            # its very first tick, whether this run's own scan enqueued it or
            # an earlier run's did. Every later tick for the same job_id has
            # attempts >= 1, so this naturally fires at most once per job and
            # correctly spans across separate scheduled runs.
            is_new = next_job.attempts == 0
            if tracker.would_start_new_candidate_beyond_cap(next_job.job_id, is_new=is_new):
                stop_reason = (
                    f"new-candidate cap reached ({budget_config.max_new_candidates})"
                )
                break

            processed = process_one(
                repository,
                manifests,
                search_provider_factory=cli._search_provider_factory,
                fetcher_factory=cli._fetcher_factory,
                model_factory=cli._model_factory(),
                reviewer_factory=cli._reviewer_factory(),
                review_config=review_config,
                clock=clock,
                max_attempts=cli._max_attempts(),
            )
            if processed is None:
                stop_reason = "no claimable work"
                break

            tracker.record_tick(job_type=processed.job_type, job_id=processed.job_id, is_new=is_new)

            current = repository.get(processed.job_id)
            job_outcomes.append(report.job_outcome_row(current, is_new_candidate=is_new))

            manifest = manifests.get(current.course_id)
            if manifest is not None:
                snapshot_job_artifacts(
                    current,
                    manifest,
                    snapshot_root,
                    budget_deltas=_tick_budget_delta(processed.job_type, budget_config),
                    clock=clock,
                )

        archived = [
            str(path)
            for path in archive_stale_jobs(
                snapshot_root, repository, retention_days=retention_days, clock=clock
            )
        ]

    pending_rows = [report.pending_approval_row(job) for job in repository.list_waiting()]

    if dry_run:
        budget_summary = {
            "planned_new_candidates": len(planned_rows),
            "max_new_candidates": budget_config.max_new_candidates,
            "max_cost_usd": budget_config.max_cost_usd,
            # No Brave/Modal/reviewer call is ever made in dry-run mode, so
            # this is a fact, not an estimate.
            "estimated_cost_usd": 0.0,
        }
    else:
        budget_summary = tracker.to_dict()

    return report.build_report(
        dry_run=dry_run,
        deficiencies=deficiency_rows,
        execution_plan=planned_rows,
        job_outcomes=job_outcomes,
        pending_approvals=pending_rows,
        budget=budget_summary,
        stop_reason=stop_reason,
        archived_job_dirs=archived,
        clock=clock,
        queue_rows_written=queue_rows_written,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.run_replenishment_cycle",
        description=(
            "Scan the four active course banks, enqueue deficient skills, and "
            "drive the worker loop for a bounded number of new candidates "
            "under an explicit call/cost budget."
        ),
    )
    parser.add_argument("--database", type=str, default=cli.DEFAULT_DATABASE_PATH)
    parser.add_argument(
        "--snapshot-root", type=Path, default=DEFAULT_SNAPSHOT_ROOT,
        help="Job-scoped artifact root (outputs/replenishment/<course>/<job_id>/)",
    )
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan and enqueue only -- no Brave/Modal/reviewer calls, no worker processing.",
    )
    parser.add_argument(
        "--max-new-candidates",
        type=int,
        default=_env_int("QUIZ_REPLENISHMENT_MAX_NEW_CANDIDATES", 3),
    )
    parser.add_argument(
        "--max-generation-calls",
        type=int,
        default=_env_int("QUIZ_REPLENISHMENT_MAX_GENERATION_CALLS", 20),
    )
    parser.add_argument(
        "--max-cost-usd",
        type=float,
        default=_env_float("QUIZ_REPLENISHMENT_MAX_COST_USD", 5.0),
    )
    parser.add_argument(
        "--cost-per-generation-call-usd",
        type=float,
        default=_env_float("QUIZ_REPLENISHMENT_COST_PER_GENERATION_CALL_USD", 0.05),
    )
    parser.add_argument(
        "--cost-per-review-call-usd",
        type=float,
        default=_env_float("QUIZ_REPLENISHMENT_COST_PER_REVIEW_CALL_USD", 0.02),
    )
    parser.add_argument(
        "--cost-per-search-call-usd",
        type=float,
        default=_env_float("QUIZ_REPLENISHMENT_COST_PER_SEARCH_CALL_USD", 0.0),
    )
    parser.add_argument(
        "--max-ticks", type=int, default=_env_int("QUIZ_REPLENISHMENT_MAX_TICKS", 500),
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=_env_int("QUIZ_REPLENISHMENT_RETENTION_DAYS", 14),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    budget_config = CycleBudgetConfig(
        max_new_candidates=args.max_new_candidates,
        max_generation_calls=args.max_generation_calls,
        max_cost_usd=args.max_cost_usd,
        cost_per_generation_call_usd=args.cost_per_generation_call_usd,
        cost_per_review_call_usd=args.cost_per_review_call_usd,
        cost_per_search_call_usd=args.cost_per_search_call_usd,
        max_ticks=args.max_ticks,
    )

    try:
        cycle_report = run_cycle(
            database=args.database,
            snapshot_root=args.snapshot_root,
            dry_run=args.dry_run,
            budget_config=budget_config,
            retention_days=args.retention_days,
        )
    except SchemaNotReadyError as exc:
        # A read-only (dry-run) caller found the schema it depends on isn't already
        # present -- stop and report this, rather than silently repairing it (only
        # initialize_schema(), never called on this path, does that). Still writes
        # latest.json/latest.md so the workflow's unconditional "Publish job
        # summary"/"Upload artifacts" steps have something coherent to show.
        args.report_dir.mkdir(parents=True, exist_ok=True)
        stopped_at = utc_now().isoformat()
        (args.report_dir / "latest.json").write_text(
            json.dumps(
                {"status": "schema_not_ready", "reason": str(exc), "generated_at": stopped_at},
                indent=2, sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        markdown = f"# Replenishment cycle: schema not ready\n\n{exc}\n"
        (args.report_dir / "latest.md").write_text(markdown, encoding="utf-8")
        _write_run_manifest(
            args.report_dir, args.snapshot_root, dry_run=args.dry_run,
            paths=_report_relative_paths(args.report_dir, args.snapshot_root),
        )
        print(markdown, file=sys.stderr)
        return 2

    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "latest.json").write_text(
        json.dumps(cycle_report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    markdown = report.render_markdown(cycle_report)
    (args.report_dir / "latest.md").write_text(markdown, encoding="utf-8")
    _write_run_manifest(
        args.report_dir, args.snapshot_root, dry_run=args.dry_run,
        paths=_run_manifest_paths(cycle_report, args.report_dir, args.snapshot_root),
    )
    print(markdown)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
