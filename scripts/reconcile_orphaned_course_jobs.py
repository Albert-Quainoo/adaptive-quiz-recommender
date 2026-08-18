"""Audited cleanup for replenishment jobs whose course_id names no current
course manifest.

Discovered while starting the controlled bank-population phase:
authoring/replenishment/worker.py's process_one() does manifests[job.course_id]
with no defensive handling for an unmapped course_id, so claim_next() -- which
always claims the oldest active job across every course -- crashes with an
unhandled KeyError the moment it reaches one. The real job database has 30+
active-status rows with course_id="ai" from 2026-08-13, predating the course's
current manifest ("intro-ai"); no manifest named "ai" exists, so these can
never be legitimately processed and were silently blocking every future
worker/process_one invocation, not just this batch's.

This only ever cancels rows for a course_id that currently maps to NO manifest
at all (refuses to run otherwise -- never touches a real, still-manifested
course's jobs) and only rows in an ACTIVE_STATUSES status (repository.cancel()
is a no-op-safe status flip, never a delete; every row's history stays in the
table). Dry run (the default) opens the repository read-only and only lists
what would change. --confirm is the one explicit step required before any row
is written, exactly like scripts/reconcile_demand_already_satisfied.py.

    python -m scripts.reconcile_orphaned_course_jobs ai            # dry run (default)
    python -m scripts.reconcile_orphaned_course_jobs ai --confirm  # apply
"""

import argparse
import sys

import authoring.replenishment.cli as cli
from authoring.replenishment.jobs import open_repository
from authoring.replenishment.manifest import load_all_manifests


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.reconcile_orphaned_course_jobs",
        description="Cancel active-status replenishment jobs whose course_id names no current manifest.",
    )
    parser.add_argument("course_id", help="the orphaned course_id to clean up (e.g. 'ai')")
    parser.add_argument("--database", type=str, default=cli.DEFAULT_DATABASE_PATH)
    parser.add_argument(
        "--confirm", action="store_true", help="apply the cancellations; omit for a read-only dry run"
    )
    args = parser.parse_args(argv)

    known_course_ids = {manifest.course_id for manifest in load_all_manifests()}
    if args.course_id in known_course_ids:
        print(
            f"error: {args.course_id!r} has a real manifest -- refusing to run "
            "(this tool only cleans up jobs for a course_id with NO manifest at all)",
            file=sys.stderr,
        )
        return 2

    repository = open_repository(args.database, read_only=not args.confirm)
    rows = repository.list_active(course_id=args.course_id)

    if not rows:
        print(f"No active-status jobs found for course_id={args.course_id!r}.")
        return 0

    print(f"course_id={args.course_id!r} has no manifest. {len(rows)} active job(s) found:\n")
    for job in rows:
        print(f"  {job.job_id}  {job.skill_id:<14} {job.job_type:<24} status={job.status}  created_at={job.created_at}")

    if not args.confirm:
        print("\nDRY RUN -- no changes made. Re-run with --confirm to apply.")
        return 0

    for job in rows:
        repository.cancel(job.job_id)
    print(f"\ncancelled {len(rows)} job(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
