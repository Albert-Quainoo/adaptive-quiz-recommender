"""Audited reconciliation for legacy demand_already_satisfied jobs.

Before authoring/replenishment/worker.py's two demand-check call sites were
changed to call mark_no_longer_needed() (see that method's docstring in
authoring/replenishment/jobs.py), a skill whose only covering blueprint had
no unsatisfied slot left was recorded as mark_permanent_failure(error_code=
"demand_already_satisfied") -- a real, correct refusal, but stored as the
wrong (blocking) status. Because authoring/replenishment/inventory.py's
derive_readiness() always reads a skill's latest_for_skill() row and maps
every "permanent_failure" onto "replenishment_failed", each such legacy row
permanently excludes its skill from every future replenishment scan -- no
later run can ever supersede it, even after the blueprint gains capacity.

This command reclassifies exactly one such row -- named explicitly by
job_id, never a bulk sweep by error_code -- into the correct, non-blocking
"no_longer_needed" status, freeing the skill for the next scan to
re-evaluate its demand fresh. It never re-runs retrieval, generation, or
review, and never enqueues anything itself.

    python -m scripts.reconcile_demand_already_satisfied <job_id>            # dry run (default)
    python -m scripts.reconcile_demand_already_satisfied <job_id> --confirm  # apply

Dry run (the default) opens the repository read-only -- the same
check_schema_ready()/postgresql_readonly guarantee
scripts/run_replenishment_cycle.py's --dry-run uses -- so omitting --confirm
can never write anything, not even implicitly. --confirm is the one explicit
step this command requires before it ever mutates a row; there is no
separate "are you sure?" prompt, so scripting this command still requires
naming --confirm on the command line.
"""

import argparse
import sys

from authoring.replenishment.demand import load_approved_item_ids
from authoring.replenishment.jobs import JobConflictError, open_repository
from authoring.replenishment.manifest import load_preparation_eligible_manifests
from authoring.replenishment.worker import (
    AmbiguousBlueprintError,
    _demand_fingerprint,
    resolve_job_blueprint,
)

import authoring.replenishment.cli as cli


def _manifest_for_course(course_id: str):
    for manifest in load_preparation_eligible_manifests():
        if manifest.course_id == course_id:
            return manifest
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.reconcile_demand_already_satisfied",
        description=(
            "Reclassify one legacy demand_already_satisfied permanent_failure job "
            "into the correct non-blocking no_longer_needed status."
        ),
    )
    parser.add_argument("job_id", help="the exact job_id to reconcile")
    parser.add_argument("--database", type=str, default=cli.DEFAULT_DATABASE_PATH)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="apply the reconciliation; omit for a read-only dry run",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repository = open_repository(args.database, read_only=not args.confirm)

    job = repository.get(args.job_id)
    if job is None:
        print(f"error: no job {args.job_id!r} found", file=sys.stderr)
        return 2
    if job.status != "permanent_failure" or job.error_code != "demand_already_satisfied":
        print(
            f"error: {args.job_id} is not a legacy demand_already_satisfied "
            f"permanent_failure (status={job.status!r}, error_code={job.error_code!r}) "
            "-- refusing to reconcile",
            file=sys.stderr,
        )
        return 2

    manifest = _manifest_for_course(job.course_id)
    if manifest is None:
        print(
            f"error: no preparation-eligible manifest for course {job.course_id!r}",
            file=sys.stderr,
        )
        return 2

    try:
        blueprint, _batch_id = resolve_job_blueprint(job)
    except AmbiguousBlueprintError as exc:
        print(
            f"error: cannot resolve one blueprint for {job.skill_id}: {exc}",
            file=sys.stderr,
        )
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if blueprint is None:
        print(
            f"error: no blueprint currently covers {job.skill_id}; cannot recompute "
            "a demand fingerprint for reconciliation",
            file=sys.stderr,
        )
        return 2

    approved_item_ids = load_approved_item_ids(manifest)
    fingerprint = _demand_fingerprint(blueprint, job.skill_id, approved_item_ids, manifest)

    print(f"job:             {job.job_id}")
    print(f"course/skill:    {job.course_id}/{job.skill_id}")
    print(f"current:         status={job.status} error_code={job.error_code}")
    print(f"historical error: {job.error_message}")
    print("target:          status=no_longer_needed error_code=demand_already_satisfied")
    print(f"demand_fingerprint (recomputed via application code): {fingerprint}")

    if not args.confirm:
        print("\nDRY RUN -- no changes made. Re-run with --confirm to apply.")
        return 0

    try:
        updated = repository.reconcile_legacy_demand_already_satisfied(
            job.job_id, demand_fingerprint=fingerprint
        )
    except JobConflictError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"\nreconciled: {updated.job_id} -> status={updated.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
