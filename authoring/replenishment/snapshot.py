"""Job-scoped artifact snapshots for the content-ops branch.

The PostgreSQL job queue (authoring/replenishment/jobs.py) is the
authoritative source of job status. It cannot hold the candidate, review,
and batch *files* the pipeline reads and writes, though -- those stay
filesystem-based (CandidateStore, GroundedReviewStore, generate_batch's
output_dir), exactly as they are for a human operator today.

For a GitHub-hosted runner (ephemeral, no disk between runs) to resume work
and let an admin actually see and act on pending content, those files need a
durable, reviewable home: a deterministic, job-scoped directory under a
dedicated content-ops branch --

    outputs/replenishment/<course_id>/<job_id>/

Everything written here is copied verbatim from files the existing pipeline
already produces (ReferenceCandidate, GroundedReview, AutomatedReviewReport,
generate_batch's manifest/pending_questions/audit files) -- never a raw HTTP
response, header, or environment variable, so no credential or DSN can leak
through this path by construction.

Writing is idempotent: every file here is overwritten in place, keyed only
by job_id/course_id, never suffixed with a run id or timestamp -- resuming
or re-snapshotting the same job produces the same paths with refreshed
content, not a duplicate.
"""

import json
import shutil
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

from authoring.grounded_review import (
    GroundedReviewStore,
    load_source_questions,
    question_content_hash,
)
from authoring.replenishment.jobs import ReplenishmentJob
from authoring.replenishment.manifest import CourseManifest
from authoring.replenishment.worker import batch_output_dir
from authoring.retrieval.store import CandidateStore
from authoring.review.reports import AutomatedReviewReportStore, review_report_path

TERMINAL_STATUSES = frozenset(
    {
        "completed",
        "cancelled",
        "permanent_failure",
        "rejected_by_automated_review",
        "rejected_deterministically",
        "no_longer_needed",
    }
)

_BATCH_FILES = ("manifest.json", "pending_questions.jsonl", "audit.jsonl")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def job_snapshot_dir(snapshot_root: Path, job: ReplenishmentJob) -> Path:
    return snapshot_root / job.course_id / job.job_id


def snapshot_job_artifacts(
    job: ReplenishmentJob,
    manifest: CourseManifest,
    snapshot_root: Path,
    *,
    budget_deltas: Mapping[str, float] | None = None,
    clock=utc_now,
) -> Path:
    """Write/refresh this job's job-scoped bundle. Safe to call repeatedly
    for the same job (e.g. once per tick) -- always overwrites in place."""
    job_dir = job_snapshot_dir(snapshot_root, job)

    _write_json(
        job_dir / "job.json",
        {
            "job_id": job.job_id,
            "course_id": job.course_id,
            "skill_id": job.skill_id,
            "job_type": job.job_type,
            "status": job.status,
            "requested_count": job.requested_count,
            "attempts": job.attempts,
            "created_at": job.created_at.isoformat(),
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "completed_at": job.completed_at.isoformat() if job.completed_at else None,
            "next_retry_at": job.next_retry_at.isoformat() if job.next_retry_at else None,
            "error_code": job.error_code,
            "error_message": job.error_message,
            # Internal pipeline bookkeeping only (batch_id, review_path,
            # automated_revision_attempts) -- never a credential or secret.
            "metadata": job.metadata,
            "snapshotted_at": clock().isoformat(),
        },
    )

    candidates = [
        candidate
        for candidate in CandidateStore(manifest.candidate_store_path).load()
        if candidate.skill_id == job.skill_id
    ]
    _write_json(
        job_dir / "candidates.json",
        [candidate.model_dump(mode="json") for candidate in candidates],
    )

    review_path_value = job.metadata.get("review_path")
    if review_path_value and Path(review_path_value).is_file():
        review_path = Path(review_path_value)
        shutil.copy2(review_path, job_dir / "review.json")
        review = GroundedReviewStore(review_path).load()

        report_store = AutomatedReviewReportStore(
            review_report_path(manifest.review_store_path, review.batch_id, job.skill_id)
        )
        if report_store.path.is_file():
            shutil.copy2(report_store.path, job_dir / "review_reports.json")

        output_dir = batch_output_dir(manifest, review.batch_id, job.skill_id)
        if output_dir.is_dir():
            for filename in _BATCH_FILES:
                source = output_dir / filename
                if source.is_file():
                    shutil.copy2(source, job_dir / filename)
            content_hashes = {
                question.question_id: question_content_hash(question.question)
                for question in load_source_questions(output_dir)
            }
            if content_hashes:
                _write_json(job_dir / "content_hashes.json", content_hashes)

    if budget_deltas:
        existing = _read_json(job_dir / "budget.json") or {
            "ticks": 0,
            "generation_calls": 0,
            "review_calls": 0,
            "search_calls": 0,
            "estimated_cost_usd": 0.0,
        }
        merged = dict(existing)
        for key, delta in budget_deltas.items():
            merged[key] = existing.get(key, 0) + delta
        _write_json(job_dir / "budget.json", merged)

    return job_dir


def archive_stale_jobs(
    snapshot_root: Path,
    job_repository,
    *,
    retention_days: int = 14,
    clock=utc_now,
) -> list[Path]:
    """Compact any job-scoped directory whose job reached a terminal state
    more than `retention_days` ago down to a small archived.json summary,
    so rejected/obsolete artifacts do not grow the branch without bound.
    Already-archived directories are left untouched (idempotent)."""
    if not snapshot_root.is_dir():
        return []

    now = clock()
    archived: list[Path] = []

    for course_dir in sorted(path for path in snapshot_root.iterdir() if path.is_dir()):
        for job_dir in sorted(path for path in course_dir.iterdir() if path.is_dir()):
            names = {child.name for child in job_dir.iterdir()}
            if names == {"archived.json"}:
                continue  # already archived; never re-timestamp

            job = job_repository.get(job_dir.name)
            if job is None or job.status not in TERMINAL_STATUSES:
                continue

            reference_time = job.completed_at or job.created_at
            age_days = (now - reference_time).total_seconds() / 86400
            if age_days < retention_days:
                continue

            for child in job_dir.iterdir():
                if child.is_file():
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)

            _write_json(
                job_dir / "archived.json",
                {
                    "job_id": job.job_id,
                    "course_id": job.course_id,
                    "skill_id": job.skill_id,
                    "final_status": job.status,
                    "error_code": job.error_code,
                    "archived_at": now.isoformat(),
                },
            )
            archived.append(job_dir)

    return archived
