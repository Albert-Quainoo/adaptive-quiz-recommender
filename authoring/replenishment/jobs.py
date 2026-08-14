"""SQLite-backed replenishment job queue.

One row tracks one skill's replenishment episode end to end. `job_type`
starts as "replenish_skill" and is advanced in place by the worker as the
episode moves through retrieval, generation, and promotion -- there is only
ever one active row per (course_id, skill_id), enforced by a partial unique
index rather than application-level locking, so repeated Streamlit reruns or
concurrent `scan` invocations cannot create duplicates.

This mirrors the connect/schema/transaction conventions of
bkt/sqlite_repository.py, but is a standalone repository: the worker runs as
a separate CLI process from Streamlit, and the two must be able to share the
database file without either blocking the other, hence WAL mode here.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

JobType = Literal[
    "retrieve_references",
    "generate_questions",
    "validate_questions",
    "automated_review",
    "automated_revision",
    "promote_approved_items",
    "replenish_skill",
]

JobStatus = Literal[
    "queued",
    "running",
    "waiting_for_reference_review",
    "waiting_for_model",
    "waiting_for_question_review",
    "waiting_for_full_human_review",
    "rejected_by_automated_review",
    "rejected_deterministically",
    "completed",
    "retryable_failure",
    "permanent_failure",
    "cancelled",
]

ACTIVE_STATUSES: tuple[JobStatus, ...] = (
    "queued",
    "running",
    "waiting_for_reference_review",
    "waiting_for_model",
    "waiting_for_question_review",
    "waiting_for_full_human_review",
    "rejected_by_automated_review",
    "rejected_deterministically",
)

WAITING_STATUSES: tuple[JobStatus, ...] = (
    "waiting_for_reference_review",
    "waiting_for_model",
    "waiting_for_question_review",
    "waiting_for_full_human_review",
    "rejected_by_automated_review",
    "rejected_deterministically",
)


class ReplenishmentJob(BaseModel):
    job_id: str = Field(min_length=1)
    course_id: str = Field(min_length=1)
    skill_id: str = Field(min_length=1)
    job_type: JobType
    status: JobStatus
    requested_count: int = Field(ge=0)
    attempts: int = Field(ge=0)
    created_at: datetime
    started_at: datetime | None = None
    completed_at: datetime | None = None
    next_retry_at: datetime | None = None
    lease_expires_at: datetime | None = None
    error_code: str | None = None
    error_message: str | None = None
    metadata: dict = Field(default_factory=dict)


class JobConflictError(RuntimeError):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _aware_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(timezone.utc)


def _optional_datetime(value: str | None) -> datetime | None:
    return _aware_datetime(value) if value is not None else None


# The index's WHERE clause is generated from ACTIVE_STATUSES, not hand-copied, so the
# two can never drift apart -- a status left out of one but not the other would let a
# repeated scan/enqueue create a second live job for the same (course_id, skill_id)
# while a row sits in the missing status.
#
# Deliberately excludes job_type: job_type is the episode's current pipeline stage,
# not part of its identity, and the worker advances it in place as the episode moves
# through retrieval, generation, review, and promotion (almost immediately after
# creation). Every real caller of enqueue() (cli.py's scan(), app/bootstrap.py's
# low-supply reporter) always passes the default job_type="replenish_skill" -- keying
# the index on job_type too let a re-enqueue for a skill already mid-pipeline (job_type
# advanced away from "replenish_skill") silently insert a second active row instead of
# being caught as a duplicate, contradicting this module's one-row-per-skill guarantee.
_ACTIVE_STATUSES_SQL = ", ".join(f"'{status}'" for status in ACTIVE_STATUSES)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS replenishment_jobs (
    job_id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    job_type TEXT NOT NULL,
    status TEXT NOT NULL,
    requested_count INTEGER NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    next_retry_at TEXT,
    lease_expires_at TEXT,
    error_code TEXT,
    error_message TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{{}}'
);

CREATE UNIQUE INDEX IF NOT EXISTS ux_replenishment_jobs_active
ON replenishment_jobs(course_id, skill_id)
WHERE status IN ({_ACTIVE_STATUSES_SQL});
"""


class SQLiteReplenishmentJobRepository:
    def __init__(self, database: str | Path) -> None:
        self.database = str(database)
        self._connection = sqlite3.connect(
            self.database, check_same_thread=False, isolation_level=None
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA journal_mode = WAL")
        self._connection.execute("PRAGMA foreign_keys = ON")

    def initialize_schema(self) -> None:
        # CREATE UNIQUE INDEX IF NOT EXISTS is a no-op if an index by this name already
        # exists under an older WHERE clause (e.g. a database file created before the
        # automated-review statuses existed), so the index is dropped and recreated
        # every call -- both operations are cheap and idempotent.
        self._connection.execute("DROP INDEX IF EXISTS ux_replenishment_jobs_active")
        self._connection.executescript(SCHEMA)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SQLiteReplenishmentJobRepository":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    @contextmanager
    def _transaction(self):
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except Exception:
            self._connection.execute("ROLLBACK")
            raise
        else:
            self._connection.execute("COMMIT")

    def enqueue(
        self,
        *,
        course_id: str,
        skill_id: str,
        job_type: JobType = "replenish_skill",
        requested_count: int,
        metadata: dict | None = None,
        clock=utc_now,
    ) -> ReplenishmentJob:
        """Insert a new job, or return the existing active one unchanged."""
        now = clock()
        job = ReplenishmentJob(
            job_id=str(uuid4()),
            course_id=course_id,
            skill_id=skill_id,
            job_type=job_type,
            status="queued",
            requested_count=requested_count,
            attempts=0,
            created_at=now,
            metadata=metadata or {},
        )
        try:
            with self._transaction():
                self._connection.execute(
                    """
                    INSERT INTO replenishment_jobs (
                        job_id, course_id, skill_id, job_type, status,
                        requested_count, attempts, created_at, metadata_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        job.job_id,
                        job.course_id,
                        job.skill_id,
                        job.job_type,
                        job.status,
                        job.requested_count,
                        job.attempts,
                        _utc_iso(job.created_at),
                        json.dumps(job.metadata, sort_keys=True),
                    ),
                )
        except sqlite3.IntegrityError:
            existing = self._active_job(course_id, skill_id)
            if existing is None:
                raise JobConflictError(
                    f"active-job uniqueness conflict for {course_id}/{skill_id}, "
                    "but no active job could be found"
                )
            return existing
        return job

    def _active_job(self, course_id: str, skill_id: str) -> ReplenishmentJob | None:
        placeholders = ", ".join("?" for _ in ACTIVE_STATUSES)
        row = self._connection.execute(
            f"""
            SELECT * FROM replenishment_jobs
            WHERE course_id = ? AND skill_id = ?
              AND status IN ({placeholders})
            ORDER BY created_at DESC LIMIT 1
            """,
            (course_id, skill_id, *ACTIVE_STATUSES),
        ).fetchone()
        return self._job_from_row(row) if row is not None else None

    def get(self, job_id: str) -> ReplenishmentJob | None:
        row = self._connection.execute(
            "SELECT * FROM replenishment_jobs WHERE job_id = ?", (job_id,)
        ).fetchone()
        return self._job_from_row(row) if row is not None else None

    def claim_next(
        self, *, lease_seconds: int = 300, clock=utc_now
    ) -> ReplenishmentJob | None:
        """Atomically claim the oldest queued job, if any."""
        now = clock()
        with self._transaction():
            row = self._connection.execute(
                "SELECT * FROM replenishment_jobs WHERE status = 'queued' "
                "ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            job = self._job_from_row(row)
            started_at = job.started_at or now
            lease_expires_at = now + timedelta(seconds=lease_seconds)
            self._connection.execute(
                """
                UPDATE replenishment_jobs
                SET status = 'running', attempts = attempts + 1,
                    started_at = ?, lease_expires_at = ?
                WHERE job_id = ?
                """,
                (_utc_iso(started_at), _utc_iso(lease_expires_at), job.job_id),
            )
        return self.get(job.job_id)

    def recover_expired_leases(self, *, clock=utc_now) -> int:
        """Reset abandoned 'running' jobs (past their lease) back to queued."""
        now = clock()
        with self._transaction():
            cursor = self._connection.execute(
                """
                UPDATE replenishment_jobs
                SET status = 'queued', lease_expires_at = NULL
                WHERE status = 'running' AND lease_expires_at < ?
                """,
                (_utc_iso(now),),
            )
            return cursor.rowcount

    def requeue_retryable(self, *, clock=utc_now) -> int:
        """Move retryable_failure jobs whose backoff has elapsed back to queued."""
        now = clock()
        with self._transaction():
            cursor = self._connection.execute(
                """
                UPDATE replenishment_jobs
                SET status = 'queued', next_retry_at = NULL
                WHERE status = 'retryable_failure' AND next_retry_at <= ?
                """,
                (_utc_iso(now),),
            )
            return cursor.rowcount

    def mark_queued(
        self,
        job_id: str,
        *,
        job_type: JobType | None = None,
        metadata: dict | None = None,
    ) -> ReplenishmentJob:
        job = self._require(job_id)
        merged_metadata = {**job.metadata, **(metadata or {})}
        with self._transaction():
            self._connection.execute(
                "UPDATE replenishment_jobs SET status = 'queued', "
                "job_type = ?, lease_expires_at = NULL, metadata_json = ? WHERE job_id = ?",
                (job_type or job.job_type, json.dumps(merged_metadata, sort_keys=True), job_id),
            )
        return self._require(job_id)

    def mark_waiting(
        self,
        job_id: str,
        status: Literal[
            "waiting_for_reference_review",
            "waiting_for_model",
            "waiting_for_question_review",
            "waiting_for_full_human_review",
            "rejected_by_automated_review",
            "rejected_deterministically",
        ],
        *,
        job_type: JobType | None = None,
        metadata: dict | None = None,
    ) -> ReplenishmentJob:
        job = self._require(job_id)
        merged_metadata = {**job.metadata, **(metadata or {})}
        with self._transaction():
            self._connection.execute(
                """
                UPDATE replenishment_jobs
                SET status = ?, job_type = ?, lease_expires_at = NULL,
                    metadata_json = ?
                WHERE job_id = ?
                """,
                (
                    status,
                    job_type or job.job_type,
                    json.dumps(merged_metadata, sort_keys=True),
                    job_id,
                ),
            )
        return self._require(job_id)

    def mark_completed(self, job_id: str, *, clock=utc_now) -> ReplenishmentJob:
        with self._transaction():
            self._connection.execute(
                """
                UPDATE replenishment_jobs
                SET status = 'completed', completed_at = ?, lease_expires_at = NULL
                WHERE job_id = ?
                """,
                (_utc_iso(clock()), job_id),
            )
        return self._require(job_id)

    def mark_retryable_failure(
        self,
        job_id: str,
        *,
        error_code: str,
        error_message: str,
        backoff_seconds: int = 60,
        max_attempts: int = 5,
        clock=utc_now,
    ) -> ReplenishmentJob:
        job = self._require(job_id)
        now = clock()
        if job.attempts >= max_attempts:
            return self.mark_permanent_failure(
                job_id, error_code=error_code, error_message=error_message
            )
        with self._transaction():
            self._connection.execute(
                """
                UPDATE replenishment_jobs
                SET status = 'retryable_failure', lease_expires_at = NULL,
                    next_retry_at = ?, error_code = ?, error_message = ?
                WHERE job_id = ?
                """,
                (
                    _utc_iso(now + timedelta(seconds=backoff_seconds * job.attempts)),
                    error_code,
                    error_message,
                    job_id,
                ),
            )
        return self._require(job_id)

    def mark_permanent_failure(
        self, job_id: str, *, error_code: str, error_message: str
    ) -> ReplenishmentJob:
        with self._transaction():
            self._connection.execute(
                """
                UPDATE replenishment_jobs
                SET status = 'permanent_failure', lease_expires_at = NULL,
                    error_code = ?, error_message = ?
                WHERE job_id = ?
                """,
                (error_code, error_message, job_id),
            )
        return self._require(job_id)

    def cancel(self, job_id: str) -> ReplenishmentJob:
        with self._transaction():
            self._connection.execute(
                "UPDATE replenishment_jobs SET status = 'cancelled', "
                "lease_expires_at = NULL WHERE job_id = ?",
                (job_id,),
            )
        return self._require(job_id)

    def list_active(
        self, *, course_id: str | None = None, skill_id: str | None = None
    ) -> list[ReplenishmentJob]:
        clauses = [f"status IN ({', '.join('?' for _ in ACTIVE_STATUSES)})"]
        parameters: list[str] = list(ACTIVE_STATUSES)
        if course_id is not None:
            clauses.append("course_id = ?")
            parameters.append(course_id)
        if skill_id is not None:
            clauses.append("skill_id = ?")
            parameters.append(skill_id)
        rows = self._connection.execute(
            f"SELECT * FROM replenishment_jobs WHERE {' AND '.join(clauses)} "
            "ORDER BY created_at",
            parameters,
        ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def list_waiting(self) -> list[ReplenishmentJob]:
        placeholders = ", ".join("?" for _ in WAITING_STATUSES)
        rows = self._connection.execute(
            f"SELECT * FROM replenishment_jobs WHERE status IN ({placeholders}) "
            "ORDER BY created_at",
            WAITING_STATUSES,
        ).fetchall()
        return [self._job_from_row(row) for row in rows]

    def latest_for_skill(
        self, course_id: str, skill_id: str
    ) -> ReplenishmentJob | None:
        row = self._connection.execute(
            """
            SELECT * FROM replenishment_jobs
            WHERE course_id = ? AND skill_id = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (course_id, skill_id),
        ).fetchone()
        return self._job_from_row(row) if row is not None else None

    def _require(self, job_id: str) -> ReplenishmentJob:
        job = self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    @staticmethod
    def _job_from_row(row: sqlite3.Row) -> ReplenishmentJob:
        return ReplenishmentJob(
            job_id=row["job_id"],
            course_id=row["course_id"],
            skill_id=row["skill_id"],
            job_type=row["job_type"],
            status=row["status"],
            requested_count=row["requested_count"],
            attempts=row["attempts"],
            created_at=_aware_datetime(row["created_at"]),
            started_at=_optional_datetime(row["started_at"]),
            completed_at=_optional_datetime(row["completed_at"]),
            next_retry_at=_optional_datetime(row["next_retry_at"]),
            lease_expires_at=_optional_datetime(row["lease_expires_at"]),
            error_code=row["error_code"],
            error_message=row["error_message"],
            metadata=json.loads(row["metadata_json"]),
        )
