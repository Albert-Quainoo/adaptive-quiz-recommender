"""SQL-backed (SQLite or PostgreSQL) append-only store for CourseApprovalRecord.

Mirrors the connect/schema/transaction conventions of
authoring/replenishment/jobs.py: BEGIN IMMEDIATE on SQLite so concurrent CLI
invocations can't race past the sequence-number check, and a UNIQUE
(course_id, sequence_number) constraint as the backstop -- a race that slips
past BEGIN IMMEDIATE (e.g. against Postgres, which does not honor
immediate_transactions) is caught as CourseApprovalConflictError rather than
silently producing two records with the same sequence number.

No update or delete method exists on this repository -- that omission is
what makes the audit trail immutable, the same convention
bkt/sqlite_repository.py uses for attempt_events.
"""

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text
from sqlalchemy.engine import RowMapping
from sqlalchemy.exc import IntegrityError

from authoring.course_catalog.records import CourseApprovalRecord
from database import create_engine_for, execute_schema_script

SCHEMA = """
CREATE TABLE IF NOT EXISTS course_approval_records (
    record_id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    sequence_number INTEGER NOT NULL,
    lifecycle_status TEXT NOT NULL,
    course_profile_version TEXT NOT NULL,
    taxonomy_version TEXT,
    approved_bank_version TEXT,
    bkt_model_version TEXT,
    decision TEXT NOT NULL,
    approver_identity TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    notes TEXT,
    auto_activate_when_ready INTEGER NOT NULL CHECK (auto_activate_when_ready IN (0, 1)),
    UNIQUE (course_id, sequence_number)
);
"""


class CourseApprovalConflictError(RuntimeError):
    pass


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _aware_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("stored timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


class SQLiteCourseApprovalRepository:
    def __init__(self, database: str | Path) -> None:
        self.database = str(database)
        self._engine = create_engine_for(database, immediate_transactions=True)

    def initialize_schema(self) -> None:
        with self._engine.begin() as connection:
            execute_schema_script(connection, SCHEMA)

    def close(self) -> None:
        self._engine.dispose()

    def __enter__(self) -> "SQLiteCourseApprovalRepository":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def append(
        self,
        *,
        course_id: str,
        lifecycle_status: str,
        course_profile_version: str,
        decision: str,
        approver_identity: str,
        decided_at: datetime,
        auto_activate_when_ready: bool,
        taxonomy_version: str | None = None,
        approved_bank_version: str | None = None,
        bkt_model_version: str | None = None,
        notes: str | None = None,
    ) -> CourseApprovalRecord:
        try:
            with self._engine.begin() as connection:
                next_sequence = self._next_sequence_number(connection, course_id)
                record = CourseApprovalRecord(
                    record_id=str(uuid4()),
                    course_id=course_id,
                    sequence_number=next_sequence,
                    lifecycle_status=lifecycle_status,
                    course_profile_version=course_profile_version,
                    taxonomy_version=taxonomy_version,
                    approved_bank_version=approved_bank_version,
                    bkt_model_version=bkt_model_version,
                    decision=decision,
                    approver_identity=approver_identity,
                    decided_at=decided_at,
                    notes=notes,
                    auto_activate_when_ready=auto_activate_when_ready,
                )
                self._insert(connection, record)
        except IntegrityError as exc:
            raise CourseApprovalConflictError(str(exc)) from exc
        return record

    def list_for_course(self, course_id: str) -> list[CourseApprovalRecord]:
        with self._engine.connect() as connection:
            rows = (
                connection.execute(
                    text(
                        """
                        SELECT * FROM course_approval_records
                        WHERE course_id = :course_id
                        ORDER BY sequence_number
                        """
                    ),
                    {"course_id": course_id},
                )
                .mappings()
                .all()
            )
        return [self._record_from_row(row) for row in rows]

    def latest_for_course(self, course_id: str) -> CourseApprovalRecord | None:
        with self._engine.connect() as connection:
            row = (
                connection.execute(
                    text(
                        """
                        SELECT * FROM course_approval_records
                        WHERE course_id = :course_id
                        ORDER BY sequence_number DESC
                        LIMIT 1
                        """
                    ),
                    {"course_id": course_id},
                )
                .mappings()
                .first()
            )
        return self._record_from_row(row) if row is not None else None

    def _next_sequence_number(self, connection, course_id: str) -> int:
        current_max = connection.execute(
            text(
                """
                SELECT COALESCE(MAX(sequence_number), 0) FROM course_approval_records
                WHERE course_id = :course_id
                """
            ),
            {"course_id": course_id},
        ).scalar_one()
        return int(current_max) + 1

    def _insert(self, connection, record: CourseApprovalRecord) -> None:
        connection.execute(
            text(
                """
                INSERT INTO course_approval_records (
                    record_id, course_id, sequence_number, lifecycle_status,
                    course_profile_version, taxonomy_version, approved_bank_version,
                    bkt_model_version, decision, approver_identity, decided_at,
                    notes, auto_activate_when_ready
                ) VALUES (
                    :record_id, :course_id, :sequence_number, :lifecycle_status,
                    :course_profile_version, :taxonomy_version, :approved_bank_version,
                    :bkt_model_version, :decision, :approver_identity, :decided_at,
                    :notes, :auto_activate_when_ready
                )
                """
            ),
            {
                "record_id": record.record_id,
                "course_id": record.course_id,
                "sequence_number": record.sequence_number,
                "lifecycle_status": record.lifecycle_status,
                "course_profile_version": record.course_profile_version,
                "taxonomy_version": record.taxonomy_version,
                "approved_bank_version": record.approved_bank_version,
                "bkt_model_version": record.bkt_model_version,
                "decision": record.decision,
                "approver_identity": record.approver_identity,
                "decided_at": _utc_iso(record.decided_at),
                "notes": record.notes,
                "auto_activate_when_ready": int(record.auto_activate_when_ready),
            },
        )

    @staticmethod
    def _record_from_row(row: RowMapping) -> CourseApprovalRecord:
        return CourseApprovalRecord(
            record_id=row["record_id"],
            course_id=row["course_id"],
            sequence_number=row["sequence_number"],
            lifecycle_status=row["lifecycle_status"],
            course_profile_version=row["course_profile_version"],
            taxonomy_version=row["taxonomy_version"],
            approved_bank_version=row["approved_bank_version"],
            bkt_model_version=row["bkt_model_version"],
            decision=row["decision"],
            approver_identity=row["approver_identity"],
            decided_at=_aware_datetime(row["decided_at"]),
            notes=row["notes"],
            auto_activate_when_ready=bool(row["auto_activate_when_ready"]),
        )
