import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import Connection, text
from sqlalchemy.engine import RowMapping
from sqlalchemy.exc import IntegrityError

from bkt.repository import AttemptConflictError
from bkt.schemas import AttemptEvent, BKTModelMetadata, MasterySnapshot
from database import create_engine_for, execute_schema_script


SCHEMA = """
CREATE TABLE IF NOT EXISTS attempt_events (
    attempt_id TEXT PRIMARY KEY,
    presentation_id TEXT NOT NULL,
    learner_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    selected_option_id TEXT NOT NULL,
    correct INTEGER NOT NULL CHECK (correct IN (0, 1)),
    attempt_order INTEGER NOT NULL,
    occurred_at TEXT NOT NULL,
    UNIQUE (learner_id, skill_id, attempt_order)
);

CREATE TABLE IF NOT EXISTS mastery_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    mastery_probability REAL NOT NULL,
    attempt_count INTEGER NOT NULL,
    source_attempt_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (source_attempt_id) REFERENCES attempt_events(attempt_id),
    UNIQUE (source_attempt_id, model_version)
);

CREATE TABLE IF NOT EXISTS recommendation_events (
    recommendation_id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    mastery_probability REAL NOT NULL,
    reason TEXT NOT NULL,
    model_version TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS content_gap_events (
    content_gap_id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL,
    completed_skill_id TEXT NOT NULL,
    completed_skill_name TEXT NOT NULL,
    newly_unlocked_skill_id TEXT NOT NULL,
    newly_unlocked_skill_name TEXT NOT NULL,
    missing_approved_content INTEGER NOT NULL CHECK (missing_approved_content IN (0, 1)),
    current_mastery_probability REAL NOT NULL,
    prerequisite_mastery_threshold REAL NOT NULL,
    reason TEXT NOT NULL,
    model_version TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bkt_model_metadata (
    model_version TEXT PRIMARY KEY,
    fitted_at TEXT NOT NULL,
    training_attempt_count INTEGER NOT NULL,
    skill_ids TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS learner_sessions (
    learner_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS question_presentations (
    presentation_id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    presentation_seed TEXT NOT NULL,
    recommendation_reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (learner_id) REFERENCES learner_sessions(learner_id)
);
"""


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _aware_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("stored timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


class SQLiteBKTRepository:
    """SQL persistence (SQLite or PostgreSQL) for immutable attempts and
    versioned mastery history."""

    def __init__(self, database: str | Path) -> None:
        self.database = str(database)
        self._engine = create_engine_for(database)

    def initialize_schema(self) -> None:
        with self._engine.begin() as connection:
            execute_schema_script(connection, SCHEMA)

    def close(self) -> None:
        self._engine.dispose()

    def __enter__(self) -> "SQLiteBKTRepository":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def save_attempt(self, attempt: AttemptEvent) -> None:
        with self._engine.connect() as connection:
            existing = self._select_attempt(connection, attempt.attempt_id)
        if existing is not None:
            if existing == attempt:
                return
            raise AttemptConflictError(
                f"attempt_id {attempt.attempt_id!r} is already used by a different attempt"
            )

        try:
            with self._engine.begin() as connection:
                self._insert_attempt(connection, attempt)
        except IntegrityError as exc:
            raise AttemptConflictError(str(exc)) from exc

    def get_attempt(self, attempt_id: str) -> AttemptEvent | None:
        with self._engine.connect() as connection:
            return self._select_attempt(connection, attempt_id)

    def list_attempts(
        self, *, learner_id: str | None = None, skill_id: str | None = None
    ) -> list[AttemptEvent]:
        clauses: list[str] = []
        parameters: dict[str, str] = {}
        if learner_id is not None:
            clauses.append("learner_id = :learner_id")
            parameters["learner_id"] = learner_id
        if skill_id is not None:
            clauses.append("skill_id = :skill_id")
            parameters["skill_id"] = skill_id
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._engine.connect() as connection:
            rows = (
                connection.execute(
                    text(
                        "SELECT * FROM attempt_events"
                        f"{where} ORDER BY attempt_order, occurred_at, attempt_id"
                    ),
                    parameters,
                )
                .mappings()
                .all()
            )
        return [self._attempt_from_row(row) for row in rows]

    def save_mastery(self, snapshot: MasterySnapshot) -> None:
        with self._engine.begin() as connection:
            self._insert_mastery(connection, snapshot)

    def save_attempt_and_mastery(
        self, attempt: AttemptEvent, snapshot: MasterySnapshot
    ) -> MasterySnapshot:
        if snapshot.source_attempt_id != attempt.attempt_id:
            raise ValueError("snapshot source_attempt_id must match the attempt")

        try:
            with self._engine.begin() as connection:
                existing = self._select_attempt(connection, attempt.attempt_id)
                if existing is not None:
                    if existing != attempt:
                        raise AttemptConflictError(
                            f"attempt_id {attempt.attempt_id!r} is already used by a different attempt"
                        )
                    stored = self._select_mastery_for_attempt(
                        connection, attempt.attempt_id
                    )
                    if stored is None:
                        raise RuntimeError(
                            "stored attempt is missing its mastery snapshot"
                        )
                    return stored

                self._insert_attempt(connection, attempt)
                self._insert_mastery(connection, snapshot)
        except IntegrityError as exc:
            raise AttemptConflictError(str(exc)) from exc
        return snapshot.model_copy(deep=True)

    def get_mastery(
        self, learner_id: str, skill_id: str
    ) -> MasterySnapshot | None:
        with self._engine.connect() as connection:
            row = (
                connection.execute(
                    text(
                        """
                        SELECT * FROM mastery_snapshots
                        WHERE learner_id = :learner_id AND skill_id = :skill_id
                        ORDER BY updated_at DESC, source_attempt_id DESC,
                                 model_version DESC, snapshot_id DESC
                        LIMIT 1
                        """
                    ),
                    {"learner_id": learner_id, "skill_id": skill_id},
                )
                .mappings()
                .first()
            )
        return self._mastery_from_row(row) if row is not None else None

    def get_mastery_for_attempt(self, attempt_id: str) -> MasterySnapshot | None:
        with self._engine.connect() as connection:
            return self._select_mastery_for_attempt(connection, attempt_id)

    def list_mastery(
        self, *, learner_id: str | None = None
    ) -> list[MasterySnapshot]:
        with self._engine.connect() as connection:
            if learner_id is None:
                rows = (
                    connection.execute(
                        text(
                            """
                            SELECT * FROM mastery_snapshots
                            ORDER BY updated_at, source_attempt_id, model_version, snapshot_id
                            """
                        )
                    )
                    .mappings()
                    .all()
                )
            else:
                rows = (
                    connection.execute(
                        text(
                            """
                            SELECT * FROM mastery_snapshots
                            WHERE learner_id = :learner_id
                            ORDER BY updated_at, source_attempt_id, model_version, snapshot_id
                            """
                        ),
                        {"learner_id": learner_id},
                    )
                    .mappings()
                    .all()
                )
        return [self._mastery_from_row(row) for row in rows]

    def save_model_metadata(self, metadata: BKTModelMetadata) -> None:
        with self._engine.begin() as connection:
            existing = self._select_model_metadata(connection, metadata.model_version)
            if existing is not None:
                if existing == metadata:
                    return
                raise ValueError(
                    f"model_version {metadata.model_version!r} already exists"
                )
            connection.execute(
                text(
                    """
                    INSERT INTO bkt_model_metadata (
                        model_version, fitted_at, training_attempt_count, skill_ids
                    ) VALUES (:model_version, :fitted_at, :training_attempt_count, :skill_ids)
                    """
                ),
                {
                    "model_version": metadata.model_version,
                    "fitted_at": _utc_iso(metadata.fitted_at),
                    "training_attempt_count": metadata.training_attempt_count,
                    "skill_ids": json.dumps(metadata.skill_ids),
                },
            )

    def get_model_metadata(self, model_version: str) -> BKTModelMetadata | None:
        with self._engine.connect() as connection:
            return self._select_model_metadata(connection, model_version)

    def _select_attempt(
        self, connection: Connection, attempt_id: str
    ) -> AttemptEvent | None:
        row = (
            connection.execute(
                text("SELECT * FROM attempt_events WHERE attempt_id = :attempt_id"),
                {"attempt_id": attempt_id},
            )
            .mappings()
            .first()
        )
        return self._attempt_from_row(row) if row is not None else None

    def _select_mastery_for_attempt(
        self, connection: Connection, attempt_id: str
    ) -> MasterySnapshot | None:
        row = (
            connection.execute(
                text(
                    """
                    SELECT * FROM mastery_snapshots
                    WHERE source_attempt_id = :attempt_id
                    ORDER BY updated_at DESC, model_version DESC, snapshot_id DESC
                    LIMIT 1
                    """
                ),
                {"attempt_id": attempt_id},
            )
            .mappings()
            .first()
        )
        return self._mastery_from_row(row) if row is not None else None

    def _select_model_metadata(
        self, connection: Connection, model_version: str
    ) -> BKTModelMetadata | None:
        row = (
            connection.execute(
                text(
                    "SELECT * FROM bkt_model_metadata WHERE model_version = :model_version"
                ),
                {"model_version": model_version},
            )
            .mappings()
            .first()
        )
        if row is None:
            return None
        return BKTModelMetadata(
            model_version=row["model_version"],
            fitted_at=_aware_datetime(row["fitted_at"]),
            training_attempt_count=row["training_attempt_count"],
            skill_ids=json.loads(row["skill_ids"]),
        )

    def _insert_attempt(self, connection: Connection, attempt: AttemptEvent) -> None:
        connection.execute(
            text(
                """
                INSERT INTO attempt_events (
                    attempt_id, presentation_id, learner_id, item_id, skill_id,
                    selected_option_id, correct, attempt_order, occurred_at
                ) VALUES (
                    :attempt_id, :presentation_id, :learner_id, :item_id, :skill_id,
                    :selected_option_id, :correct, :attempt_order, :occurred_at
                )
                """
            ),
            {
                "attempt_id": attempt.attempt_id,
                "presentation_id": attempt.presentation_id,
                "learner_id": attempt.learner_id,
                "item_id": attempt.item_id,
                "skill_id": attempt.skill_id,
                "selected_option_id": attempt.selected_option_id,
                "correct": int(attempt.correct),
                "attempt_order": attempt.attempt_order,
                "occurred_at": _utc_iso(attempt.occurred_at),
            },
        )

    def _insert_mastery(self, connection: Connection, snapshot: MasterySnapshot) -> None:
        existing = (
            connection.execute(
                text(
                    """
                    SELECT * FROM mastery_snapshots
                    WHERE source_attempt_id = :source_attempt_id AND model_version = :model_version
                    """
                ),
                {
                    "source_attempt_id": snapshot.source_attempt_id,
                    "model_version": snapshot.model_version,
                },
            )
            .mappings()
            .first()
        )
        if existing is not None:
            stored = self._mastery_from_row(existing)
            if (
                stored.learner_id == snapshot.learner_id
                and stored.skill_id == snapshot.skill_id
                and stored.mastery_probability == snapshot.mastery_probability
                and stored.attempt_count == snapshot.attempt_count
            ):
                return
            raise ValueError(
                "mastery snapshot already exists for source attempt and model version"
            )

        connection.execute(
            text(
                """
                INSERT INTO mastery_snapshots (
                    snapshot_id, learner_id, skill_id, mastery_probability,
                    attempt_count, source_attempt_id, model_version, updated_at
                ) VALUES (
                    :snapshot_id, :learner_id, :skill_id, :mastery_probability,
                    :attempt_count, :source_attempt_id, :model_version, :updated_at
                )
                """
            ),
            {
                "snapshot_id": str(uuid4()),
                "learner_id": snapshot.learner_id,
                "skill_id": snapshot.skill_id,
                "mastery_probability": snapshot.mastery_probability,
                "attempt_count": snapshot.attempt_count,
                "source_attempt_id": snapshot.source_attempt_id,
                "model_version": snapshot.model_version,
                "updated_at": _utc_iso(snapshot.updated_at),
            },
        )

    @staticmethod
    def _attempt_from_row(row: RowMapping) -> AttemptEvent:
        return AttemptEvent(
            attempt_id=row["attempt_id"],
            presentation_id=row["presentation_id"],
            learner_id=row["learner_id"],
            item_id=row["item_id"],
            skill_id=row["skill_id"],
            selected_option_id=row["selected_option_id"],
            correct=bool(row["correct"]),
            attempt_order=row["attempt_order"],
            occurred_at=_aware_datetime(row["occurred_at"]),
        )

    @staticmethod
    def _mastery_from_row(row: RowMapping) -> MasterySnapshot:
        return MasterySnapshot(
            learner_id=row["learner_id"],
            skill_id=row["skill_id"],
            mastery_probability=row["mastery_probability"],
            attempt_count=row["attempt_count"],
            source_attempt_id=row["source_attempt_id"],
            model_version=row["model_version"],
            updated_at=_aware_datetime(row["updated_at"]),
        )
