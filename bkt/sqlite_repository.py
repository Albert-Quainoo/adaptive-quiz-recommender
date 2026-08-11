import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from bkt.repository import AttemptConflictError
from bkt.schemas import AttemptEvent, BKTModelMetadata, MasterySnapshot


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
    """SQLite persistence for immutable attempts and versioned mastery history."""

    def __init__(self, database: str | Path) -> None:
        self.database = str(database)
        # Streamlit caches the shared controller across rerun threads. Controller
        # calls are serialized, so allow that single connection to follow them.
        self._connection = sqlite3.connect(self.database, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")

    def initialize_schema(self) -> None:
        self._connection.executescript(SCHEMA)

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> "SQLiteBKTRepository":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def save_attempt(self, attempt: AttemptEvent) -> None:
        existing = self.get_attempt(attempt.attempt_id)
        if existing is not None:
            if existing == attempt:
                return
            raise AttemptConflictError(
                f"attempt_id {attempt.attempt_id!r} is already used by a different attempt"
            )

        try:
            with self._connection:
                self._insert_attempt(attempt)
        except sqlite3.IntegrityError as exc:
            raise AttemptConflictError(str(exc)) from exc

    def get_attempt(self, attempt_id: str) -> AttemptEvent | None:
        row = self._connection.execute(
            "SELECT * FROM attempt_events WHERE attempt_id = ?", (attempt_id,)
        ).fetchone()
        return self._attempt_from_row(row) if row is not None else None

    def list_attempts(
        self, *, learner_id: str | None = None, skill_id: str | None = None
    ) -> list[AttemptEvent]:
        clauses: list[str] = []
        parameters: list[str] = []
        if learner_id is not None:
            clauses.append("learner_id = ?")
            parameters.append(learner_id)
        if skill_id is not None:
            clauses.append("skill_id = ?")
            parameters.append(skill_id)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._connection.execute(
            "SELECT * FROM attempt_events"
            f"{where} ORDER BY attempt_order, occurred_at, attempt_id",
            parameters,
        ).fetchall()
        return [self._attempt_from_row(row) for row in rows]

    def save_mastery(self, snapshot: MasterySnapshot) -> None:
        with self._connection:
            self._insert_mastery(snapshot)

    def save_attempt_and_mastery(
        self, attempt: AttemptEvent, snapshot: MasterySnapshot
    ) -> MasterySnapshot:
        if snapshot.source_attempt_id != attempt.attempt_id:
            raise ValueError("snapshot source_attempt_id must match the attempt")

        try:
            with self._connection:
                existing = self.get_attempt(attempt.attempt_id)
                if existing is not None:
                    if existing != attempt:
                        raise AttemptConflictError(
                            f"attempt_id {attempt.attempt_id!r} is already used by a different attempt"
                        )
                    stored = self.get_mastery_for_attempt(attempt.attempt_id)
                    if stored is None:
                        raise RuntimeError(
                            "stored attempt is missing its mastery snapshot"
                        )
                    return stored

                self._insert_attempt(attempt)
                self._insert_mastery(snapshot)
        except sqlite3.IntegrityError as exc:
            raise AttemptConflictError(str(exc)) from exc
        return snapshot.model_copy(deep=True)

    def get_mastery(
        self, learner_id: str, skill_id: str
    ) -> MasterySnapshot | None:
        row = self._connection.execute(
            """
            SELECT * FROM mastery_snapshots
            WHERE learner_id = ? AND skill_id = ?
            ORDER BY updated_at DESC, source_attempt_id DESC,
                     model_version DESC, snapshot_id DESC
            LIMIT 1
            """,
            (learner_id, skill_id),
        ).fetchone()
        return self._mastery_from_row(row) if row is not None else None

    def get_mastery_for_attempt(self, attempt_id: str) -> MasterySnapshot | None:
        row = self._connection.execute(
            """
            SELECT * FROM mastery_snapshots
            WHERE source_attempt_id = ?
            ORDER BY updated_at DESC, model_version DESC, snapshot_id DESC
            LIMIT 1
            """,
            (attempt_id,),
        ).fetchone()
        return self._mastery_from_row(row) if row is not None else None

    def list_mastery(
        self, *, learner_id: str | None = None
    ) -> list[MasterySnapshot]:
        if learner_id is None:
            rows = self._connection.execute(
                """
                SELECT * FROM mastery_snapshots
                ORDER BY updated_at, source_attempt_id, model_version, snapshot_id
                """
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT * FROM mastery_snapshots
                WHERE learner_id = ?
                ORDER BY updated_at, source_attempt_id, model_version, snapshot_id
                """,
                (learner_id,),
            ).fetchall()
        return [self._mastery_from_row(row) for row in rows]

    def save_model_metadata(self, metadata: BKTModelMetadata) -> None:
        values = (
            metadata.model_version,
            _utc_iso(metadata.fitted_at),
            metadata.training_attempt_count,
            json.dumps(metadata.skill_ids),
        )
        with self._connection:
            existing = self.get_model_metadata(metadata.model_version)
            if existing is not None:
                if existing == metadata:
                    return
                raise ValueError(
                    f"model_version {metadata.model_version!r} already exists"
                )
            self._connection.execute(
                """
                INSERT INTO bkt_model_metadata (
                    model_version, fitted_at, training_attempt_count, skill_ids
                ) VALUES (?, ?, ?, ?)
                """,
                values,
            )

    def get_model_metadata(self, model_version: str) -> BKTModelMetadata | None:
        row = self._connection.execute(
            "SELECT * FROM bkt_model_metadata WHERE model_version = ?",
            (model_version,),
        ).fetchone()
        if row is None:
            return None
        return BKTModelMetadata(
            model_version=row["model_version"],
            fitted_at=_aware_datetime(row["fitted_at"]),
            training_attempt_count=row["training_attempt_count"],
            skill_ids=json.loads(row["skill_ids"]),
        )

    def _insert_attempt(self, attempt: AttemptEvent) -> None:
        self._connection.execute(
            """
            INSERT INTO attempt_events (
                attempt_id, presentation_id, learner_id, item_id, skill_id,
                selected_option_id, correct, attempt_order, occurred_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                attempt.attempt_id,
                attempt.presentation_id,
                attempt.learner_id,
                attempt.item_id,
                attempt.skill_id,
                attempt.selected_option_id,
                int(attempt.correct),
                attempt.attempt_order,
                _utc_iso(attempt.occurred_at),
            ),
        )

    def _insert_mastery(self, snapshot: MasterySnapshot) -> None:
        existing = self._connection.execute(
            """
            SELECT * FROM mastery_snapshots
            WHERE source_attempt_id = ? AND model_version = ?
            """,
            (snapshot.source_attempt_id, snapshot.model_version),
        ).fetchone()
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

        self._connection.execute(
            """
            INSERT INTO mastery_snapshots (
                snapshot_id, learner_id, skill_id, mastery_probability,
                attempt_count, source_attempt_id, model_version, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(uuid4()),
                snapshot.learner_id,
                snapshot.skill_id,
                snapshot.mastery_probability,
                snapshot.attempt_count,
                snapshot.source_attempt_id,
                snapshot.model_version,
                _utc_iso(snapshot.updated_at),
            ),
        )

    @staticmethod
    def _attempt_from_row(row: sqlite3.Row) -> AttemptEvent:
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
    def _mastery_from_row(row: sqlite3.Row) -> MasterySnapshot:
        return MasterySnapshot(
            learner_id=row["learner_id"],
            skill_id=row["skill_id"],
            mastery_probability=row["mastery_probability"],
            attempt_count=row["attempt_count"],
            source_attempt_id=row["source_attempt_id"],
            model_version=row["model_version"],
            updated_at=_aware_datetime(row["updated_at"]),
        )
