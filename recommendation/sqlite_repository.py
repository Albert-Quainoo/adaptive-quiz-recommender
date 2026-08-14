from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import text

from api.bank import BankItem
from bkt.schemas import MasterySnapshot
from bkt.sqlite_repository import SQLiteBKTRepository, _aware_datetime, _utc_iso
from recommendation.schemas import (
    ContentGapEvent,
    ContentGapResult,
    RecommendationEvent,
    RecommendationResult,
)
from taxonomy.schemas import SkillDefinition


class SQLiteRecommendationRepository(SQLiteBKTRepository):
    """SQL adaptive-loop repository (SQLite or PostgreSQL) with an
    in-memory taxonomy and item bank."""

    def __init__(
        self,
        database: str | Path,
        *,
        skills: Sequence[SkillDefinition],
        items: Sequence[BankItem],
    ) -> None:
        super().__init__(database)
        self._skills = [skill.model_copy(deep=True) for skill in skills]
        self._items = [item.model_copy(deep=True) for item in items]

    def list_skills(self) -> list[SkillDefinition]:
        return [skill.model_copy(deep=True) for skill in self._skills]

    def list_latest_mastery(self, learner_id: str) -> list[MasterySnapshot]:
        with self._engine.connect() as connection:
            rows = (
                connection.execute(
                    text(
                        """
                        SELECT * FROM mastery_snapshots AS candidate
                        WHERE candidate.learner_id = :learner_id
                          AND candidate.snapshot_id = (
                              SELECT latest.snapshot_id
                              FROM mastery_snapshots AS latest
                              WHERE latest.learner_id = candidate.learner_id
                                AND latest.skill_id = candidate.skill_id
                              ORDER BY latest.updated_at DESC,
                                       latest.source_attempt_id DESC,
                                       latest.model_version DESC,
                                       latest.snapshot_id DESC
                              LIMIT 1
                          )
                        ORDER BY candidate.skill_id
                        """
                    ),
                    {"learner_id": learner_id},
                )
                .mappings()
                .all()
            )
        return [self._mastery_from_row(row) for row in rows]

    def list_approved_buildable_items(self) -> list[BankItem]:
        return [item.model_copy(deep=True) for item in self._items]

    def save_learner(self, learner_id: str, *, created_at: datetime) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO learner_sessions (learner_id, created_at)
                    VALUES (:learner_id, :created_at)
                    ON CONFLICT (learner_id) DO NOTHING
                    """
                ),
                {"learner_id": learner_id, "created_at": _utc_iso(created_at)},
            )

    def learner_exists(self, learner_id: str) -> bool:
        with self._engine.connect() as connection:
            row = connection.execute(
                text("SELECT 1 FROM learner_sessions WHERE learner_id = :learner_id"),
                {"learner_id": learner_id},
            ).first()
        return row is not None

    def save_presentation(
        self,
        *,
        presentation_id: str,
        learner_id: str,
        item_id: str,
        presentation_seed: int,
        recommendation_reason: str,
        created_at: datetime,
    ) -> None:
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO question_presentations (
                        presentation_id, learner_id, item_id, presentation_seed,
                        recommendation_reason, created_at
                    ) VALUES (
                        :presentation_id, :learner_id, :item_id, :presentation_seed,
                        :recommendation_reason, :created_at
                    )
                    """
                ),
                {
                    "presentation_id": presentation_id,
                    "learner_id": learner_id,
                    "item_id": item_id,
                    "presentation_seed": str(presentation_seed),
                    "recommendation_reason": recommendation_reason,
                    "created_at": _utc_iso(created_at),
                },
            )

    def get_presentation(self, presentation_id: str):
        with self._engine.connect() as connection:
            return (
                connection.execute(
                    text(
                        "SELECT * FROM question_presentations WHERE presentation_id = :presentation_id"
                    ),
                    {"presentation_id": presentation_id},
                )
                .mappings()
                .first()
            )

    def list_attempts(
        self, learner_id: str | None = None, skill_id: str | None = None
    ):
        return super().list_attempts(learner_id=learner_id, skill_id=skill_id)

    def save_recommendation(
        self, recommendation: RecommendationResult
    ) -> RecommendationEvent:
        event = RecommendationEvent(
            **recommendation.model_dump(),
            recommendation_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
        )
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO recommendation_events (
                        recommendation_id, learner_id, skill_id, item_id,
                        difficulty, mastery_probability, reason, model_version,
                        policy_version, created_at
                    ) VALUES (
                        :recommendation_id, :learner_id, :skill_id, :item_id,
                        :difficulty, :mastery_probability, :reason, :model_version,
                        :policy_version, :created_at
                    )
                    """
                ),
                {
                    "recommendation_id": event.recommendation_id,
                    "learner_id": event.learner_id,
                    "skill_id": event.skill_id,
                    "item_id": event.item_id,
                    "difficulty": event.difficulty,
                    "mastery_probability": event.mastery_probability,
                    "reason": event.reason,
                    "model_version": event.model_version,
                    "policy_version": event.policy_version,
                    "created_at": _utc_iso(event.created_at),
                },
            )
        return event

    def list_recommendations(
        self, *, learner_id: str | None = None
    ) -> list[RecommendationEvent]:
        with self._engine.connect() as connection:
            if learner_id is None:
                rows = (
                    connection.execute(
                        text(
                            """
                            SELECT * FROM recommendation_events
                            ORDER BY created_at, recommendation_id
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
                            SELECT * FROM recommendation_events
                            WHERE learner_id = :learner_id
                            ORDER BY created_at, recommendation_id
                            """
                        ),
                        {"learner_id": learner_id},
                    )
                    .mappings()
                    .all()
                )
        return [
            RecommendationEvent(
                recommendation_id=row["recommendation_id"],
                learner_id=row["learner_id"],
                skill_id=row["skill_id"],
                item_id=row["item_id"],
                difficulty=row["difficulty"],
                mastery_probability=row["mastery_probability"],
                reason=row["reason"],
                model_version=row["model_version"],
                policy_version=row["policy_version"],
                created_at=_aware_datetime(row["created_at"]),
            )
            for row in rows
        ]

    def save_content_gap(self, gap: ContentGapResult) -> ContentGapEvent:
        event = ContentGapEvent(
            **gap.model_dump(),
            content_gap_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
        )
        with self._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO content_gap_events (
                        content_gap_id, learner_id, completed_skill_id,
                        completed_skill_name, newly_unlocked_skill_id,
                        newly_unlocked_skill_name, missing_approved_content,
                        current_mastery_probability, prerequisite_mastery_threshold,
                        reason, model_version, policy_version, created_at
                    ) VALUES (
                        :content_gap_id, :learner_id, :completed_skill_id,
                        :completed_skill_name, :newly_unlocked_skill_id,
                        :newly_unlocked_skill_name, :missing_approved_content,
                        :current_mastery_probability, :prerequisite_mastery_threshold,
                        :reason, :model_version, :policy_version, :created_at
                    )
                    """
                ),
                {
                    "content_gap_id": event.content_gap_id,
                    "learner_id": event.learner_id,
                    "completed_skill_id": event.completed_skill_id,
                    "completed_skill_name": event.completed_skill_name,
                    "newly_unlocked_skill_id": event.newly_unlocked_skill_id,
                    "newly_unlocked_skill_name": event.newly_unlocked_skill_name,
                    "missing_approved_content": int(event.missing_approved_content),
                    "current_mastery_probability": event.current_mastery_probability,
                    "prerequisite_mastery_threshold": event.prerequisite_mastery_threshold,
                    "reason": event.reason,
                    "model_version": event.model_version,
                    "policy_version": event.policy_version,
                    "created_at": _utc_iso(event.created_at),
                },
            )
        return event

    def list_content_gaps(
        self, *, learner_id: str | None = None
    ) -> list[ContentGapEvent]:
        with self._engine.connect() as connection:
            if learner_id is None:
                rows = (
                    connection.execute(
                        text(
                            "SELECT * FROM content_gap_events ORDER BY created_at, content_gap_id"
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
                            SELECT * FROM content_gap_events
                            WHERE learner_id = :learner_id
                            ORDER BY created_at, content_gap_id
                            """
                        ),
                        {"learner_id": learner_id},
                    )
                    .mappings()
                    .all()
                )
        return [
            ContentGapEvent(
                content_gap_id=row["content_gap_id"],
                learner_id=row["learner_id"],
                completed_skill_id=row["completed_skill_id"],
                completed_skill_name=row["completed_skill_name"],
                newly_unlocked_skill_id=row["newly_unlocked_skill_id"],
                newly_unlocked_skill_name=row["newly_unlocked_skill_name"],
                missing_approved_content=bool(row["missing_approved_content"]),
                current_mastery_probability=row["current_mastery_probability"],
                prerequisite_mastery_threshold=row["prerequisite_mastery_threshold"],
                reason=row["reason"],
                model_version=row["model_version"],
                policy_version=row["policy_version"],
                created_at=_aware_datetime(row["created_at"]),
            )
            for row in rows
        ]
