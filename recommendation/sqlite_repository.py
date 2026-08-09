from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from api.bank import BankItem
from bkt.schemas import MasterySnapshot
from bkt.sqlite_repository import SQLiteBKTRepository, _aware_datetime, _utc_iso
from recommendation.schemas import RecommendationEvent, RecommendationResult
from taxonomy.schemas import SkillDefinition


class SQLiteRecommendationRepository(SQLiteBKTRepository):
    """SQLite adaptive-loop repository with an in-memory taxonomy and item bank."""

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
        rows = self._connection.execute(
            """
            SELECT * FROM mastery_snapshots AS candidate
            WHERE candidate.learner_id = ?
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
            """,
            (learner_id,),
        ).fetchall()
        return [self._mastery_from_row(row) for row in rows]

    def list_approved_buildable_items(self) -> list[BankItem]:
        return [item.model_copy(deep=True) for item in self._items]

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
        with self._connection:
            self._connection.execute(
                """
                INSERT INTO recommendation_events (
                    recommendation_id, learner_id, skill_id, item_id,
                    difficulty, mastery_probability, reason, model_version,
                    policy_version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.recommendation_id,
                    event.learner_id,
                    event.skill_id,
                    event.item_id,
                    event.difficulty,
                    event.mastery_probability,
                    event.reason,
                    event.model_version,
                    event.policy_version,
                    _utc_iso(event.created_at),
                ),
            )
        return event

    def list_recommendations(
        self, *, learner_id: str | None = None
    ) -> list[RecommendationEvent]:
        if learner_id is None:
            rows = self._connection.execute(
                """
                SELECT * FROM recommendation_events
                ORDER BY created_at, recommendation_id
                """
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT * FROM recommendation_events
                WHERE learner_id = ?
                ORDER BY created_at, recommendation_id
                """,
                (learner_id,),
            ).fetchall()
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
