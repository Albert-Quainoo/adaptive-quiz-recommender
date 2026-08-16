import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from sqlalchemy import event, text

from api.bank import BankItem
from bkt.schemas import MasterySnapshot
from bkt.sqlite_repository import SQLiteBKTRepository, _aware_datetime, _utc_iso
from database import connection_scope
from recommendation.schemas import (
    ContentGapEvent,
    ContentGapResult,
    RecommendationEvent,
    RecommendationResult,
)
from taxonomy.schemas import SkillDefinition


class SQLiteRecommendationRepository(SQLiteBKTRepository):
    """SQL adaptive-loop repository (SQLite or PostgreSQL) with an
    in-memory taxonomy and item bank, bound to exactly one course (see
    SQLiteBKTRepository's course_id binding). skills/items are already
    that course's own -- they are never loaded from a shared, unfiltered
    source -- so course isolation here is enforced twice over: once by the
    in-memory catalogue this instance was constructed with, once by every
    query filtering on self.course_id."""

    def __init__(
        self,
        database: str | Path,
        *,
        course_id: str,
        skills: Sequence[SkillDefinition],
        items: Sequence[BankItem],
    ) -> None:
        super().__init__(database, course_id=course_id)
        self._skills = [skill.model_copy(deep=True) for skill in skills]
        self._items = [item.model_copy(deep=True) for item in items]
        self._confirmed_learners: set[str] = set()
        self._confirmed_learners_lock = threading.Lock()

    def list_skills(self) -> list[SkillDefinition]:
        return [skill.model_copy(deep=True) for skill in self._skills]

    def list_latest_mastery(self, learner_id: str) -> list[MasterySnapshot]:
        with connection_scope(self._engine, write=False) as connection:
            rows = (
                connection.execute(
                    text(
                        """
                        SELECT * FROM mastery_snapshots AS candidate
                        WHERE candidate.learner_id = :learner_id
                          AND candidate.course_id = :course_id
                          AND candidate.snapshot_id = (
                              SELECT latest.snapshot_id
                              FROM mastery_snapshots AS latest
                              WHERE latest.learner_id = candidate.learner_id
                                AND latest.course_id = candidate.course_id
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
                    {"learner_id": learner_id, "course_id": self.course_id},
                )
                .mappings()
                .all()
            )
        return [self._mastery_from_row(row) for row in rows]

    def list_approved_buildable_items(self) -> list[BankItem]:
        return [item.model_copy(deep=True) for item in self._items]

    def save_learner(self, learner_id: str, *, created_at: datetime) -> None:
        """Already idempotent at the database level (ON CONFLICT DO
        NOTHING) even without the in-memory check below -- but a
        learner_sessions row, once *committed*, is never deleted or
        modified for the lifetime of this repository instance (this
        repository, and this set, persist across many learner actions over
        the whole process's lifetime -- see app/multi_course.py's
        CourseCatalogController caching), so remembering "already
        confirmed" here safely skips the round trip on every call after
        the first for the same learner, not just within one request.

        Marking a learner confirmed is deferred to this connection's own
        "commit" event rather than done right after the INSERT executes:
        when this call is participating in a caller's unit_of_work (see
        database.py), that shared connection's transaction can still be
        rolled back later by something *else* that fails further down the
        same action -- if this cache were updated immediately, a later
        rollback would leave the row NOT actually in the database while
        this process still believed it was, permanently skipping the
        write (until process restart) and breaking every future
        foreign-key-dependent write for that learner. The "commit" event
        never fires on rollback (verified), so the cache can never drift
        ahead of what is actually durably stored. Not a caller-supplied
        flag: nothing external can mark a learner "confirmed" except this
        method's own write actually having committed."""
        with self._confirmed_learners_lock:
            if learner_id in self._confirmed_learners:
                return
        with connection_scope(self._engine, write=True) as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO learner_sessions (learner_id, course_id, created_at)
                    VALUES (:learner_id, :course_id, :created_at)
                    ON CONFLICT (learner_id, course_id) DO NOTHING
                    """
                ),
                {
                    "learner_id": learner_id,
                    "course_id": self.course_id,
                    "created_at": _utc_iso(created_at),
                },
            )

            def _mark_confirmed(_connection: object) -> None:
                with self._confirmed_learners_lock:
                    self._confirmed_learners.add(learner_id)

            event.listen(connection, "commit", _mark_confirmed, once=True)

    def learner_exists(self, learner_id: str) -> bool:
        with connection_scope(self._engine, write=False) as connection:
            row = connection.execute(
                text(
                    """
                    SELECT 1 FROM learner_sessions
                    WHERE learner_id = :learner_id AND course_id = :course_id
                    """
                ),
                {"learner_id": learner_id, "course_id": self.course_id},
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
        with connection_scope(self._engine, write=True) as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO question_presentations (
                        presentation_id, course_id, learner_id, item_id,
                        presentation_seed, recommendation_reason, created_at
                    ) VALUES (
                        :presentation_id, :course_id, :learner_id, :item_id,
                        :presentation_seed, :recommendation_reason, :created_at
                    )
                    """
                ),
                {
                    "presentation_id": presentation_id,
                    "course_id": self.course_id,
                    "learner_id": learner_id,
                    "item_id": item_id,
                    "presentation_seed": str(presentation_seed),
                    "recommendation_reason": recommendation_reason,
                    "created_at": _utc_iso(created_at),
                },
            )

    def get_presentation(self, presentation_id: str):
        with connection_scope(self._engine, write=False) as connection:
            return (
                connection.execute(
                    text(
                        """
                        SELECT * FROM question_presentations
                        WHERE presentation_id = :presentation_id AND course_id = :course_id
                        """
                    ),
                    {"presentation_id": presentation_id, "course_id": self.course_id},
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
        self._require_own_course(recommendation.course_id, what="recommendation")
        event = RecommendationEvent(
            **recommendation.model_dump(),
            recommendation_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
        )
        with connection_scope(self._engine, write=True) as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO recommendation_events (
                        recommendation_id, course_id, learner_id, skill_id, item_id,
                        difficulty, mastery_probability, reason, model_version,
                        policy_version, created_at
                    ) VALUES (
                        :recommendation_id, :course_id, :learner_id, :skill_id, :item_id,
                        :difficulty, :mastery_probability, :reason, :model_version,
                        :policy_version, :created_at
                    )
                    """
                ),
                {
                    "recommendation_id": event.recommendation_id,
                    "course_id": event.course_id,
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
        clauses = ["course_id = :course_id"]
        parameters: dict[str, str] = {"course_id": self.course_id}
        if learner_id is not None:
            clauses.append("learner_id = :learner_id")
            parameters["learner_id"] = learner_id
        where = f" WHERE {' AND '.join(clauses)}"
        with connection_scope(self._engine, write=False) as connection:
            rows = (
                connection.execute(
                    text(
                        "SELECT * FROM recommendation_events"
                        f"{where} ORDER BY created_at, recommendation_id"
                    ),
                    parameters,
                )
                .mappings()
                .all()
            )
        return [
            RecommendationEvent(
                recommendation_id=row["recommendation_id"],
                course_id=row["course_id"],
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
        self._require_own_course(gap.course_id, what="content gap")
        event = ContentGapEvent(
            **gap.model_dump(),
            content_gap_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
        )
        with connection_scope(self._engine, write=True) as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO content_gap_events (
                        content_gap_id, course_id, learner_id, completed_skill_id,
                        completed_skill_name, newly_unlocked_skill_id,
                        newly_unlocked_skill_name, missing_approved_content,
                        current_mastery_probability, prerequisite_mastery_threshold,
                        reason, model_version, policy_version, created_at
                    ) VALUES (
                        :content_gap_id, :course_id, :learner_id, :completed_skill_id,
                        :completed_skill_name, :newly_unlocked_skill_id,
                        :newly_unlocked_skill_name, :missing_approved_content,
                        :current_mastery_probability, :prerequisite_mastery_threshold,
                        :reason, :model_version, :policy_version, :created_at
                    )
                    """
                ),
                {
                    "content_gap_id": event.content_gap_id,
                    "course_id": event.course_id,
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
        clauses = ["course_id = :course_id"]
        parameters: dict[str, str] = {"course_id": self.course_id}
        if learner_id is not None:
            clauses.append("learner_id = :learner_id")
            parameters["learner_id"] = learner_id
        where = f" WHERE {' AND '.join(clauses)}"
        with connection_scope(self._engine, write=False) as connection:
            rows = (
                connection.execute(
                    text(
                        "SELECT * FROM content_gap_events"
                        f"{where} ORDER BY created_at, content_gap_id"
                    ),
                    parameters,
                )
                .mappings()
                .all()
            )
        return [
            ContentGapEvent(
                content_gap_id=row["content_gap_id"],
                course_id=row["course_id"],
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
