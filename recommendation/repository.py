from datetime import datetime, timezone
from typing import Protocol, Sequence
from uuid import uuid4

from api.bank import BankItem
from bkt.schemas import AttemptEvent, MasterySnapshot
from recommendation.schemas import RecommendationEvent, RecommendationResult
from taxonomy.schemas import SkillDefinition


class RecommendationRepository(Protocol):
    def list_skills(self) -> list[SkillDefinition]: ...

    def list_latest_mastery(self, learner_id: str) -> list[MasterySnapshot]: ...

    def list_approved_buildable_items(self) -> list[BankItem]: ...

    def list_attempts(self, learner_id: str) -> list[AttemptEvent]: ...

    def save_recommendation(
        self, recommendation: RecommendationResult
    ) -> RecommendationEvent: ...


class InMemoryRecommendationRepository:
    """Test/demo repository. Supplied items are declared approved and buildable."""

    def __init__(
        self,
        *,
        skills: Sequence[SkillDefinition],
        items: Sequence[BankItem],
        mastery: Sequence[MasterySnapshot] = (),
        attempts: Sequence[AttemptEvent] = (),
    ) -> None:
        self._skills = [skill.model_copy(deep=True) for skill in skills]
        self._items = [item.model_copy(deep=True) for item in items]
        self._mastery = [snapshot.model_copy(deep=True) for snapshot in mastery]
        self._attempts = [attempt.model_copy(deep=True) for attempt in attempts]
        self._recommendations: list[RecommendationEvent] = []

    def list_skills(self) -> list[SkillDefinition]:
        return [skill.model_copy(deep=True) for skill in self._skills]

    def list_latest_mastery(self, learner_id: str) -> list[MasterySnapshot]:
        latest: dict[str, MasterySnapshot] = {}
        for snapshot in self._mastery:
            if snapshot.learner_id != learner_id:
                continue
            current = latest.get(snapshot.skill_id)
            if current is None or (snapshot.updated_at, snapshot.source_attempt_id) > (
                current.updated_at,
                current.source_attempt_id,
            ):
                latest[snapshot.skill_id] = snapshot
        return [
            latest[skill_id].model_copy(deep=True) for skill_id in sorted(latest)
        ]

    def list_approved_buildable_items(self) -> list[BankItem]:
        return [item.model_copy(deep=True) for item in self._items]

    def list_attempts(self, learner_id: str) -> list[AttemptEvent]:
        attempts = [
            attempt.model_copy(deep=True)
            for attempt in self._attempts
            if attempt.learner_id == learner_id
        ]
        return sorted(
            attempts,
            key=lambda attempt: (
                attempt.attempt_order,
                attempt.occurred_at,
                attempt.attempt_id,
            ),
        )

    def save_recommendation(
        self, recommendation: RecommendationResult
    ) -> RecommendationEvent:
        event = RecommendationEvent(
            **recommendation.model_dump(),
            recommendation_id=str(uuid4()),
            created_at=datetime.now(timezone.utc),
        )
        self._recommendations.append(event.model_copy(deep=True))
        return event

    def list_recommendations(
        self, *, learner_id: str | None = None
    ) -> list[RecommendationEvent]:
        return [
            event.model_copy(deep=True)
            for event in self._recommendations
            if learner_id is None or event.learner_id == learner_id
        ]
