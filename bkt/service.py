from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime, timezone

from bkt.model import BKTModel
from bkt.repository import BKTRepository
from bkt.schemas import AttemptEvent, BKTModelMetadata, MasterySnapshot


class BKTService:
    """Coordinates scoring, pyBKT state updates, and persistence."""

    def __init__(
        self,
        model: BKTModel,
        repository: BKTRepository,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.model = model
        self.repository = repository
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def fit(self, attempts: Sequence[AttemptEvent]) -> BKTModelMetadata:
        metadata = self.model.fit(attempts)
        self.repository.save_model_metadata(metadata)
        return metadata

    def process_attempt(self, attempt: AttemptEvent) -> MasterySnapshot:
        history = self.repository.list_attempts(
            learner_id=attempt.learner_id, skill_id=attempt.skill_id
        )
        probability = self.model.update_mastery([*history, attempt])
        snapshot = self._snapshot(attempt, probability)

        self.repository.save_attempt(attempt)
        self.repository.save_mastery(snapshot)
        return snapshot

    def replay(self, *, learner_id: str | None = None) -> list[MasterySnapshot]:
        """Recompute stored snapshots from immutable attempts after a model change."""

        grouped: dict[tuple[str, str], list[AttemptEvent]] = defaultdict(list)
        for attempt in self.repository.list_attempts(learner_id=learner_id):
            grouped[(attempt.learner_id, attempt.skill_id)].append(attempt)

        snapshots: list[MasterySnapshot] = []
        for attempts in grouped.values():
            for end in range(1, len(attempts) + 1):
                attempt = attempts[end - 1]
                probability = self.model.update_mastery(attempts[:end])
                snapshot = self._snapshot(attempt, probability)
                self.repository.save_mastery(snapshot)
                snapshots.append(snapshot)
        return snapshots

    def _snapshot(
        self, attempt: AttemptEvent, probability: float
    ) -> MasterySnapshot:
        return MasterySnapshot(
            learner_id=attempt.learner_id,
            skill_id=attempt.skill_id,
            mastery_probability=probability,
            model_version=self.model.model_version,
            source_attempt_id=attempt.attempt_id,
            updated_at=self._clock(),
        )
