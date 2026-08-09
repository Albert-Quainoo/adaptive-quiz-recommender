from collections import defaultdict
from collections.abc import Callable, Sequence
from datetime import datetime, timezone

from api.bank import BankItem
from api.presentation import QuestionPresentation, score_response
from bkt.model import BKTModel
from bkt.repository import AttemptConflictError, BKTRepository
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

    def process_attempt(
        self,
        attempt: AttemptEvent,
        *,
        item: BankItem,
        presentation: QuestionPresentation,
    ) -> MasterySnapshot:
        attempt = self._score_attempt(attempt, item, presentation)
        existing = self.repository.get_attempt(attempt.attempt_id)
        if existing is not None:
            if existing != attempt:
                raise AttemptConflictError(
                    f"attempt_id {attempt.attempt_id!r} is already used by a different attempt"
                )
            snapshot = self.repository.get_mastery_for_attempt(attempt.attempt_id)
            if snapshot is None:
                raise RuntimeError("stored attempt is missing its mastery snapshot")
            return snapshot

        history = self.repository.list_attempts(
            learner_id=attempt.learner_id, skill_id=attempt.skill_id
        )
        ordered_history = sorted([*history, attempt], key=self._attempt_sort_key)
        probability = self.model.update_mastery(ordered_history)
        snapshot = self._snapshot(
            attempt, probability, attempt_count=len(ordered_history)
        )

        return self.repository.save_attempt_and_mastery(attempt, snapshot)

    def replay(self, *, learner_id: str | None = None) -> list[MasterySnapshot]:
        """Recompute stored snapshots from immutable attempts after a model change."""

        grouped: dict[tuple[str, str], list[AttemptEvent]] = defaultdict(list)
        for attempt in self.repository.list_attempts(learner_id=learner_id):
            grouped[(attempt.learner_id, attempt.skill_id)].append(attempt)

        snapshots: list[MasterySnapshot] = []
        for attempts in grouped.values():
            attempts.sort(key=self._attempt_sort_key)
            for end in range(1, len(attempts) + 1):
                attempt = attempts[end - 1]
                probability = self.model.update_mastery(attempts[:end])
                snapshot = self._snapshot(attempt, probability, attempt_count=end)
                self.repository.save_mastery(snapshot)
                snapshots.append(snapshot)
        return snapshots

    def _snapshot(
        self, attempt: AttemptEvent, probability: float, *, attempt_count: int
    ) -> MasterySnapshot:
        return MasterySnapshot(
            learner_id=attempt.learner_id,
            skill_id=attempt.skill_id,
            mastery_probability=probability,
            attempt_count=attempt_count,
            model_version=self.model.model_version,
            source_attempt_id=attempt.attempt_id,
            updated_at=self._clock(),
        )

    @staticmethod
    def _attempt_sort_key(attempt: AttemptEvent) -> tuple[int, datetime, str]:
        return attempt.attempt_order, attempt.occurred_at, attempt.attempt_id

    @staticmethod
    def _score_attempt(
        attempt: AttemptEvent,
        item: BankItem,
        presentation: QuestionPresentation,
    ) -> AttemptEvent:
        if item.item_id != attempt.item_id:
            raise ValueError("attempt item_id does not match the BankItem")
        if item.skill_id != attempt.skill_id:
            raise ValueError("attempt skill_id does not match the BankItem")
        if presentation.presentation_id != attempt.presentation_id:
            raise ValueError("attempt presentation_id does not match the presentation")
        correct = score_response(
            item,
            presentation,
            submitted_option_id=attempt.selected_option_id,
        )
        return attempt.model_copy(update={"correct": correct})
