from typing import Protocol

from bkt.schemas import AttemptEvent, BKTModelMetadata, MasterySnapshot


class AttemptConflictError(ValueError):
    """An attempt identifier was reused for a different immutable event."""


class AttemptRepository(Protocol):
    def save_attempt(self, attempt: AttemptEvent) -> None: ...

    def get_attempt(self, attempt_id: str) -> AttemptEvent | None: ...

    def list_attempts(
        self, *, learner_id: str | None = None, skill_id: str | None = None
    ) -> list[AttemptEvent]: ...


class MasteryRepository(Protocol):
    def save_mastery(self, snapshot: MasterySnapshot) -> None: ...

    def get_mastery(self, learner_id: str, skill_id: str) -> MasterySnapshot | None: ...

    def get_mastery_for_attempt(self, attempt_id: str) -> MasterySnapshot | None: ...

    def list_mastery(
        self, *, learner_id: str | None = None
    ) -> list[MasterySnapshot]: ...


class ModelMetadataRepository(Protocol):
    def save_model_metadata(self, metadata: BKTModelMetadata) -> None: ...

    def get_model_metadata(self, model_version: str) -> BKTModelMetadata | None: ...


class BKTRepository(
    AttemptRepository, MasteryRepository, ModelMetadataRepository, Protocol
):
    def save_attempt_and_mastery(
        self, attempt: AttemptEvent, snapshot: MasterySnapshot
    ) -> MasterySnapshot: ...


class InMemoryBKTRepository:
    """Deterministic repository used by the pure integration and its tests."""

    def __init__(self) -> None:
        self._attempts: dict[str, AttemptEvent] = {}
        self._mastery: list[MasterySnapshot] = []
        self._metadata: dict[str, BKTModelMetadata] = {}

    def save_attempt(self, attempt: AttemptEvent) -> None:
        existing = self._attempts.get(attempt.attempt_id)
        if existing is not None:
            if existing == attempt:
                return
            raise AttemptConflictError(
                f"attempt_id {attempt.attempt_id!r} is already used by a different attempt"
            )
        self._attempts[attempt.attempt_id] = attempt.model_copy(deep=True)

    def get_attempt(self, attempt_id: str) -> AttemptEvent | None:
        attempt = self._attempts.get(attempt_id)
        return attempt.model_copy(deep=True) if attempt else None

    def list_attempts(
        self, *, learner_id: str | None = None, skill_id: str | None = None
    ) -> list[AttemptEvent]:
        attempts = [
            attempt.model_copy(deep=True)
            for attempt in self._attempts.values()
            if (learner_id is None or attempt.learner_id == learner_id)
            and (skill_id is None or attempt.skill_id == skill_id)
        ]
        return sorted(
            attempts,
            key=lambda attempt: (
                attempt.attempt_order,
                attempt.occurred_at,
                attempt.attempt_id,
            ),
        )

    def save_mastery(self, snapshot: MasterySnapshot) -> None:
        self._mastery.append(snapshot.model_copy(deep=True))

    def save_attempt_and_mastery(
        self, attempt: AttemptEvent, snapshot: MasterySnapshot
    ) -> MasterySnapshot:
        if snapshot.source_attempt_id != attempt.attempt_id:
            raise ValueError("snapshot source_attempt_id must match the attempt")
        existing = self.get_attempt(attempt.attempt_id)
        if existing is not None:
            if existing != attempt:
                raise AttemptConflictError(
                    f"attempt_id {attempt.attempt_id!r} is already used by a different attempt"
                )
            stored = self.get_mastery_for_attempt(attempt.attempt_id)
            if stored is None:
                raise RuntimeError("stored attempt is missing its mastery snapshot")
            return stored

        self.save_attempt(attempt)
        self.save_mastery(snapshot)
        return snapshot.model_copy(deep=True)

    def get_mastery(self, learner_id: str, skill_id: str) -> MasterySnapshot | None:
        for snapshot in reversed(self._mastery):
            if snapshot.learner_id == learner_id and snapshot.skill_id == skill_id:
                return snapshot.model_copy(deep=True)
        return None

    def get_mastery_for_attempt(self, attempt_id: str) -> MasterySnapshot | None:
        for snapshot in self._mastery:
            if snapshot.source_attempt_id == attempt_id:
                return snapshot.model_copy(deep=True)
        return None

    def list_mastery(
        self, *, learner_id: str | None = None
    ) -> list[MasterySnapshot]:
        return [
            snapshot.model_copy(deep=True)
            for snapshot in self._mastery
            if learner_id is None or snapshot.learner_id == learner_id
        ]

    def save_model_metadata(self, metadata: BKTModelMetadata) -> None:
        self._metadata[metadata.model_version] = metadata.model_copy(deep=True)

    def get_model_metadata(self, model_version: str) -> BKTModelMetadata | None:
        metadata = self._metadata.get(model_version)
        return metadata.model_copy(deep=True) if metadata else None
