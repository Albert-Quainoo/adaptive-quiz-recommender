from typing import Protocol

from bkt.schemas import AttemptEvent, BKTModelMetadata, MasterySnapshot


class AttemptRepository(Protocol):
    def save_attempt(self, attempt: AttemptEvent) -> None: ...

    def list_attempts(
        self, *, learner_id: str | None = None, skill_id: str | None = None
    ) -> list[AttemptEvent]: ...


class MasteryRepository(Protocol):
    def save_mastery(self, snapshot: MasterySnapshot) -> None: ...

    def get_mastery(self, learner_id: str, skill_id: str) -> MasterySnapshot | None: ...

    def list_mastery(
        self, *, learner_id: str | None = None
    ) -> list[MasterySnapshot]: ...


class ModelMetadataRepository(Protocol):
    def save_model_metadata(self, metadata: BKTModelMetadata) -> None: ...

    def get_model_metadata(self, model_version: str) -> BKTModelMetadata | None: ...


class BKTRepository(
    AttemptRepository, MasteryRepository, ModelMetadataRepository, Protocol
):
    pass


class InMemoryBKTRepository:
    """Deterministic repository used by the pure integration and its tests."""

    def __init__(self) -> None:
        self._attempts: dict[str, AttemptEvent] = {}
        self._attempt_order: list[str] = []
        self._mastery: dict[str, MasterySnapshot] = {}
        self._mastery_order: list[str] = []
        self._metadata: dict[str, BKTModelMetadata] = {}

    def save_attempt(self, attempt: AttemptEvent) -> None:
        if attempt.attempt_id in self._attempts:
            raise ValueError(f"attempt {attempt.attempt_id!r} already exists")
        self._attempts[attempt.attempt_id] = attempt.model_copy(deep=True)
        self._attempt_order.append(attempt.attempt_id)

    def list_attempts(
        self, *, learner_id: str | None = None, skill_id: str | None = None
    ) -> list[AttemptEvent]:
        attempts = (self._attempts[key] for key in self._attempt_order)
        return [
            attempt.model_copy(deep=True)
            for attempt in attempts
            if (learner_id is None or attempt.learner_id == learner_id)
            and (skill_id is None or attempt.skill_id == skill_id)
        ]

    def save_mastery(self, snapshot: MasterySnapshot) -> None:
        key = snapshot.source_attempt_id
        if key not in self._mastery:
            self._mastery_order.append(key)
        self._mastery[key] = snapshot.model_copy(deep=True)

    def get_mastery(self, learner_id: str, skill_id: str) -> MasterySnapshot | None:
        for key in reversed(self._mastery_order):
            snapshot = self._mastery[key]
            if snapshot.learner_id == learner_id and snapshot.skill_id == skill_id:
                return snapshot.model_copy(deep=True)
        return None

    def list_mastery(
        self, *, learner_id: str | None = None
    ) -> list[MasterySnapshot]:
        return [
            self._mastery[key].model_copy(deep=True)
            for key in self._mastery_order
            if learner_id is None or self._mastery[key].learner_id == learner_id
        ]

    def save_model_metadata(self, metadata: BKTModelMetadata) -> None:
        self._metadata[metadata.model_version] = metadata.model_copy(deep=True)

    def get_model_metadata(self, model_version: str) -> BKTModelMetadata | None:
        metadata = self._metadata.get(model_version)
        return metadata.model_copy(deep=True) if metadata else None
