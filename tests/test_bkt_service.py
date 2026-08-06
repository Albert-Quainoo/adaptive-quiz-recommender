from datetime import datetime, timezone

from bkt import AttemptEvent, BKTService, InMemoryBKTRepository


NOW = datetime(2026, 8, 6, tzinfo=timezone.utc)


class FakeBKTModel:
    model_version = "v1"

    def __init__(self):
        self.histories = []

    def update_mastery(self, attempts):
        self.histories.append(list(attempts))
        return len(attempts) / 10


def test_service_updates_from_ordered_history_then_persists_attempt_and_snapshot():
    repository = InMemoryBKTRepository()
    model = FakeBKTModel()
    service = BKTService(model, repository, clock=lambda: NOW)

    first = service.process_attempt(
        AttemptEvent(attempt_id="a-1", learner_id="one", skill_id="skill", correct=True)
    )
    second = service.process_attempt(
        AttemptEvent(attempt_id="a-2", learner_id="one", skill_id="skill", correct=False)
    )

    assert [attempt.attempt_id for attempt in model.histories[-1]] == ["a-1", "a-2"]
    assert first.mastery_probability == 0.1
    assert second.mastery_probability == 0.2
    assert repository.get_mastery("one", "skill") == second
