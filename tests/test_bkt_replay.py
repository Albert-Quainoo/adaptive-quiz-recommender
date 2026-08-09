from bkt import AttemptEvent, BKTService, InMemoryBKTRepository


class ReplayModel:
    model_version = "refit-v2"

    def update_mastery(self, attempts):
        return sum(
            index * int(attempt.correct)
            for index, attempt in enumerate(attempts, start=1)
        ) / 10


def test_replay_rebuilds_each_snapshot_without_duplicating_attempts():
    repository = InMemoryBKTRepository()
    for index in range(2):
        repository.save_attempt(
            AttemptEvent(
                attempt_id=f"a-{index}",
                presentation_id=f"p-{index}",
                attempt_order=index,
                learner_id="learner",
                item_id="item",
                skill_id="skill",
                selected_option_id="option",
                correct=True,
            )
        )

    snapshots = BKTService(ReplayModel(), repository).replay()

    assert [snapshot.mastery_probability for snapshot in snapshots] == [0.1, 0.3]
    assert all(snapshot.model_version == "refit-v2" for snapshot in snapshots)
    assert len(repository.list_attempts()) == 2
    assert [snapshot.attempt_count for snapshot in snapshots] == [1, 2]
    assert len(repository.list_mastery()) == 2


def test_replay_preserves_existing_snapshot_history():
    repository = InMemoryBKTRepository()
    service = BKTService(ReplayModel(), repository)
    for index, correct in enumerate([True, False, True], start=1):
        repository.save_attempt(
            AttemptEvent(
                attempt_id=f"a-{index}",
                presentation_id=f"p-{index}",
                attempt_order=index,
                learner_id="learner",
                item_id="item",
                skill_id="skill",
                selected_option_id="option",
                correct=correct,
            )
        )

    first_replay = service.replay()
    second_replay = service.replay()

    assert [snapshot.mastery_probability for snapshot in second_replay] == [
        snapshot.mastery_probability for snapshot in first_replay
    ]
    assert len(repository.list_attempts()) == 3
    assert len(repository.list_mastery()) == 6
