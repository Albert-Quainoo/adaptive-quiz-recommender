from bkt import AttemptEvent, BKTService, InMemoryBKTRepository


class ReplayModel:
    model_version = "refit-v2"

    def update_mastery(self, attempts):
        return 0.2 * len(attempts)


def test_replay_rebuilds_each_snapshot_without_duplicating_attempts():
    repository = InMemoryBKTRepository()
    for index in range(2):
        repository.save_attempt(
            AttemptEvent(
                attempt_id=f"a-{index}",
                learner_id="learner",
                skill_id="skill",
                correct=True,
            )
        )

    snapshots = BKTService(ReplayModel(), repository).replay()

    assert [snapshot.mastery_probability for snapshot in snapshots] == [0.2, 0.4]
    assert all(snapshot.model_version == "refit-v2" for snapshot in snapshots)
    assert len(repository.list_attempts()) == 2
    assert len(repository.list_mastery()) == 2
