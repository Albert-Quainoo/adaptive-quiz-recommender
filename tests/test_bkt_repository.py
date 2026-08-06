from bkt import AttemptEvent, InMemoryBKTRepository, MasterySnapshot


def test_in_memory_repository_filters_attempts_and_returns_latest_mastery():
    repository = InMemoryBKTRepository()
    repository.save_attempt(
        AttemptEvent(attempt_id="a-1", learner_id="one", skill_id="skill", correct=True)
    )
    repository.save_attempt(
        AttemptEvent(attempt_id="a-2", learner_id="two", skill_id="skill", correct=False)
    )
    repository.save_mastery(
        MasterySnapshot(
            learner_id="one",
            skill_id="skill",
            mastery_probability=0.3,
            model_version="v1",
            source_attempt_id="a-1",
        )
    )
    repository.save_mastery(
        MasterySnapshot(
            learner_id="one",
            skill_id="skill",
            mastery_probability=0.7,
            model_version="v2",
            source_attempt_id="a-3",
        )
    )

    assert [item.attempt_id for item in repository.list_attempts(learner_id="one")] == [
        "a-1"
    ]
    assert repository.get_mastery("one", "skill").mastery_probability == 0.7
