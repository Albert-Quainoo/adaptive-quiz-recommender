from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from bkt import (
    AttemptConflictError,
    AttemptEvent,
    InMemoryBKTRepository,
    MasterySnapshot,
)


NOW = datetime(2026, 8, 7, tzinfo=timezone.utc)


def attempt(attempt_id: str, *, order: int, occurred_at: datetime) -> AttemptEvent:
    return AttemptEvent(
        attempt_id=attempt_id,
        course_id="test-course",
        presentation_id=f"p-{attempt_id}",
        attempt_order=order,
        learner_id="one",
        item_id="item",
        skill_id="skill",
        selected_option_id="option",
        correct=True,
        occurred_at=occurred_at,
    )


def test_in_memory_repository_filters_attempts_and_returns_latest_mastery():
    repository = InMemoryBKTRepository()
    repository.save_attempt(
        AttemptEvent(
            attempt_id="a-1",
        course_id="test-course",
            presentation_id="p-1",
            attempt_order=2,
            learner_id="one",
            item_id="item",
            skill_id="skill",
            selected_option_id="option",
            correct=True,
            occurred_at=NOW,
        )
    )
    repository.save_attempt(
        AttemptEvent(
            attempt_id="a-2",
        course_id="test-course",
            presentation_id="p-2",
            attempt_order=1,
            learner_id="two",
            item_id="item",
            skill_id="skill",
            selected_option_id="option",
            correct=False,
            occurred_at=NOW,
        )
    )
    repository.save_attempt(
        AttemptEvent(
            attempt_id="a-3",
        course_id="test-course",
            presentation_id="p-3",
            attempt_order=1,
            learner_id="one",
            item_id="item",
            skill_id="skill",
            selected_option_id="option",
            correct=False,
            occurred_at=NOW - timedelta(seconds=1),
        )
    )
    repository.save_mastery(
        MasterySnapshot(
            learner_id="one",
            course_id="test-course",
            skill_id="skill",
            mastery_probability=0.3,
            attempt_count=1,
            model_version="v1",
            source_attempt_id="a-1",
        )
    )
    repository.save_mastery(
        MasterySnapshot(
            learner_id="one",
            course_id="test-course",
            skill_id="skill",
            mastery_probability=0.7,
            attempt_count=2,
            model_version="v2",
            source_attempt_id="a-3",
        )
    )

    assert [item.attempt_id for item in repository.list_attempts(learner_id="one")] == [
        "a-3",
        "a-1",
    ]
    assert repository.get_mastery("one", "skill").mastery_probability == 0.7


def test_attempt_order_uses_timestamp_and_id_as_tie_breakers():
    repository = InMemoryBKTRepository()
    for event in [
        attempt("z", order=1, occurred_at=NOW),
        attempt("b", order=1, occurred_at=NOW - timedelta(seconds=1)),
        attempt("a", order=1, occurred_at=NOW - timedelta(seconds=1)),
    ]:
        repository.save_attempt(event)

    assert [event.attempt_id for event in repository.list_attempts()] == [
        "a",
        "b",
        "z",
    ]


def test_attempts_are_immutable_and_repository_rejects_conflicting_ids():
    repository = InMemoryBKTRepository()
    event = attempt("a-1", order=1, occurred_at=NOW)
    repository.save_attempt(event)
    repository.save_attempt(event)

    with pytest.raises(ValidationError):
        event.correct = False
    with pytest.raises(AttemptConflictError, match="already used"):
        repository.save_attempt(event.model_copy(update={"attempt_order": 2}))

    assert repository.list_attempts() == [event]


def test_mastery_snapshots_are_appended_as_history():
    repository = InMemoryBKTRepository()
    snapshots = [
        MasterySnapshot(
            learner_id="one",
            course_id="test-course",
            skill_id="skill",
            mastery_probability=probability,
            attempt_count=index,
            model_version=f"v{index}",
            source_attempt_id="a-1",
            updated_at=NOW + timedelta(seconds=index),
        )
        for index, probability in [(1, 0.3), (2, 0.7)]
    ]

    for snapshot in snapshots:
        repository.save_mastery(snapshot)

    assert repository.list_mastery() == snapshots
    assert repository.get_mastery("one", "skill") == snapshots[-1]
