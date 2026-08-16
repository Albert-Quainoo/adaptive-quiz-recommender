from datetime import datetime, timezone

import pytest

from api.bank import BankItem
from api.presentation import present_bank_item
from api.schemas import QuizQuestion
from bkt import (
    AttemptConflictError,
    AttemptEvent,
    BKTService,
    InMemoryBKTRepository,
)


NOW = datetime(2026, 8, 6, tzinfo=timezone.utc)


class FakeBKTModel:
    model_version = "v1"

    def __init__(self):
        self.histories = []

    def update_mastery(self, attempts):
        self.histories.append(list(attempts))
        return len(attempts) / 10


def bank_item() -> BankItem:
    return BankItem(
        item_id="item-1",
        skill_id="skill",
        provenance="hand_authored",
        question=QuizQuestion(
            question="Which answer is correct?",
            options=["Correct", "Wrong A", "Wrong B", "Wrong C"],
            correct_answer="Correct",
            explanation="Correct is correct.",
            concept="Testing",
            difficulty="introductory",
        ),
    )


def submitted_attempt(
    attempt_id: str,
    attempt_order: int,
    *,
    select_correct: bool,
    claimed_correct: bool,
):
    item = bank_item()
    presentation = present_bank_item(
        item, learner_id="one", attempt_id=attempt_id
    )
    selected = next(
        option
        for option in presentation.presented_options
        if (option.value == item.question.correct_answer) == select_correct
    )
    attempt = AttemptEvent(
        attempt_id=attempt_id,
        course_id="test-course",
        presentation_id=presentation.presentation_id,
        learner_id="one",
        item_id="item-1",
        skill_id="skill",
        selected_option_id=selected.option_id,
        correct=claimed_correct,
        attempt_order=attempt_order,
        occurred_at=NOW,
    )
    return attempt, item, presentation


def test_service_updates_from_ordered_history_then_persists_attempt_and_snapshot():
    repository = InMemoryBKTRepository()
    model = FakeBKTModel()
    service = BKTService(model, repository, clock=lambda: NOW)

    first_attempt, item, first_presentation = submitted_attempt(
        "a-1", 1, select_correct=True, claimed_correct=False
    )
    second_attempt, _, second_presentation = submitted_attempt(
        "a-2", 2, select_correct=False, claimed_correct=True
    )
    first = service.process_attempt(
        first_attempt, item=item, presentation=first_presentation
    )
    second = service.process_attempt(
        second_attempt, item=item, presentation=second_presentation
    )

    assert [attempt.attempt_id for attempt in model.histories[-1]] == ["a-1", "a-2"]
    assert [attempt.correct for attempt in model.histories[-1]] == [True, False]
    assert first.mastery_probability == 0.1
    assert second.mastery_probability == 0.2
    assert second.attempt_count == 2
    assert repository.get_mastery("one", "skill") == second


def test_identical_attempt_is_idempotent_and_conflicting_attempt_is_rejected():
    repository = InMemoryBKTRepository()
    model = FakeBKTModel()
    service = BKTService(model, repository, clock=lambda: NOW)
    attempt, item, presentation = submitted_attempt(
        "a-1", 1, select_correct=True, claimed_correct=False
    )

    first = service.process_attempt(attempt, item=item, presentation=presentation)
    duplicate = service.process_attempt(attempt, item=item, presentation=presentation)

    assert duplicate == first
    assert len(model.histories) == 1
    assert len(repository.list_attempts()) == 1
    assert len(repository.list_mastery()) == 1

    conflicting = attempt.model_copy(update={"attempt_order": 2})
    with pytest.raises(AttemptConflictError, match="already used"):
        service.process_attempt(conflicting, item=item, presentation=presentation)

    assert len(model.histories) == 1
    assert len(repository.list_mastery()) == 1
