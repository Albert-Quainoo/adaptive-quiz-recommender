import random

import pytest

from api.bank import BankItem
from api.presentation import (
    derive_presentation_seed,
    present_bank_item,
    presentation_from_seed,
    score_response,
)
from api.schemas import QuizQuestion


def bank_item() -> BankItem:
    return BankItem(
        item_id="AI-SRC-08-stable-item",
        skill_id="AI-SRC-08",
        provenance="generated",
        question=QuizQuestion(
            question="Which option is supported?",
            options=["Correct", "Distractor A", "Distractor B", "Distractor C"],
            correct_answer="Correct",
            explanation="Correct is supported.",
            concept="Presentation",
            difficulty="intermediate",
        ),
    )


def answer_index(item: BankItem, seed: int) -> int:
    values = [option.value for option in presentation_from_seed(item, seed).presented_options]
    return values.index(item.question.correct_answer)


def test_same_presentation_seed_reproduces_same_order_and_identity():
    item = bank_item()
    first = presentation_from_seed(item, 4815162342)
    second = presentation_from_seed(item, 4815162342)

    assert first == second
    assert first.presentation_id == second.presentation_id
    assert [option.value for option in first.presented_options] == [
        option.value for option in second.presented_options
    ]


def test_stable_inputs_derive_the_same_seed_and_attempt_changes_it():
    first = derive_presentation_seed("item", "learner", "attempt-1")
    assert first == derive_presentation_seed("item", "learner", "attempt-1")
    assert first != derive_presentation_seed("item", "learner", "attempt-2")


def test_different_seeds_move_the_answer_between_positions():
    item = bank_item()
    positions = {answer_index(item, seed) for seed in range(32)}

    assert positions == {0, 1, 2, 3}


def test_correct_answer_remains_present_exactly_once():
    item = bank_item()
    for seed in range(32):
        values = [
            option.value
            for option in presentation_from_seed(item, seed).presented_options
        ]
        assert values.count(item.question.correct_answer) == 1


def test_scoring_by_value_and_stable_option_id_after_shuffle():
    item = bank_item()
    presentation = present_bank_item(
        item,
        learner_id="learner-17",
        attempt_id="attempt-3",
    )
    correct_option = next(
        option
        for option in presentation.presented_options
        if option.value == item.question.correct_answer
    )
    wrong_option = next(
        option
        for option in presentation.presented_options
        if option.value != item.question.correct_answer
    )

    assert score_response(item, presentation, submitted_value=correct_option.value)
    assert score_response(item, presentation, submitted_option_id=correct_option.option_id)
    assert not score_response(item, presentation, submitted_value=wrong_option.value)
    assert not score_response(item, presentation, submitted_option_id=wrong_option.option_id)


def test_presentation_does_not_mutate_canonical_bank_item_or_global_rng():
    item = bank_item()
    before = item.model_dump()
    global_state = random.getstate()

    presentation_from_seed(item, 99)

    assert item.model_dump() == before
    assert random.getstate() == global_state


def test_correct_answer_is_not_always_presented_at_index_zero():
    item = bank_item()
    positions = [answer_index(item, seed) for seed in range(100)]

    assert any(position != 0 for position in positions)
    assert len(set(positions)) == 4


def test_tampered_presentation_cannot_be_scored():
    item = bank_item()
    presentation = presentation_from_seed(item, 5)
    presentation.presented_options.reverse()

    with pytest.raises(ValueError, match="does not match its seed"):
        score_response(item, presentation, submitted_value="Correct")
