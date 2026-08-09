from datetime import datetime, timedelta, timezone
from importlib.metadata import version

import pytest
from pyBKT.models import Model, Roster

from api.bank import BankItem
from api.presentation import present_bank_item
from api.schemas import QuizQuestion
from bkt import (
    AttemptConflictError,
    AttemptEvent,
    BKTModel,
    BKTService,
    InMemoryBKTRepository,
)


NOW = datetime(2026, 8, 7, tzinfo=timezone.utc)
SKILLS = ["skill-a", "skill-b"]


def training_event(
    *, learner_id: str, skill_id: str, attempt_order: int, correct: bool
) -> AttemptEvent:
    suffix = f"{learner_id}-{skill_id}-{attempt_order}"
    return AttemptEvent(
        attempt_id=f"training-{suffix}",
        presentation_id=f"training-presentation-{suffix}",
        learner_id=learner_id,
        item_id=f"item-{skill_id}",
        skill_id=skill_id,
        selected_option_id="training-option",
        correct=correct,
        attempt_order=attempt_order,
        occurred_at=NOW + timedelta(seconds=attempt_order),
    )


@pytest.fixture(scope="module")
def fitted_model():
    sequences = [
        [False, False, False, True, False, True, True, True, True, True],
        [False, False, True, False, True, True, True, True, True, True],
    ]
    attempts = [
        training_event(
            learner_id=f"training-learner-{learner_index}",
            skill_id=skill_id,
            attempt_order=attempt_order,
            correct=correct,
        )
        for skill_id in SKILLS
        for learner_index in range(8)
        for attempt_order, correct in enumerate(sequences[learner_index % 2], start=1)
    ]
    engine = Model(seed=42, num_fits=1, parallel=False)
    model = BKTModel(engine, model_version="synthetic-v1")
    model.fit(attempts)
    return model, engine


def bank_item(skill_id: str) -> BankItem:
    return BankItem(
        item_id=f"item-{skill_id}",
        skill_id=skill_id,
        provenance="hand_authored",
        question=QuizQuestion(
            question=f"Question for {skill_id}",
            options=["Correct", "Wrong A", "Wrong B", "Wrong C"],
            correct_answer="Correct",
            explanation="Synthetic deterministic item.",
            concept=skill_id,
            difficulty="introductory",
        ),
    )


def submission(
    *,
    attempt_id: str,
    learner_id: str,
    skill_id: str,
    attempt_order: int,
    correct: bool,
):
    item = bank_item(skill_id)
    presentation = present_bank_item(
        item, learner_id=learner_id, attempt_id=attempt_id
    )
    selected = next(
        option
        for option in presentation.presented_options
        if (option.value == item.question.correct_answer) == correct
    )
    event = AttemptEvent(
        attempt_id=attempt_id,
        presentation_id=presentation.presentation_id,
        learner_id=learner_id,
        item_id=item.item_id,
        skill_id=skill_id,
        selected_option_id=selected.option_id,
        correct=not correct,
        attempt_order=attempt_order,
        occurred_at=NOW + timedelta(minutes=attempt_order),
    )
    return event, item, presentation


def test_real_pybkt_fit_exposes_fitted_parameters(fitted_model):
    model, _ = fitted_model

    parameters = model.get_parameters()

    assert version("pyBKT") == "1.4.3"
    assert not parameters.empty
    assert set(parameters.index.get_level_values("skill")) == set(SKILLS)


def test_real_roster_updates_are_bounded_directional_and_isolated(fitted_model):
    _, engine = fitted_model
    roster = Roster(
        students=["correct-learner", "incorrect-learner"],
        skills=SKILLS,
        model=engine,
    )
    untouched_learner = roster.get_mastery_prob("skill-a", "incorrect-learner")
    untouched_skill = roster.get_mastery_prob("skill-b", "correct-learner")
    correct_probabilities = []

    for _ in range(4):
        roster.update_state("skill-a", "correct-learner", 1)
        correct_probabilities.append(
            roster.get_mastery_prob("skill-a", "correct-learner")
        )
    assert roster.get_mastery_prob(
        "skill-a", "incorrect-learner"
    ) == pytest.approx(untouched_learner)
    for _ in range(4):
        roster.update_state("skill-a", "incorrect-learner", 0)

    incorrect_probability = roster.get_mastery_prob(
        "skill-a", "incorrect-learner"
    )
    assert all(0.0 <= probability <= 1.0 for probability in correct_probabilities)
    assert 0.0 <= incorrect_probability <= 1.0
    assert correct_probabilities == sorted(correct_probabilities)
    assert correct_probabilities[-1] > incorrect_probability
    assert roster.get_mastery_prob("skill-b", "correct-learner") == pytest.approx(
        untouched_skill
    )
    assert untouched_learner == pytest.approx(
        Roster(
            students=["incorrect-learner"], skills="skill-a", model=engine
        ).get_mastery_prob("skill-a", "incorrect-learner")
    )


def test_real_model_replay_idempotency_conflicts_and_history(fitted_model):
    model, _ = fitted_model
    repository = InMemoryBKTRepository()
    service = BKTService(model, repository, clock=lambda: NOW)
    requests = [
        submission(
            attempt_id=f"online-{index}",
            learner_id="online-learner",
            skill_id="skill-a",
            attempt_order=index,
            correct=correct,
        )
        for index, correct in enumerate([False, True, True, True], start=1)
    ]

    online = [
        service.process_attempt(event, item=item, presentation=presentation)
        for event, item, presentation in requests
    ]
    duplicate = service.process_attempt(
        requests[-1][0], item=requests[-1][1], presentation=requests[-1][2]
    )

    assert duplicate == online[-1]
    assert len(repository.list_attempts()) == 4
    assert len(repository.list_mastery()) == 4
    assert all(0.0 <= snapshot.mastery_probability <= 1.0 for snapshot in online)
    assert online[-1].mastery_probability > online[0].mastery_probability

    conflicting = requests[-1][0].model_copy(update={"attempt_order": 99})
    with pytest.raises(AttemptConflictError, match="already used"):
        service.process_attempt(
            conflicting, item=requests[-1][1], presentation=requests[-1][2]
        )

    replayed = service.replay(learner_id="online-learner")
    assert [snapshot.mastery_probability for snapshot in replayed] == pytest.approx(
        [snapshot.mastery_probability for snapshot in online]
    )
    assert len(repository.list_mastery()) == 8
