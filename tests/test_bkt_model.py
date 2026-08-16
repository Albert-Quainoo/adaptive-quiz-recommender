from datetime import datetime, timezone

import pandas as pd
import pytest
from pyBKT.models import Model, Roster

from bkt import AttemptEvent, BKTModel


NOW = datetime(2026, 8, 6, tzinfo=timezone.utc)


class FakePyBKTModel:
    def __init__(self):
        self.fit_frame = None
        self.predict_frame = None

    def fit(self, *, data):
        self.fit_frame = data.copy()

    def predict(self, *, data):
        self.predict_frame = data.copy()
        result = data.copy()
        result["state_predictions"] = [0.2, 0.4, 0.75][: len(data)]
        return result


def attempt(
    attempt_id: str,
    correct: bool,
    *,
    attempt_order: int = 0,
    learner_id: str = "learner",
    skill_id: str = "skill",
) -> AttemptEvent:
    return AttemptEvent(
        attempt_id=attempt_id,
        course_id="test-course",
        presentation_id=f"presentation-{attempt_id}",
        attempt_order=attempt_order,
        learner_id=learner_id,
        item_id=f"item-{skill_id}",
        skill_id=skill_id,
        selected_option_id="option-1",
        correct=correct,
        occurred_at=NOW,
    )


def test_model_delegates_fit_and_prediction_to_pybkt():
    engine = FakePyBKTModel()
    model = BKTModel(
        engine, course_id="test-course", model_version="v1", clock=lambda: NOW
    )
    attempts = [
        attempt("a-1", True, attempt_order=1),
        attempt("a-2", False, attempt_order=2),
    ]

    metadata = model.fit(attempts)
    predictions = model.predict(attempts)

    assert metadata.training_attempt_count == 2
    assert metadata.skill_ids == ["skill"]
    pd.testing.assert_frame_equal(engine.fit_frame, engine.predict_frame)
    assert predictions["state_predictions"].tolist() == [0.2, 0.4]


def test_model_rejects_prediction_before_fit():
    model = BKTModel(FakePyBKTModel(), course_id="test-course")
    with pytest.raises(RuntimeError, match="must be fitted"):
        model.predict([attempt("a-1", True)])


@pytest.mark.parametrize(
    "attempts",
    [
        [attempt("a-1", True), attempt("a-2", False, learner_id="other")],
        [attempt("a-1", True), attempt("a-2", False, skill_id="other")],
    ],
)
def test_update_mastery_rejects_mixed_histories(attempts):
    model = BKTModel(FakePyBKTModel(), course_id="test-course")
    model.fit([attempt("training", True)])

    with pytest.raises(ValueError, match="exactly one learner and one skill"):
        model.update_mastery(attempts)


def test_update_mastery_matches_a_directly_constructed_pybkt_roster():
    training_attempts = [
        attempt(
            f"training-{index}",
            correct,
            attempt_order=index,
            learner_id=f"learner-{index // 4}",
        )
        for index, correct in enumerate(
            [False, False, True, True, False, True, True, True]
        )
    ]
    engine = Model(seed=42, num_fits=1)
    model = BKTModel(engine, course_id="test-course", model_version="v1")
    model.fit(training_attempts)
    history = [
        attempt("a-3", True, attempt_order=3),
        attempt("a-1", False, attempt_order=1),
        attempt("a-2", True, attempt_order=2),
    ]

    expected_roster = Roster(students=["learner"], skills="skill", model=engine)
    for response in [False, True, True]:
        expected_roster.update_state("skill", "learner", int(response))

    assert model.update_mastery(history) == pytest.approx(
        expected_roster.get_mastery_prob("skill", "learner")
    )
