from datetime import datetime, timezone

import pandas as pd
import pytest

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


def attempt(attempt_id: str, correct: bool) -> AttemptEvent:
    return AttemptEvent(
        attempt_id=attempt_id,
        learner_id="learner",
        skill_id="skill",
        correct=correct,
        occurred_at=NOW,
    )


def test_model_delegates_fit_prediction_and_post_attempt_update_to_pybkt():
    engine = FakePyBKTModel()
    model = BKTModel(
        engine, model_version="v1", clock=lambda: NOW
    )
    attempts = [attempt("a-1", True), attempt("a-2", False)]

    metadata = model.fit(attempts)
    predictions = model.predict(attempts)
    mastery = model.update_mastery(attempts)

    assert metadata.training_attempt_count == 2
    assert metadata.skill_ids == ["skill"]
    pd.testing.assert_frame_equal(engine.fit_frame, engine.predict_frame.iloc[:2])
    assert predictions["state_predictions"].tolist() == [0.2, 0.4]
    assert engine.predict_frame.iloc[-1].to_dict() == {
        "order_id": 2,
        "user_id": "learner",
        "skill_name": "skill",
        "correct": -1,
    }
    assert mastery == 0.75


def test_model_rejects_prediction_before_fit():
    model = BKTModel(FakePyBKTModel())
    with pytest.raises(RuntimeError, match="must be fitted"):
        model.predict([attempt("a-1", True)])
