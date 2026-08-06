from bkt import AttemptEvent, PYBKT_COLUMNS, attempts_to_dataframe


def test_adapter_preserves_attempt_order_and_uses_pybkt_columns():
    attempts = [
        AttemptEvent(
            attempt_id="later", learner_id="learner", skill_id="skill", correct=False
        ),
        AttemptEvent(
            attempt_id="earlier", learner_id="learner", skill_id="skill", correct=True
        ),
    ]

    frame = attempts_to_dataframe(attempts)

    assert frame.columns.tolist() == PYBKT_COLUMNS
    assert frame.to_dict("records") == [
        {"order_id": 0, "user_id": "learner", "skill_name": "skill", "correct": 0},
        {"order_id": 1, "user_id": "learner", "skill_name": "skill", "correct": 1},
    ]
