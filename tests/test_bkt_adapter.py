from bkt import AttemptEvent, PYBKT_COLUMNS, attempts_to_dataframe


def test_adapter_preserves_attempt_order_and_uses_pybkt_columns():
    attempts = [
        AttemptEvent(
            attempt_id="later",
            course_id="test-course",
            presentation_id="presentation-later",
            attempt_order=2,
            learner_id="learner",
            item_id="item",
            skill_id="skill",
            selected_option_id="option-1",
            correct=False,
        ),
        AttemptEvent(
            attempt_id="earlier",
            course_id="test-course",
            presentation_id="presentation-earlier",
            attempt_order=1,
            learner_id="learner",
            item_id="item",
            skill_id="skill",
            selected_option_id="option-1",
            correct=True,
        ),
    ]

    frame = attempts_to_dataframe(attempts)

    assert frame.columns.tolist() == PYBKT_COLUMNS
    assert frame.to_dict("records") == [
        {"order_id": 1, "user_id": "learner", "skill_name": "skill", "correct": 1},
        {"order_id": 2, "user_id": "learner", "skill_name": "skill", "correct": 0},
    ]
