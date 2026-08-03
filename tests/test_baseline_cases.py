from evaluation.baseline_cases import BASELINE_CASES


def test_case_id_uniqueness():
    case_ids = [case.case_id for case in BASELINE_CASES]
    
    assert len(case_ids) == len(set(case_ids))

def test_difficulty_levels():
    difficulties = {
    case.request.difficulty
    for case in BASELINE_CASES
}

    assert difficulties == {
        "introductory",
        "intermediate",
        "advanced",
    }

def test_include_question_count():
    question_counts = {
        case.request.question_count 
        for case in BASELINE_CASES
    }

    assert question_counts == {
        1,
        3,
        5,
    }

def test_positive_token_budget():
    assert all(
        case.max_new_tokens  == 1200
        for case in BASELINE_CASES
    )


def test_topic_learning_objective_not_empty():
    assert all (
        case.request.topic.strip()
        for case in BASELINE_CASES
    )

    assert all(
        case.request.learning_objective
        for case in BASELINE_CASES
    )


def test_nine_cases_present():
    assert len(BASELINE_CASES) == 9

