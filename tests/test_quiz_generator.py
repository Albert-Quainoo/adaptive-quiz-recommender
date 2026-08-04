from api.quiz_generator import token_budget


def test_token_budget_scales_with_question_count():
    assert token_budget(5) > token_budget(3) > token_budget(1)


def test_token_budget_covers_longest_observed_response():
    # BASE_SEARCH_003 (5 questions) truncated at 1200 tokens under prompt v2.
    assert token_budget(5) > 1200
