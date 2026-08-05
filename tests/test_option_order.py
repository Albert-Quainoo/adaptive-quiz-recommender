"""Every template must place the answer unpredictably.

Templates build the correct answer first and the distractors after it, so
without a shuffle every templated item would be answerable by picking option
one. That is worth its own test file: the damage is not just an easy quiz, it
is response data in which position predicts correctness, which is exactly the
signal knowledge tracing is meant to read as understanding.
"""

import pytest

from taxonomy.loader import course_paths, load_skills
from templates.registry import generate_templated_question

SEEDS = range(24)

TEMPLATED_SKILLS = [
    skill
    for skill in load_skills(*course_paths("ai")).skills
    if skill.generation_strategy == "templated"
]

IDS = [skill.template_id for skill in TEMPLATED_SKILLS]


def answer_index(skill, seed) -> int:
    question = generate_templated_question(skill, "intermediate", seed=seed)

    return question.options.index(question.correct_answer)


@pytest.mark.parametrize("skill", TEMPLATED_SKILLS, ids=IDS)
def test_every_template_is_covered(skill):
    assert skill.template_id in {
        "search.bfs_trace",
        "search.dfs_trace",
        "search.ucs_trace",
        "search.greedy_trace",
        "search.astar_trace",
        "nn.forward_trace",
    }


@pytest.mark.parametrize("skill", TEMPLATED_SKILLS, ids=IDS)
def test_the_same_seed_produces_the_same_option_order(skill):
    first = generate_templated_question(skill, "intermediate", seed=5)
    second = generate_templated_question(skill, "intermediate", seed=5)

    assert first.options == second.options
    assert first == second


@pytest.mark.parametrize("skill", TEMPLATED_SKILLS, ids=IDS)
def test_the_correct_answer_survives_the_shuffle(skill):
    for seed in SEEDS:
        question = generate_templated_question(skill, "intermediate", seed=seed)

        assert question.correct_answer in question.options
        assert len(set(question.options)) == 4


@pytest.mark.parametrize("skill", TEMPLATED_SKILLS, ids=IDS)
def test_the_answer_moves_between_positions(skill):
    positions = {answer_index(skill, seed) for seed in SEEDS}

    assert len(positions) > 1


@pytest.mark.parametrize("skill", TEMPLATED_SKILLS, ids=IDS)
def test_the_answer_is_not_always_first(skill):
    positions = [answer_index(skill, seed) for seed in SEEDS]

    assert any(position != 0 for position in positions)


@pytest.mark.parametrize("skill", TEMPLATED_SKILLS, ids=IDS)
def test_every_position_is_reachable(skill):
    positions = {answer_index(skill, seed) for seed in SEEDS}

    assert positions == {0, 1, 2, 3}
