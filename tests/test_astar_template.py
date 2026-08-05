import random

import pytest

from api.schemas import QuizQuestion
from taxonomy.schemas import SkillDefinition
from templates.astar import (
    AStarInstance,
    UnusableInstance,
    check_heuristics,
    distances_to_goal,
    expansion_order,
    generate,
    generate_astar_instance,
    solve_astar,
)

EDGE_COSTS = {
    ("S", "A"): 1,
    ("S", "B"): 4,
    ("A", "C"): 2,
    ("B", "C"): 1,
    ("B", "D"): 5,
    ("C", "G"): 3,
    ("D", "G"): 1,
}

HEURISTICS = {"S": 4, "A": 4, "B": 3, "C": 2, "D": 1, "G": 0}


def astar_skill() -> SkillDefinition:
    return SkillDefinition(
        skill_id="AI-SRC-10",
        topic="Search and Problem Solving",
        subtopic="Informed search",
        name="A-star Search",
        learning_objective="Trace A-star Search using (f(n)=g(n)+h(n)).",
        cognitive_process="apply",
        generation_strategy="templated",
        template_id="search.astar_trace",
    )


def test_solver_produces_the_hand_checked_trace():
    solution = solve_astar(AStarInstance(EDGE_COSTS, HEURISTICS))

    assert solution.expansion_order == ("S", "A", "C", "G")
    assert [(step.g, step.h, step.f) for step in solution.expansions] == [
        (0, 4, 4),
        (1, 4, 5),
        (3, 2, 5),
        (6, 0, 6),
    ]
    assert solution.path == ("S", "A", "C", "G")
    assert solution.path_cost == 6


def test_tied_frontier_is_rejected():
    tied = HEURISTICS | {"B": 1}

    with pytest.raises(UnusableInstance, match="tie on f"):
        solve_astar(AStarInstance(EDGE_COSTS, tied))


def test_inconsistent_heuristic_is_rejected():
    inconsistent = HEURISTICS | {"A": 10}

    with pytest.raises(UnusableInstance, match="inconsistent across"):
        check_heuristics(AStarInstance(EDGE_COSTS, inconsistent))


def test_goal_heuristic_must_be_zero():
    with pytest.raises(UnusableInstance, match="goal must have a heuristic of 0"):
        check_heuristics(AStarInstance(EDGE_COSTS, HEURISTICS | {"G": 1}))


@pytest.mark.parametrize("difficulty", ["introductory", "intermediate", "advanced"])
def test_generated_instances_are_admissible(difficulty):
    for seed in range(20):
        instance = generate_astar_instance(difficulty, random.Random(seed))
        distances = distances_to_goal(instance.edge_costs)

        check_heuristics(instance)

        assert all(cost >= 0 for cost in instance.edge_costs.values())
        assert all(
            instance.heuristics[node] <= distances[node] for node in instance.heuristics
        )


@pytest.mark.parametrize("difficulty", ["introductory", "intermediate", "advanced"])
def test_solvable_instances_run_from_start_to_goal(difficulty):
    solved = 0

    for seed in range(20):
        instance = generate_astar_instance(difficulty, random.Random(seed))

        try:
            solution = solve_astar(instance)
        except UnusableInstance:
            # A tied frontier is rejected by design; generate() reseeds past it.
            continue

        solved += 1
        assert solution.expansion_order[0] == "S"
        assert solution.expansion_order[-1] == "G"

    assert solved > 0


def test_the_same_seed_produces_an_identical_question():
    first = generate(astar_skill(), "intermediate", seed=99)
    second = generate(astar_skill(), "intermediate", seed=99)

    assert first.model_dump() == second.model_dump()


def test_different_seeds_produce_different_questions():
    questions = {
        generate(astar_skill(), "intermediate", seed=seed).question
        for seed in range(10)
    }

    assert len(questions) > 1


def test_a_fixed_seed_produces_the_expected_answer():
    # Seed 7 settles on S-A=8, S-B=2, A-C=3, A-D=8, B-D=7, B-E=9, C-F=5,
    # D-F=3, E-F=7, E-G=9, F-G=5 with h = S:14, A:10, B:12, C:9, D:7, E:8,
    # F:4, G:0. Worked by hand, the frontier expands S(f=14), B(f=14),
    # D(f=16), F(f=16), G(f=17), reaching the goal by S -> B -> D -> F -> G
    # at a cost of 17.
    question = generate(astar_skill(), "intermediate", seed=7)

    assert question.correct_answer == "S -> B -> D -> F -> G"
    assert "S-A = 8" in question.question
    assert "h(B) = 12" in question.question


@pytest.mark.parametrize("seed", range(15))
def test_options_are_distinct_and_contain_the_answer(seed):
    question = generate(astar_skill(), "intermediate", seed=seed)

    assert len(question.options) == 4
    assert len(set(question.options)) == 4
    assert question.correct_answer in question.options


@pytest.mark.parametrize("seed", range(5))
def test_generated_question_revalidates(seed):
    question = generate(astar_skill(), "advanced", seed=seed)

    assert QuizQuestion.model_validate(question.model_dump()) == question


def test_question_carries_the_skill_name_and_requested_difficulty():
    question = generate(astar_skill(), "introductory", seed=3)

    assert question.concept == "A-star Search"
    assert question.difficulty == "introductory"


def test_a_missing_seed_still_produces_a_valid_question():
    question = generate(astar_skill(), "intermediate")

    assert question.correct_answer in question.options


def test_the_explanation_reports_the_solver_values():
    question = generate(astar_skill(), "intermediate", seed=7)

    assert "S (g=0, h=14, f=14)" in question.explanation
    assert "D (g=9, h=7, f=16)" in question.explanation
    assert "S -> B -> D -> F -> G at a path cost of 17" in question.explanation


@pytest.mark.parametrize(
    "difficulty, expected_shapes",
    [
        ("introductory", {5, 6}),
        ("intermediate", {7, 8}),
        ("advanced", {9, 10}),
    ],
)
def test_graphs_grow_with_difficulty(difficulty, expected_shapes):
    shapes = {
        len(generate_astar_instance(difficulty, random.Random(seed)).nodes)
        for seed in range(20)
    }

    assert shapes == expected_shapes


@pytest.mark.parametrize("seed", range(15))
def test_distractors_are_the_orders_a_confused_student_would_pick(seed):
    instance = generate_astar_instance("intermediate", random.Random(seed))

    greedy = expansion_order(instance, "heuristic")
    uniform_cost = expansion_order(instance, "cost")

    assert greedy[0] == "S" and greedy[-1] == "G"
    assert uniform_cost[0] == "S" and uniform_cost[-1] == "G"
