import pytest

from api.schemas import QuizQuestion
from taxonomy.schemas import SkillDefinition
from templates.graphs import (
    GraphInstance,
    UnusableInstance,
    solve,
    uninformed_order,
)
from templates.registry import generate_templated_question
from templates.search_traces import BFS, DFS, GREEDY, UCS, build_trace_question

# The same hand-checked graph the A-star tests use.
#
#   S-A=1  S-B=4  A-C=2  B-C=1  B-D=5  C-G=3  D-G=1
#   h: S=4  A=4  B=3  C=2  D=1  G=0
#
# Adjacency in alphabetical order:
#   S: A B    A: C S    B: C D S    C: A B G    D: B G    G: C D
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

FIXTURE = GraphInstance(EDGE_COSTS, HEURISTICS)

SPECS = (BFS, DFS, UCS, GREEDY)


def skill_for(spec) -> SkillDefinition:
    return SkillDefinition(
        skill_id="AI-SRC-04",
        topic="Search and Problem Solving",
        subtopic="Uninformed search",
        name=spec.algorithm,
        learning_objective=f"Trace {spec.algorithm}.",
        cognitive_process="apply",
        generation_strategy="templated",
        template_id=spec.template_id,
    )


# --- hand-checked traversals -------------------------------------------------

def test_breadth_first_order_is_hand_checked():
    # S, then A and B; A reaches C, B reaches D, C reaches G.
    assert uninformed_order(FIXTURE, "breadth") == ("S", "A", "B", "C", "D", "G")


def test_depth_first_order_is_hand_checked():
    # S pushes B then A, so A is on top; A -> C -> G ends it.
    assert uninformed_order(FIXTURE, "depth") == ("S", "A", "C", "G")


def test_uniform_cost_order_is_hand_checked():
    solution = solve(FIXTURE, "cost")

    assert solution.expansion_order == ("S", "A", "C", "B", "G")
    assert [step.g for step in solution.expansions] == [0, 1, 3, 4, 6]
    assert solution.path == ("S", "A", "C", "G")
    assert solution.path_cost == 6


def test_greedy_order_is_hand_checked():
    solution = solve(FIXTURE, "heuristic")

    assert solution.expansion_order == ("S", "B", "D", "G")
    assert [step.h for step in solution.expansions] == [4, 3, 1, 0]
    assert solution.path_cost == 10


def test_greedy_can_be_worse_than_uniform_cost():
    # The point of the greedy skill: lowest h(n) found a costlier path.
    assert solve(FIXTURE, "heuristic").path_cost > solve(FIXTURE, "cost").path_cost


def test_uninformed_orders_ignore_edge_costs():
    expensive = GraphInstance({pair: 99 for pair in EDGE_COSTS}, HEURISTICS)

    assert uninformed_order(expensive, "breadth") == uninformed_order(
        FIXTURE, "breadth"
    )
    assert uninformed_order(expensive, "depth") == uninformed_order(FIXTURE, "depth")


def test_tied_frontier_is_rejected_for_priority_searches():
    tied = GraphInstance(EDGE_COSTS, HEURISTICS | {"B": 1})

    with pytest.raises(UnusableInstance, match="tie on f"):
        solve(tied, "astar")


# --- the four templates ------------------------------------------------------

@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.template_id)
def test_template_is_registered_and_routed(spec):
    question = generate_templated_question(skill_for(spec), "intermediate", seed=3)

    assert isinstance(question, QuizQuestion)
    assert question.concept == spec.algorithm


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.template_id)
@pytest.mark.parametrize("seed", range(8))
def test_options_are_distinct_and_contain_the_answer(spec, seed):
    question = generate_templated_question(skill_for(spec), "intermediate", seed=seed)

    assert len(question.options) == 4
    assert len(set(question.options)) == 4
    assert question.correct_answer in question.options
    assert QuizQuestion.model_validate(question.model_dump()) == question


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.template_id)
def test_the_same_seed_produces_an_identical_question(spec):
    first = generate_templated_question(skill_for(spec), "advanced", seed=42)
    second = generate_templated_question(skill_for(spec), "advanced", seed=42)

    assert first.model_dump() == second.model_dump()


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.template_id)
def test_different_seeds_produce_different_questions(spec):
    questions = {
        generate_templated_question(skill_for(spec), "intermediate", seed=seed).question
        for seed in range(8)
    }

    assert len(questions) > 1


@pytest.mark.parametrize("spec", SPECS, ids=lambda spec: spec.template_id)
def test_the_frontier_rule_is_stated_in_the_question(spec):
    question = generate_templated_question(skill_for(spec), "intermediate", seed=5)

    assert spec.rule in question.question


def test_only_greedy_shows_the_heuristic():
    for spec in SPECS:
        question = generate_templated_question(skill_for(spec), "intermediate", seed=5)

        assert ("h(S) =" in question.question) is spec.shows_heuristics


@pytest.mark.parametrize(
    "spec, expected",
    [
        (BFS, "S -> A -> B -> C -> D -> G"),
        (DFS, "S -> A -> C -> G"),
        (UCS, "S -> A -> C -> B -> G"),
        (GREEDY, "S -> B -> D -> G"),
    ],
    ids=lambda value: getattr(value, "template_id", ""),
)
def test_each_template_answers_with_its_hand_checked_order(spec, expected):
    question = build_trace_question(spec, skill_for(spec), "intermediate", FIXTURE)

    assert question.correct_answer == expected
    assert len(set(question.options)) == 4


def test_each_template_answers_differently_on_the_same_graph():
    answers = {
        build_trace_question(spec, skill_for(spec), "intermediate", FIXTURE)
        .correct_answer
        for spec in SPECS
    }

    assert len(answers) == 4
