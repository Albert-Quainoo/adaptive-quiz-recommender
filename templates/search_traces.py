"""BFS, DFS, uniform-cost and greedy best-first expansion-order templates.

Each one shows the learner a graph and asks for the expansion order. They share
the instance generator and question shape; they differ in how the frontier is
ordered, which distractors a confused learner would fall for, and whether the
heuristic is worth showing at all.
"""

import random
from collections.abc import Callable
from dataclasses import dataclass

from api.schemas import QuizQuestion, difficulty_level
from taxonomy.schemas import SkillDefinition
from templates.graphs import (
    MAX_ATTEMPTS,
    GraphInstance,
    UnusableInstance,
    expansion_order,
    format_edges,
    format_heuristics,
    format_order,
    generate_instance,
    solve,
    swap_middle,
    uninformed_order,
)
from templates.registry import TemplateError, register

Order = tuple[str, ...]


@dataclass(frozen=True)
class TraceSpec:
    template_id: str
    algorithm: str
    rule: str
    shows_heuristics: bool
    trace: Callable[[GraphInstance], tuple[Order, str]]
    distractors: Callable[[GraphInstance], list[Order]]


def breadth_first_trace(instance: GraphInstance) -> tuple[Order, str]:
    order = uninformed_order(instance, "breadth")

    return order, f"Taking the frontier in that order gives {format_order(order)}."


def depth_first_trace(instance: GraphInstance) -> tuple[Order, str]:
    order = uninformed_order(instance, "depth")

    return order, f"Taking the frontier in that order gives {format_order(order)}."


def uniform_cost_trace(instance: GraphInstance) -> tuple[Order, str]:
    solution = solve(instance, "cost")
    working = ", ".join(
        f"{expansion.node} (g={expansion.g})" for expansion in solution.expansions
    )

    return solution.expansion_order, (
        f"That gives {working}. The goal is reached by "
        f"{format_order(solution.path)} at a path cost of {solution.path_cost}."
    )


def greedy_trace(instance: GraphInstance) -> tuple[Order, str]:
    solution = solve(instance, "heuristic")
    working = ", ".join(
        f"{expansion.node} (h={expansion.h})" for expansion in solution.expansions
    )

    return solution.expansion_order, (
        f"That gives {working}. The goal is reached by "
        f"{format_order(solution.path)} at a path cost of {solution.path_cost}, "
        f"which greedy search does not try to minimise."
    )


BFS = TraceSpec(
    template_id="search.bfs_trace",
    algorithm="Breadth-First Search",
    rule=(
        "Breadth-First Search takes nodes off the frontier first-in, first-out, "
        "adds a node to the frontier the first time it is reached, and considers "
        "a node's neighbours in alphabetical order"
    ),
    shows_heuristics=False,
    trace=breadth_first_trace,
    # Depth-first order for a learner who confuses the two frontiers, and
    # uniform-cost order for one who wrongly lets the edge costs matter.
    distractors=lambda instance: [
        uninformed_order(instance, "depth"),
        expansion_order(instance, "cost"),
    ],
)

DFS = TraceSpec(
    template_id="search.dfs_trace",
    algorithm="Depth-First Search",
    rule=(
        "Depth-First Search takes nodes off the frontier last-in, first-out, "
        "adds a node to the frontier the first time it is reached, and considers "
        "a node's neighbours in alphabetical order"
    ),
    shows_heuristics=False,
    trace=depth_first_trace,
    distractors=lambda instance: [
        uninformed_order(instance, "breadth"),
        expansion_order(instance, "cost"),
    ],
)

UCS = TraceSpec(
    template_id="search.ucs_trace",
    algorithm="Uniform-Cost Search",
    rule=(
        "Uniform-Cost Search always expands the frontier node with the lowest "
        "path cost g(n)"
    ),
    shows_heuristics=False,
    trace=uniform_cost_trace,
    # Both uninformed orders: a learner who ignores cost lands on one of them.
    distractors=lambda instance: [
        uninformed_order(instance, "breadth"),
        uninformed_order(instance, "depth"),
    ],
)

GREEDY = TraceSpec(
    template_id="search.greedy_trace",
    algorithm="Greedy Best-First Search",
    rule=(
        "Greedy Best-First Search always expands the frontier node with the "
        "lowest heuristic value h(n)"
    ),
    shows_heuristics=True,
    # A-star order for a learner who adds g(n) in as well, uniform-cost order
    # for one who uses g(n) instead of h(n).
    distractors=lambda instance: [
        expansion_order(instance, "astar"),
        expansion_order(instance, "cost"),
    ],
    trace=greedy_trace,
)


def build_trace_question(
    spec: TraceSpec,
    skill: SkillDefinition,
    difficulty: difficulty_level,
    instance: GraphInstance,
) -> QuizQuestion:
    order, working = spec.trace(instance)
    correct = format_order(order)

    options = [correct] + [
        format_order(distractor) for distractor in spec.distractors(instance)
    ]
    options.append(format_order(swap_middle(order)))

    if len(set(options)) != len(options):
        raise UnusableInstance("two answer options are identical.")

    heuristics = (
        f"The heuristic values are: {format_heuristics(instance)}. "
        if spec.shows_heuristics
        else ""
    )

    return QuizQuestion(
        question=(
            f"An undirected graph has these edge costs: {format_edges(instance)}. "
            f"{heuristics}"
            f"{spec.rule}. "
            f"Starting at {instance.start} with goal {instance.goal}, in what order "
            f"does it expand the nodes?"
        ),
        options=options,
        correct_answer=correct,
        explanation=f"{spec.rule}. {working}",
        concept=skill.name,
        difficulty=difficulty,
    )


def generate_trace(
    spec: TraceSpec,
    skill: SkillDefinition,
    difficulty: difficulty_level,
    seed: int | None,
) -> QuizQuestion:
    if seed is None:
        seed = random.randrange(2**32)

    rng = random.Random(seed)

    for _ in range(MAX_ATTEMPTS):
        try:
            instance = generate_instance(difficulty, rng)

            return build_trace_question(spec, skill, difficulty, instance)
        except UnusableInstance:
            continue

    raise TemplateError(
        f"{spec.template_id} could not build an unambiguous {difficulty} instance "
        f"from seed {seed}."
    )


@register(BFS.template_id)
def generate_bfs(
    skill: SkillDefinition, difficulty: difficulty_level, seed: int | None = None
) -> QuizQuestion:
    return generate_trace(BFS, skill, difficulty, seed)


@register(DFS.template_id)
def generate_dfs(
    skill: SkillDefinition, difficulty: difficulty_level, seed: int | None = None
) -> QuizQuestion:
    return generate_trace(DFS, skill, difficulty, seed)


@register(UCS.template_id)
def generate_ucs(
    skill: SkillDefinition, difficulty: difficulty_level, seed: int | None = None
) -> QuizQuestion:
    return generate_trace(UCS, skill, difficulty, seed)


@register(GREEDY.template_id)
def generate_greedy(
    skill: SkillDefinition, difficulty: difficulty_level, seed: int | None = None
) -> QuizQuestion:
    return generate_trace(GREEDY, skill, difficulty, seed)
