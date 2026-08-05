import random

from api.schemas import QuizQuestion, difficulty_level
from taxonomy.schemas import SkillDefinition
from templates.graphs import (
    MAX_ATTEMPTS,
    Expansion,
    GraphInstance,
    Solution,
    UnusableInstance,
    check_heuristics,
    distances_to_goal,
    expansion_order,
    format_edges,
    format_heuristics,
    format_order,
    generate_instance,
    shuffled_options,
    solve,
    swap_middle,
)
from templates.registry import TemplateError, register

TEMPLATE_ID = "search.astar_trace"

# A-star reads on a graph instance like any other search; these aliases keep
# the names the template and its tests have always used.
AStarInstance = GraphInstance
AStarSolution = Solution

__all__ = [
    "AStarInstance",
    "AStarSolution",
    "Expansion",
    "UnusableInstance",
    "build_astar_question",
    "check_heuristics",
    "distances_to_goal",
    "expansion_order",
    "generate",
    "generate_astar_instance",
    "solve_astar",
]


def generate_astar_instance(
    difficulty: difficulty_level, rng: random.Random
) -> GraphInstance:
    return generate_instance(difficulty, rng)


def solve_astar(instance: GraphInstance) -> Solution:
    """Expand nodes by lowest f(n) = g(n) + h(n)."""
    return solve(instance, "astar")


def build_astar_question(
    skill: SkillDefinition,
    difficulty: difficulty_level,
    solution: AStarSolution,
    rng: random.Random,
) -> QuizQuestion:
    instance = solution.instance

    correct = format_order(solution.expansion_order)

    options = shuffled_options(
        [
            correct,
            format_order(expansion_order(instance, "heuristic")),
            format_order(expansion_order(instance, "cost")),
            format_order(swap_middle(solution.expansion_order)),
        ],
        rng,
    )

    trace = ", ".join(
        f"{expansion.node} (g={expansion.g}, h={expansion.h}, f={expansion.f})"
        for expansion in solution.expansions
    )

    return QuizQuestion(
        question=(
            f"An undirected graph has these edge costs: {format_edges(instance)}. "
            f"The heuristic values are: {format_heuristics(instance)}. "
            f"Starting at {instance.start} with goal {instance.goal}, in what order "
            f"does A-star search expand the nodes, using f(n) = g(n) + h(n)?"
        ),
        options=options,
        correct_answer=correct,
        explanation=(
            f"A-star expands the frontier node with the lowest f(n) = g(n) + h(n). "
            f"That gives {trace}. "
            f"The goal is reached by {format_order(solution.path)} "
            f"at a path cost of {solution.path_cost}."
        ),
        concept=skill.name,
        difficulty=difficulty,
    )


@register(TEMPLATE_ID)
def generate(
    skill: SkillDefinition,
    difficulty: difficulty_level,
    seed: int | None = None,
) -> QuizQuestion:
    if seed is None:
        seed = random.randrange(2**32)

    rng = random.Random(seed)

    for _ in range(MAX_ATTEMPTS):
        try:
            instance = generate_astar_instance(difficulty, rng)

            return build_astar_question(
                skill, difficulty, solve_astar(instance), rng
            )
        except UnusableInstance:
            continue

    raise TemplateError(
        f"{TEMPLATE_ID} could not build an unambiguous {difficulty} instance "
        f"from seed {seed}."
    )
