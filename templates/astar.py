import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from api.schemas import QuizQuestion, difficulty_level
from taxonomy.schemas import SkillDefinition
from templates.registry import TemplateError, register

TEMPLATE_ID = "search.astar_trace"

START = "S"
GOAL = "G"

# A small family of topologies, so a learner meeting several of these questions
# does not see the same graph shape every time. In each one every node reaches
# the goal, and the start is never adjacent to it, so an instance always has a
# solution worth tracing. Costs and heuristic values vary within a topology.
TOPOLOGIES = (
    (
        ("S", "A"),
        ("S", "B"),
        ("A", "C"),
        ("B", "C"),
        ("B", "D"),
        ("C", "G"),
        ("D", "G"),
    ),
    (
        ("S", "A"),
        ("S", "B"),
        ("A", "C"),
        ("A", "D"),
        ("B", "D"),
        ("C", "E"),
        ("D", "E"),
        ("D", "G"),
        ("E", "G"),
    ),
)

COST_RANGES = {
    "introductory": (1, 5),
    "intermediate": (1, 9),
    "advanced": (1, 15),
}

HEURISTIC_SCALES = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

MAX_ATTEMPTS = 200


class UnusableInstance(ValueError):
    pass


def neighbours(
    edge_costs: dict[tuple[str, str], int], node: str
) -> Iterator[tuple[str, int]]:
    for (left, right), cost in edge_costs.items():
        if left == node:
            yield right, cost
        elif right == node:
            yield left, cost


def nodes_of(edge_costs: dict[tuple[str, str], int]) -> tuple[str, ...]:
    ordered = dict.fromkeys(node for pair in edge_costs for node in pair)

    return tuple(ordered)


@dataclass(frozen=True)
class AStarInstance:
    edge_costs: dict[tuple[str, str], int]
    heuristics: dict[str, int]
    start: str = START
    goal: str = GOAL

    @property
    def nodes(self) -> tuple[str, ...]:
        return nodes_of(self.edge_costs)

    def neighbours(self, node: str) -> Iterator[tuple[str, int]]:
        return neighbours(self.edge_costs, node)


@dataclass(frozen=True)
class Expansion:
    node: str
    g: int
    h: int

    @property
    def f(self) -> int:
        return self.g + self.h


@dataclass(frozen=True)
class AStarSolution:
    instance: AStarInstance
    expansions: tuple[Expansion, ...]
    path: tuple[str, ...]
    path_cost: int

    @property
    def expansion_order(self) -> tuple[str, ...]:
        return tuple(expansion.node for expansion in self.expansions)


def distances_to_goal(edge_costs: dict[tuple[str, str], int]) -> dict[str, int]:
    distances = {GOAL: 0}
    frontier = {GOAL: 0}

    while frontier:
        node = min(frontier, key=lambda candidate: frontier[candidate])
        distance = frontier.pop(node)
        distances[node] = distance

        for neighbour, cost in neighbours(edge_costs, node):
            if neighbour in distances:
                continue

            candidate = distance + cost

            if candidate < frontier.get(neighbour, candidate + 1):
                frontier[neighbour] = candidate

    return distances


def solve_astar(instance: AStarInstance) -> AStarSolution:
    """Expand nodes by lowest f(n) = g(n) + h(n).

    An instance whose frontier ever holds two nodes with the same lowest f is
    rejected rather than broken by a tie rule, so every question has one
    defensible expansion order.
    """
    frontier = {instance.start: 0}
    parents: dict[str, str] = {}
    expansions: list[Expansion] = []
    expanded: set[str] = set()

    while frontier:
        ranked = sorted(
            frontier.items(),
            key=lambda item: (item[1] + instance.heuristics[item[0]], item[0]),
        )
        node, g = ranked[0]

        if len(ranked) > 1:
            runner_up, runner_up_g = ranked[1]

            if g + instance.heuristics[node] == (
                runner_up_g + instance.heuristics[runner_up]
            ):
                raise UnusableInstance(
                    f"{node} and {runner_up} tie on f, so the expansion order "
                    f"is ambiguous."
                )

        del frontier[node]
        expanded.add(node)
        expansions.append(Expansion(node, g, instance.heuristics[node]))

        if node == instance.goal:
            path = [node]

            while path[-1] != instance.start:
                path.append(parents[path[-1]])

            return AStarSolution(
                instance=instance,
                expansions=tuple(expansions),
                path=tuple(reversed(path)),
                path_cost=g,
            )

        for neighbour, cost in instance.neighbours(node):
            if neighbour in expanded:
                continue

            candidate = g + cost

            if candidate < frontier.get(neighbour, candidate + 1):
                frontier[neighbour] = candidate
                parents[neighbour] = node

    raise UnusableInstance(f"{instance.goal} is unreachable from {instance.start}.")


def expansion_order(
    instance: AStarInstance, priority: Literal["heuristic", "cost"]
) -> tuple[str, ...]:
    """Expansion order under a different priority, used to build distractors.

    "heuristic" is what greedy best-first search would do and "cost" is what
    uniform-cost search would do, so a student who confuses A-star with either
    lands on one of these. Ties break alphabetically: these orders only have to
    be wrong and reproducible, not defensible.
    """
    frontier = {instance.start: 0}
    order: list[str] = []
    expanded: set[str] = set()

    while frontier:
        if priority == "heuristic":
            node = min(frontier, key=lambda item: (instance.heuristics[item], item))
        else:
            node = min(frontier, key=lambda item: (frontier[item], item))
        g = frontier.pop(node)
        expanded.add(node)
        order.append(node)

        if node == instance.goal:
            break

        for neighbour, cost in instance.neighbours(node):
            if neighbour in expanded:
                continue

            candidate = g + cost

            if candidate < frontier.get(neighbour, candidate + 1):
                frontier[neighbour] = candidate

    return tuple(order)


def check_heuristics(instance: AStarInstance) -> None:
    if instance.heuristics[instance.goal] != 0:
        raise UnusableInstance("the goal must have a heuristic of 0.")

    if any(value < 0 for value in instance.heuristics.values()):
        raise UnusableInstance("heuristic values must not be negative.")

    for (left, right), cost in instance.edge_costs.items():
        if abs(instance.heuristics[left] - instance.heuristics[right]) > cost:
            raise UnusableInstance(
                f"the heuristic is inconsistent across {left}-{right}."
            )


def generate_astar_instance(
    difficulty: difficulty_level, rng: random.Random
) -> AStarInstance:
    """Draw costs and heuristics until they form a consistent instance.

    Scaling the true distance keeps the heuristic admissible, but integer
    rounding can still break consistency across a cheap edge, so each draw is
    checked and redrawn rather than repaired.
    """
    low, high = COST_RANGES[difficulty]

    for _ in range(MAX_ATTEMPTS):
        topology = rng.choice(TOPOLOGIES)
        edge_costs = {pair: rng.randint(low, high) for pair in topology}
        distances = distances_to_goal(edge_costs)
        scale = rng.choice(HEURISTIC_SCALES)

        heuristics = {
            node: max(0, int(distances[node] * scale) - rng.randint(0, 1))
            for node in nodes_of(edge_costs)
        }
        heuristics[GOAL] = 0

        instance = AStarInstance(edge_costs=edge_costs, heuristics=heuristics)

        try:
            check_heuristics(instance)
        except UnusableInstance:
            continue

        return instance

    raise UnusableInstance(f"no consistent {difficulty} instance after {MAX_ATTEMPTS}.")


def format_order(order: tuple[str, ...]) -> str:
    return " -> ".join(order)


def build_astar_question(
    skill: SkillDefinition,
    difficulty: difficulty_level,
    solution: AStarSolution,
) -> QuizQuestion:
    instance = solution.instance

    edges = ", ".join(
        f"{left}-{right} = {cost}" for (left, right), cost in instance.edge_costs.items()
    )
    heuristics = ", ".join(
        f"h({node}) = {instance.heuristics[node]}" for node in instance.nodes
    )

    correct = format_order(solution.expansion_order)
    swapped = list(solution.expansion_order)
    swapped[1], swapped[2] = swapped[2], swapped[1]

    options = [
        correct,
        format_order(expansion_order(instance, "heuristic")),
        format_order(expansion_order(instance, "cost")),
        format_order(tuple(swapped)),
    ]

    if len(set(options)) != len(options):
        raise UnusableInstance("two answer options are identical.")

    trace = ", ".join(
        f"{expansion.node} (g={expansion.g}, h={expansion.h}, f={expansion.f})"
        for expansion in solution.expansions
    )

    return QuizQuestion(
        question=(
            f"An undirected graph has these edge costs: {edges}. "
            f"The heuristic values are: {heuristics}. "
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

            return build_astar_question(skill, difficulty, solve_astar(instance))
        except UnusableInstance:
            continue

    raise TemplateError(
        f"{TEMPLATE_ID} could not build an unambiguous {difficulty} instance "
        f"from seed {seed}."
    )
