"""Graph instances and traversals shared by the search-trace templates.

Every search template asks the same question - in what order does this
algorithm expand the nodes - and differs only in how the frontier is ordered,
so the instance, the traversals and the tie handling live here once.
"""

import random
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal

from api.schemas import difficulty_level

START = "S"
GOAL = "G"

# Topologies grow with difficulty, because that is the only lever BFS and DFS
# respond to - they never look at edge costs, so widening the cost range leaves
# their questions exactly as hard. Two shapes per tier keep a learner from
# seeing the same graph every time. In each one every node reaches the goal and
# the start is never adjacent to it, so an instance always has a trace worth
# following.
TOPOLOGIES_BY_DIFFICULTY = {
    "introductory": (
        (
            ("S", "A"),
            ("S", "B"),
            ("A", "C"),
            ("B", "C"),
            ("C", "G"),
        ),
        (
            ("S", "A"),
            ("S", "B"),
            ("A", "C"),
            ("B", "C"),
            ("B", "D"),
            ("C", "G"),
            ("D", "G"),
        ),
    ),
    "intermediate": (
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
        (
            ("S", "A"),
            ("S", "B"),
            ("A", "C"),
            ("A", "D"),
            ("B", "D"),
            ("B", "E"),
            ("C", "F"),
            ("D", "F"),
            ("E", "F"),
            ("E", "G"),
            ("F", "G"),
        ),
    ),
    "advanced": (
        (
            ("S", "A"),
            ("S", "B"),
            ("A", "C"),
            ("A", "D"),
            ("B", "D"),
            ("B", "E"),
            ("C", "F"),
            ("D", "F"),
            ("D", "H"),
            ("E", "F"),
            ("E", "H"),
            ("F", "G"),
            ("H", "G"),
        ),
        (
            ("S", "A"),
            ("S", "B"),
            ("A", "C"),
            ("A", "D"),
            ("B", "D"),
            ("B", "E"),
            ("C", "F"),
            ("D", "F"),
            ("D", "H"),
            ("E", "H"),
            ("E", "J"),
            ("F", "G"),
            ("H", "J"),
            ("H", "G"),
            ("J", "G"),
        ),
    ),
}

COST_RANGES = {
    "introductory": (1, 5),
    "intermediate": (1, 9),
    "advanced": (1, 15),
}

HEURISTIC_SCALES = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

MAX_ATTEMPTS = 200

# How the frontier is ordered: by path cost so far, by heuristic estimate, or
# by their sum. Uniform-cost, greedy best-first and A-star respectively.
Priority = Literal["cost", "heuristic", "astar"]


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
class GraphInstance:
    edge_costs: dict[tuple[str, str], int]
    heuristics: dict[str, int]
    start: str = START
    goal: str = GOAL

    @property
    def nodes(self) -> tuple[str, ...]:
        return nodes_of(self.edge_costs)

    def neighbours(self, node: str) -> Iterator[tuple[str, int]]:
        return neighbours(self.edge_costs, node)

    def priority_of(self, node: str, g: int, priority: Priority) -> int:
        if priority == "cost":
            return g

        if priority == "heuristic":
            return self.heuristics[node]

        return g + self.heuristics[node]


@dataclass(frozen=True)
class Expansion:
    node: str
    g: int
    h: int
    f: int


@dataclass(frozen=True)
class Solution:
    instance: GraphInstance
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


def solve(instance: GraphInstance, priority: Priority) -> Solution:
    """Expand the frontier node with the lowest priority value.

    An instance whose frontier ever holds two nodes with the same lowest value
    is rejected rather than broken by a tie rule, so every question has one
    defensible expansion order.
    """
    frontier = {instance.start: 0}
    parents: dict[str, str] = {}
    expansions: list[Expansion] = []
    expanded: set[str] = set()

    while frontier:
        ranked = sorted(
            frontier.items(),
            key=lambda item: (instance.priority_of(item[0], item[1], priority), item[0]),
        )
        node, g = ranked[0]

        if len(ranked) > 1:
            runner_up, runner_up_g = ranked[1]

            if instance.priority_of(node, g, priority) == instance.priority_of(
                runner_up, runner_up_g, priority
            ):
                raise UnusableInstance(
                    f"{node} and {runner_up} tie on f, so the expansion order "
                    f"is ambiguous."
                )

        del frontier[node]
        expanded.add(node)
        expansions.append(
            Expansion(
                node=node,
                g=g,
                h=instance.heuristics[node],
                f=instance.priority_of(node, g, priority),
            )
        )

        if node == instance.goal:
            path = [node]

            while path[-1] != instance.start:
                path.append(parents[path[-1]])

            return Solution(
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


def expansion_order(instance: GraphInstance, priority: Priority) -> tuple[str, ...]:
    """Expansion order under a given priority, used to build distractors.

    A student who confuses one informed search for another lands on one of
    these. Ties break alphabetically: a distractor only has to be wrong and
    reproducible, not defensible.
    """
    frontier = {instance.start: 0}
    order: list[str] = []
    expanded: set[str] = set()

    while frontier:
        node = min(
            frontier,
            key=lambda item: (instance.priority_of(item, frontier[item], priority), item),
        )
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


def uninformed_order(
    instance: GraphInstance, strategy: Literal["breadth", "depth"]
) -> tuple[str, ...]:
    """Expansion order for BFS or DFS, visiting neighbours alphabetically.

    Neither algorithm looks at edge costs, so the only thing that decides the
    order is the frontier discipline plus a neighbour rule. The rule is stated
    in the question rather than left for the learner to guess.
    """
    frontier = [instance.start]
    order: list[str] = []
    reached = {instance.start}

    while frontier:
        node = frontier.pop(0 if strategy == "breadth" else -1)
        order.append(node)

        if node == instance.goal:
            break

        unvisited = sorted(
            neighbour for neighbour, _ in instance.neighbours(node)
            if neighbour not in reached
        )

        for neighbour in unvisited if strategy == "breadth" else reversed(unvisited):
            reached.add(neighbour)
            frontier.append(neighbour)

    return tuple(order)


def check_heuristics(instance: GraphInstance) -> None:
    if instance.heuristics[instance.goal] != 0:
        raise UnusableInstance("the goal must have a heuristic of 0.")

    if any(value < 0 for value in instance.heuristics.values()):
        raise UnusableInstance("heuristic values must not be negative.")

    for (left, right), cost in instance.edge_costs.items():
        if abs(instance.heuristics[left] - instance.heuristics[right]) > cost:
            raise UnusableInstance(
                f"the heuristic is inconsistent across {left}-{right}."
            )


def generate_instance(
    difficulty: difficulty_level, rng: random.Random
) -> GraphInstance:
    """Draw costs and heuristics until they form a consistent instance.

    Scaling the true distance keeps the heuristic admissible, but integer
    rounding can still break consistency across a cheap edge, so each draw is
    checked and redrawn rather than repaired.
    """
    low, high = COST_RANGES[difficulty]
    topologies = TOPOLOGIES_BY_DIFFICULTY[difficulty]

    for _ in range(MAX_ATTEMPTS):
        topology = rng.choice(topologies)
        edge_costs = {pair: rng.randint(low, high) for pair in topology}
        distances = distances_to_goal(edge_costs)
        scale = rng.choice(HEURISTIC_SCALES)

        heuristics = {
            node: max(0, int(distances[node] * scale) - rng.randint(0, 1))
            for node in nodes_of(edge_costs)
        }
        heuristics[GOAL] = 0

        instance = GraphInstance(edge_costs=edge_costs, heuristics=heuristics)

        try:
            check_heuristics(instance)
        except UnusableInstance:
            continue

        return instance

    raise UnusableInstance(f"no consistent {difficulty} instance after {MAX_ATTEMPTS}.")


def format_order(order: tuple[str, ...]) -> str:
    return " -> ".join(order)


def format_edges(instance: GraphInstance) -> str:
    return ", ".join(
        f"{left}-{right} = {cost}"
        for (left, right), cost in instance.edge_costs.items()
    )


def format_heuristics(instance: GraphInstance) -> str:
    return ", ".join(
        f"h({node}) = {instance.heuristics[node]}" for node in instance.nodes
    )


def shuffled_options(options: list[str], rng: random.Random) -> list[str]:
    """Place the correct answer at an unpredictable index.

    Templates build the correct answer first, so without this every templated
    item would be answerable by picking option one - and the response data
    would teach the knowledge tracer that position, not understanding, predicts
    correctness. The shuffle uses the instance's own rng, so a seed still
    reproduces the item exactly.
    """
    if len(set(options)) != len(options):
        raise UnusableInstance("two answer options are identical.")

    shuffled = list(options)
    rng.shuffle(shuffled)

    return shuffled


def swap_middle(order: tuple[str, ...]) -> tuple[str, ...]:
    """The correct order with two expansions transposed - a near-miss option."""
    swapped = list(order)
    swapped[1], swapped[2] = swapped[2], swapped[1]

    return tuple(swapped)
