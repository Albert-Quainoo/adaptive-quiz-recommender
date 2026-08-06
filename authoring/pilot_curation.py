"""Preliminary human-review proposals for grounded-pilot-20260805-v2."""

from datetime import datetime, timezone
from pathlib import Path

from api.schemas import QuizQuestion
from authoring.grounded_review import (
    CurationItem,
    GroundedReview,
    RevisionProvenance,
    propose_revision,
    source_hashes,
)
from authoring.grounded_batch import PendingQuestion


PROPOSAL_EDITOR = "grounded-curation-proposal"
PROPOSAL_TIME = datetime(2026, 8, 6, 12, 0, tzinfo=timezone.utc)
SOURCE_HASHES = {
    "audit.jsonl": "4c2dae537e2f6c80aaee77d58b6da4c32a904d2363a55d8ae5cea418e4bea6eb",
    "manifest.json": "12a88215aa5e310b0ac8b95fad7f9fb47df8754fc5cb8ac5344cbcea08210dd4",
    "pending_questions.jsonl": "e7813868daa027e292094245e9a8c4280385a436a107c138c1dcd08d9f017931",
    "summary.json": "09c82810c75b1328a321259f8160024e9623698a99b7d31dfde9dc56fedc2d63",
}


RECOMMENDATIONS = {
    "AI-SRC-01-INT-01": ("reject", "The stem supplies the target formulation and the long options mix unrelated scenarios rather than testing one clear decision."),
    "AI-SRC-01-INT-02": ("reject", "The selected option contradicts the definition of an empty assignment by assigning a variable."),
    "AI-SRC-01-INT-03": ("reject", "The item assesses initial state instead of actions and invents unsupported absence-of-relationship facts."),
    "AI-SRC-01-INT-04": ("propose_revision", "Retain the transition-model focus but replace the answer-revealing stem with a concrete state transition."),
    "AI-SRC-01-INT-05": ("propose_revision", "The core goal-test distinction is sound; tighten the scenario and use role-based distractors."),
    "AI-SRC-01-INT-06": ("reject", "The item does not assess action cost versus path cost and several options could describe non-initial states."),
    "AI-SRC-01-INT-07": ("propose_revision", "Restore the assigned missing-component diagnosis instead of repeating initial-state recall."),
    "AI-SRC-01-INT-08": ("reject", "The stem is a false non-question statement and the selected option incorrectly adds violated constraints to an empty assignment."),
    "AI-SRC-01-INT-09": ("reject", "The item invents a no-connections state and does not formalize the sourced Degrees representation."),
    "AI-SRC-01-INT-10": ("reject", "The selected state contains an assignment while the explanation claims it is the empty assignment."),
    "AI-SRC-02-INT-01": ("reject", "The comparison asks about both structures but selects an option describing only the state-space graph, leaving another option independently true."),
    "AI-SRC-02-INT-02": ("reject", "Two options can be read as correct and the state-space restriction is stated imprecisely."),
    "AI-SRC-02-INT-03": ("reject", "The item refers to an unavailable diagram and offers overlapping descriptions of the same path."),
    "AI-SRC-02-INT-04": ("reject", "The selected equality claim contradicts the source: a search tree is at least as large as its state-space graph."),
    "AI-SRC-02-INT-05": ("reject", "The explanation incorrectly claims that a cycle makes the goal unreachable."),
    "AI-SRC-02-INT-06": ("propose_revision", "Keep the cycle consequence but separate infinite tree depth from claims about goal reachability or DFS behavior."),
    "AI-SRC-02-INT-07": ("propose_revision", "Use a concrete same-state/different-path case to test search-tree node identity."),
    "AI-SRC-02-INT-08": ("propose_revision", "Remove the definition embedded in the stem and ask students to choose the representation from a concise requirement."),
    "AI-SRC-02-INT-09": ("reject", "The declarative stem repeats the selected option and does not diagnose the intended invalid finite-tree claim."),
    "AI-SRC-02-INT-10": ("propose_revision", "Clarify that the notation is a root-to-node path, not a state, and ask what the node records."),
    "AI-SRC-08-INT-01": ("reject", "The selected answer describes f(n), not how h(n) estimates remaining cost."),
    "AI-SRC-08-INT-02": ("propose_revision", "Replace the answer-revealing definition with a four-way-grid Manhattan-distance calculation."),
    "AI-SRC-08-INT-03": ("propose_revision", "Add frontier heuristic values so the learner must apply Greedy ordering rather than recall its definition."),
    "AI-SRC-08-INT-04": ("reject", "The selected answer conflates the heuristic h(n) with A-star's combined f(n)."),
    "AI-SRC-08-INT-05": ("reject", "Two options state the same correct sum, so the item lacks a unique defensible answer."),
    "AI-SRC-08-INT-06": ("propose_revision", "Manhattan distance needs an explicitly stated four-way grid with unit horizontal and vertical moves."),
    "AI-SRC-08-INT-07": ("reject", "The selected answer again describes f(n) while the stem asks specifically about h(n)."),
    "AI-SRC-08-INT-08": ("propose_revision", "Turn the accurate comparison into an applied priority-ordering scenario."),
    "AI-SRC-08-INT-09": ("reject", "The selected answer conflates remaining-cost estimation with combined start-to-goal estimation."),
    "AI-SRC-08-INT-10": ("reject", "The selected option names individual action costs instead of the accumulated g(n) omitted by Greedy ordering."),
}


def revised(
    stem: str,
    options: list[str],
    correct: str,
    explanation: str,
    concept: str,
) -> QuizQuestion:
    return QuizQuestion(
        question=stem,
        options=options,
        correct_answer=correct,
        explanation=explanation,
        concept=concept,
        difficulty="intermediate",
    )


PROPOSED_QUESTIONS = {
    "AI-SRC-01-INT-04": revised(
        "A constraint-satisfaction search state is {X = red}. The action assigns Y = blue. What should the transition model return?",
        [
            "The resulting state {X = red, Y = blue}",
            "The action 'assign Y = blue' without a resulting state",
            "A Boolean indicating whether every constraint is satisfied",
            "The accumulated cost of all assignments made so far",
        ],
        "The resulting state {X = red, Y = blue}",
        "A transition model maps the current state and an action to the resulting state, so the new assignment retains X = red and adds Y = blue.",
        "Transition model",
    ),
    "AI-SRC-01-INT-05": revised(
        "Which goal test is appropriate when a constraint-satisfaction problem is formulated as search?",
        [
            "Every variable has a value and every constraint is satisfied",
            "At least one variable has been assigned a value",
            "The latest assignment changed the current state",
            "The accumulated path cost is greater than zero",
        ],
        "Every variable has a value and every constraint is satisfied",
        "A CSP solution is reached only when the assignment is complete and satisfies all constraints.",
        "Goal test",
    ),
    "AI-SRC-01-INT-07": revised(
        "A course formulation defines a search problem using an initial state, actions, a transition model, a goal test, and a path-cost function. If the first four have been specified, which component is missing?",
        ["Path-cost function", "Start state", "Successor state", "Search-tree depth"],
        "Path-cost function",
        "In this formulation, the initial state, actions, transition model and goal test are already specified, leaving the path-cost function as the missing component.",
        "Missing search-problem component",
    ),
    "AI-SRC-02-INT-06": revised(
        "A finite state-space graph contains a reachable cycle. Why can its corresponding search tree have infinite depth?",
        [
            "The cycle can be traversed repeatedly, producing paths of unbounded length",
            "Every cycle makes every goal state unreachable",
            "A cycle creates infinitely many distinct states in the state-space graph",
            "The search tree stores each state only once",
        ],
        "The cycle can be traversed repeatedly, producing paths of unbounded length",
        "Search-tree nodes represent paths, so repeatedly following a cycle can create arbitrarily long paths even when the state-space graph is finite.",
        "Cycles and search-tree depth",
    ),
    "AI-SRC-02-INT-07": revised(
        "Two action sequences, S → A → C and S → B → C, reach the same state C. How are they represented in a search tree?",
        [
            "As two distinct nodes because the paths to C are different",
            "As one node because both sequences end in state C",
            "As no nodes because repeated states are forbidden in a search tree",
            "As one edge directly connecting S to C",
        ],
        "As two distinct nodes because the paths to C are different",
        "A search-tree node records the path as well as the resulting state, so different paths to C produce distinct nodes.",
        "Search-tree node identity",
    ),
    "AI-SRC-02-INT-08": revised(
        "Which representation can contain a separate node for every candidate action sequence beginning at a start state?",
        ["Search tree", "State-space graph", "Goal-test function", "Transition model"],
        "Search tree",
        "A search tree represents candidate paths from the root, while a state-space graph represents each underlying state without duplicating it for every path.",
        "Search tree representation",
    ),
    "AI-SRC-02-INT-10": revised(
        "A search-tree node is reached along S → d → e → r → f → G. What does that node encode?",
        [
            "The complete path from S through d, e, r, and f to G",
            "Only the terminal state G",
            "Only the most recent action f → G",
            "An unordered set of the visited states",
        ],
        "The complete path from S through d, e, r, and f to G",
        "Each search-tree node represents the entire path from the start state to that node, not only its terminal state.",
        "Path encoding in search trees",
    ),
    "AI-SRC-08-INT-02": revised(
        "On a four-way square grid with unit-cost horizontal and vertical moves, what is the Manhattan-distance heuristic from (2, 1) to (5, 5)?",
        ["7", "5", "4", "25"],
        "7",
        "Manhattan distance is the sum of the absolute horizontal and vertical coordinate differences:\n\n|5 - 2| + |5 - 1| = 3 + 4 = 7.",
        "Manhattan-distance heuristic",
    ),
    "AI-SRC-08-INT-03": revised(
        "Greedy Best-First Search has frontier nodes A, B, C, and D with h-values 4, 2, 6, and 3 respectively. Which node is expanded next?",
        ["B", "D", "A", "C"],
        "B",
        "Greedy Best-First Search prioritizes the lowest h-value, and B has the lowest estimate at 2.",
        "Greedy Best-First priority",
    ),
    "AI-SRC-08-INT-06": revised(
        "A robot moves on a square grid using unit-cost moves north, south, east, or west; diagonal moves are not allowed. Which heuristic appropriately estimates remaining distance to a target cell?",
        [
            "The sum of the absolute horizontal and vertical coordinate differences",
            "The accumulated cost from the start to the robot's current cell",
            "The number of cells already expanded by the search",
            "The product of the horizontal and vertical coordinate differences",
        ],
        "The sum of the absolute horizontal and vertical coordinate differences",
        "With four-way unit-cost movement, Manhattan distance matches the horizontal plus vertical moves needed when obstacles are ignored.",
        "Heuristic design for four-way grids",
    ),
    "AI-SRC-08-INT-08": revised(
        "Two frontier nodes have values A: g = 3, h = 8 and B: g = 6, h = 2. Which node does Greedy Best-First prioritize, and which does Uniform-Cost Search/Dijkstra's algorithm prioritize?",
        [
            "Greedy prioritizes B; Uniform-Cost Search/Dijkstra's algorithm prioritizes A",
            "Greedy prioritizes A; Uniform-Cost Search/Dijkstra's algorithm prioritizes B",
            "Both prioritize A",
            "Both prioritize B",
        ],
        "Greedy prioritizes B; Uniform-Cost Search/Dijkstra's algorithm prioritizes A",
        "Greedy prioritizes B because B has the lower h-value. Uniform-Cost Search/Dijkstra's algorithm prioritizes A because A has the lower g-value.",
        "Greedy versus Dijkstra priority",
    ),
}


def build_pilot_review(batch_path: Path) -> GroundedReview:
    actual_hashes = source_hashes(batch_path)
    if actual_hashes != SOURCE_HASHES:
        raise ValueError("grounded-pilot-20260805-v2 source batch is not immutable")
    sources = [
        PendingQuestion.model_validate_json(line)
        for line in (batch_path / "pending_questions.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if {source.intent_id for source in sources} != set(RECOMMENDATIONS):
        raise ValueError("source batch intents do not match the curation decisions")
    items = []
    for source in sources:
        recommendation, reason = RECOMMENDATIONS[source.intent_id]
        item = CurationItem(
            original_question_id=source.question_id,
            skill_id=source.skill_id,
            intent_id=source.intent_id,
            recommendation=recommendation,
            recommendation_reason=reason,
        )
        if recommendation == "propose_revision":
            item = propose_revision(
                item,
                source.question,
                PROPOSED_QUESTIONS[source.intent_id],
                PROPOSAL_EDITOR,
                reason,
                edited_at=PROPOSAL_TIME,
                provenance=RevisionProvenance.from_source(source),
            )
        items.append(item)
    return GroundedReview(
        batch_id="grounded-pilot-20260805-v2",
        source_hashes=SOURCE_HASHES,
        items=items,
    )
