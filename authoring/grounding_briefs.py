"""Versioned canonical briefs used by grounded authoring and generation."""

import hashlib
import json

from pydantic import BaseModel, Field


GROUNDING_BRIEF_VERSION = "pilot-grounding-v1"


class CanonicalGroundingBrief(BaseModel):
    skill_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    statements: list[str] = Field(min_length=1)
    intent_reference_ids: dict[str, list[str]] = Field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


PILOT_GROUNDING_BRIEFS = {
    brief.skill_id: brief
    for brief in (
        CanonicalGroundingBrief(
            skill_id="AI-FND-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "Intelligent-system capabilities include problem solving, reasoning, decision making, and learning.",
                "Concrete AI applications include face recognition, game playing, speech processing, and route finding.",
                "Automation alone does not establish that a system is intelligent.",
                "An intelligent agent receives percepts from its environment and selects actions based on them.",
                "Introductory questions should test recognition or simple interpretation rather than formal notation or full-list recall.",
            ],
            intent_reference_ids={
                "AI-FND-01-INT-01": [
                    "AI-FND-01-8bbbddaf2aa6",
                    "AI-FND-01-b50c85fa00a5",
                ],
                "AI-FND-01-INT-02": ["AI-FND-01-d03d77e0aca2"],
                "AI-FND-01-INT-03": ["AI-FND-01-f7c5eb1ccf76"],
            },
        ),
        CanonicalGroundingBrief(
            skill_id="AI-AGT-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "An agent perceives its environment and acts upon that environment.",
                "Sensors supply percepts or information from the environment; actuators carry out actions that affect it.",
                "PEAS means performance measure, environment, actuators, and sensors.",
                "A partially observable environment does not provide the agent with full state information.",
                "Questions must not treat the performance measure, environment, actuator, and sensor as interchangeable roles.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="AI-SRC-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "Initial state and start state are synonyms.",
                "An empty assignment has no assigned variables.",
                "Do not invent actor relationships or search states not stated in the sources.",
                "Action cost is the cost of an individual action; path cost is accumulated over a path, and the two must not be conflated.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="AI-SRC-02",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "A search tree is at least as large as its state-space graph.",
                "Cycles may make the search tree infinitely deep.",
                "Cycles do not imply that a goal is unreachable.",
                "Search-tree nodes distinguish paths, so a state may occur repeatedly.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="AI-SRC-03",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "The frontier contains generated search nodes that are waiting to be selected for expansion.",
                "Expansion removes a selected node, applies the available actions through the result model, and generates child nodes.",
                "In the project solvers, reached means discovered: BFS and DFS record a state when it is admitted to the frontier.",
                "Explored or expanded means a state has been removed from the frontier and expanded; these names must not be used as silent synonyms for reached.",
                "A repeated state is not added again when the existing record is already at least as good.",
                "For an unexpanded state in a cost-sensitive frontier, a newly discovered lower path cost replaces the stored frontier cost and parent record.",
                "The current solver does not reopen a state after it has entered the expanded set, so questions must not imply that it does.",
                "Questions for this skill explain frontier, reached-state, duplicate, and expansion mechanics without asking for a BFS, DFS, UCS, Greedy, or A-star expansion trace.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="AI-SRC-08",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "h(n) estimates the remaining forward cost.",
                "g(n) is the accumulated cost from the start.",
                "f(n) = g(n) + h(n).",
                "Greedy Best-First Search prioritizes h(n).",
                "UCS and Dijkstra's algorithm prioritize g(n).",
                "Manhattan distance must only be used for an appropriate grid setting.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DSA-CPX-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "Asymptotic (Big-O) notation describes how an algorithm's running time grows as input size increases, not its exact running time.",
                "Changing a constant factor in a running-time expression shifts where two growth curves cross, but does not change which one eventually grows faster.",
                "A faster computer or a constant-factor code optimization does not change an algorithm's asymptotic growth rate.",
                "Linear search is O(n); binary search on sorted data is O(log n) and is more efficient for large inputs.",
                "Binary search requires the data to already be sorted; it does not work correctly on unsorted data.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DSA-LST-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "An array stores its elements in contiguous memory, which allows any element to be read directly by index in constant time.",
                "Inserting into or deleting from the middle of an array-based list requires shifting the surrounding elements.",
                "A linked-list node stores an element together with a pointer (the next field) to the following node, not a copy of the following element's value.",
                "A singly linked list must be traversed node by node from the head to reach a given position; it does not support constant-time indexed access like an array.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DSA-STK-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "A stack is Last-In, First-Out (LIFO): elements are pushed onto and popped from the same end, called the top.",
                "A queue is First-In, First-Out (FIFO): elements are enqueued at the back and dequeued from the front.",
                "A stack pop always returns the most recently pushed remaining element, never the earliest one.",
                "A queue dequeue always returns the earliest remaining enqueued element, never the most recent one.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DSA-SRC-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "Linear search examines elements one at a time, in order, until it finds the target or exhausts the list; its running time is O(n).",
                "Binary search only works on sorted data: it compares the target to the middle element and continues searching only the half of the list that could still contain the target.",
                "Binary search's running time is O(log n), which is more efficient than linear search's O(n) for large sorted inputs.",
                "Binary search must discard, not continue searching, the half of the list that cannot contain the target.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DSA-SRT-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "Bubble sort repeatedly compares adjacent elements and swaps them if out of order; one pass does not fully sort the list.",
                "Selection sort repeatedly finds the smallest element in the unsorted portion and moves it to the front of that portion.",
                "Insertion sort inserts each next element into its correct position among the already-sorted elements that precede it.",
                "Merge sort is divide-and-conquer: it splits the list in half, recursively sorts each half, then merges the two sorted halves.",
                "Quicksort is divide-and-conquer: it selects a pivot, partitions the remaining elements into those less than and greater than the pivot, and recursively sorts each partition -- it does not split the list at a fixed midpoint the way merge sort does.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DSA-HSH-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "A hash function maps a search key to a position (index) in the hash table; it does not sort the table.",
                "A collision occurs when two different keys hash to the same table position; a collision is a normal, expected event, not a hash-function failure.",
                "Under chaining, a colliding key is appended to the linked list already stored at that index rather than overwriting the existing entry.",
                "Under open addressing, when a key's home position is already occupied, the collision resolution policy probes a sequence of other slots until it finds a free one, rather than failing the insertion.",
            ],
        ),
        CanonicalGroundingBrief(
            skill_id="DSA-TGR-01",
            version=GROUNDING_BRIEF_VERSION,
            statements=[
                "In a binary search tree, every node in the left subtree of a node with key K has a key less than or equal to K, and every node in the right subtree has a key greater than K.",
                "An inorder traversal of a binary search tree visits its nodes in ascending sorted order.",
                "In a max-heap, every node's value is greater than or equal to the values of its children; a heap is not ordered the way a binary search tree is.",
                "A graph traversal (such as breadth-first or depth-first search) visits every vertex exactly once; a visiting order that skips or repeats a vertex is not a valid traversal.",
            ],
        ),
    )
}


def grounding_brief(skill_id: str) -> CanonicalGroundingBrief:
    try:
        return PILOT_GROUNDING_BRIEFS[skill_id]
    except KeyError as error:
        raise ValueError(f"no canonical grounding brief for {skill_id}") from error
