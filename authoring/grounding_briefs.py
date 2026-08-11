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
    )
}


def grounding_brief(skill_id: str) -> CanonicalGroundingBrief:
    try:
        return PILOT_GROUNDING_BRIEFS[skill_id]
    except KeyError as error:
        raise ValueError(f"no canonical grounding brief for {skill_id}") from error
