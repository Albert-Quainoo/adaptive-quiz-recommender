"""Versioned canonical briefs used by grounded authoring and generation."""

import hashlib
import json

from pydantic import BaseModel, Field


GROUNDING_BRIEF_VERSION = "pilot-grounding-v1"


class CanonicalGroundingBrief(BaseModel):
    skill_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    statements: list[str] = Field(min_length=1)

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
