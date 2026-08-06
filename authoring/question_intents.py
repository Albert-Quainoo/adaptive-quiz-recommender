"""Reviewed intent blueprints for grounded question generation."""

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator


PILOT_BATCH_ID = "grounded-pilot-20260805-v2"
PILOT_PROMPT_VERSION = "v3.3"
PILOT_BLUEPRINT_PATH = (
    Path(__file__).resolve().parent / "blueprints" / f"{PILOT_BATCH_ID}.json"
)


class QuestionIntent(BaseModel):
    intent_id: str = Field(min_length=1)
    skill_id: str = Field(min_length=1)
    assessment_focus: str = Field(min_length=1)
    question_archetype: str = Field(min_length=1)
    preferred_reference_ids: list[str] = Field(min_length=1)
    required_concepts: list[str] = Field(min_length=1)
    prohibited_conflations: list[str] = Field(min_length=1)

    @field_validator("intent_id", "skill_id", mode="before")
    @classmethod
    def normalise_identifier(cls, value: str) -> str:
        return value.strip().upper() if isinstance(value, str) else value

    @field_validator(
        "assessment_focus",
        "question_archetype",
        "preferred_reference_ids",
        "required_concepts",
        "prohibited_conflations",
        mode="before",
    )
    @classmethod
    def strip_text(cls, value):
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return [item.strip() if isinstance(item, str) else item for item in value]
        return value

    @model_validator(mode="after")
    def identifiers_match(self) -> "QuestionIntent":
        if not self.intent_id.startswith(f"{self.skill_id}-INT-"):
            raise ValueError("intent_id must be namespaced by skill_id")
        if len(self.preferred_reference_ids) != len(set(self.preferred_reference_ids)):
            raise ValueError("preferred_reference_ids must be distinct")
        return self


class PilotBlueprint(BaseModel):
    batch_id: str
    prompt_version: str
    review_status: str
    intents: list[QuestionIntent] = Field(min_length=1)

    @model_validator(mode="after")
    def intents_are_distinct(self) -> "PilotBlueprint":
        ids = [intent.intent_id for intent in self.intents]
        if len(ids) != len(set(ids)):
            raise ValueError("intent ids must be distinct")
        return self


def load_pilot_blueprint(path: Path = PILOT_BLUEPRINT_PATH) -> PilotBlueprint:
    blueprint = PilotBlueprint.model_validate_json(path.read_text(encoding="utf-8"))
    if blueprint.batch_id != PILOT_BATCH_ID:
        raise ValueError(f"blueprint batch_id must be {PILOT_BATCH_ID}")
    if blueprint.prompt_version != PILOT_PROMPT_VERSION:
        raise ValueError(f"blueprint prompt_version must be {PILOT_PROMPT_VERSION}")
    return blueprint


def intents_by_skill(blueprint: PilotBlueprint) -> dict[str, list[QuestionIntent]]:
    grouped: dict[str, list[QuestionIntent]] = {}
    for intent in blueprint.intents:
        grouped.setdefault(intent.skill_id, []).append(intent)
    return grouped


def format_blueprint_review(blueprint: PilotBlueprint) -> str:
    """Return deterministic JSON suitable for pre-run human review."""
    return json.dumps(blueprint.model_dump(mode="json"), indent=2, sort_keys=True)
