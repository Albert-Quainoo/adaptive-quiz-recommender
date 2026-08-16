from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from api.schemas import difficulty_level


RecommendationReason = Literal[
    "foundational_unseen_skill",
    "lowest_mastery_eligible_skill",
    "prerequisite_exhausted_unlock",
    "requested_difficulty_used",
    "fallback_difficulty_used",
]


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    learner_id: str = Field(min_length=1)
    available_skill_ids: list[str] = Field(min_length=1)
    excluded_item_ids: list[str] = Field(default_factory=list)
    requested_difficulty: difficulty_level | None = None
    # Round-checkpoint learner choice ("focus on weak areas"): narrows skill
    # selection to skills still below the introductory mastery threshold.
    # False (the default) is the normal adaptive-progression policy.
    restrict_to_weak_skills: bool = False

    @field_validator("learner_id", mode="before")
    @classmethod
    def strip_learner_id(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value

    @field_validator("available_skill_ids", "excluded_item_ids")
    @classmethod
    def validate_identifiers(cls, value: list[str]) -> list[str]:
        normalised = [identifier.strip() for identifier in value]
        if any(not identifier for identifier in normalised):
            raise ValueError("identifiers cannot be empty")
        if len(normalised) != len(set(normalised)):
            raise ValueError("identifiers cannot contain duplicates")
        return normalised


class RecommendationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    learner_id: str = Field(min_length=1)
    course_id: str = Field(min_length=1)
    skill_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    difficulty: difficulty_level
    mastery_probability: float = Field(ge=0.0, le=1.0)
    reason: RecommendationReason
    model_version: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)


class RecommendationEvent(RecommendationResult):
    """One immutable recommendation decision recorded for audit and replay."""

    model_config = ConfigDict(frozen=True)

    recommendation_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ContentGapResult(BaseModel):
    """An unlocked pilot skill that cannot be served from the approved bank."""

    model_config = ConfigDict(frozen=True)

    learner_id: str = Field(min_length=1)
    course_id: str = Field(min_length=1)
    completed_skill_id: str = Field(min_length=1)
    completed_skill_name: str = Field(min_length=1)
    newly_unlocked_skill_id: str = Field(min_length=1)
    newly_unlocked_skill_name: str = Field(min_length=1)
    missing_approved_content: bool = True
    current_mastery_probability: float = Field(ge=0.0, le=1.0)
    prerequisite_mastery_threshold: float = Field(ge=0.0, le=1.0)
    reason: Literal["missing_approved_content_for_unlocked_skill"] = (
        "missing_approved_content_for_unlocked_skill"
    )
    model_version: str = Field(min_length=1)
    policy_version: str = Field(min_length=1)


class ContentGapEvent(ContentGapResult):
    content_gap_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
