from datetime import datetime, timezone

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AttemptEvent(BaseModel):
    """One scored learner response, in the order it was submitted.

    course_id is the explicit ownership/isolation boundary for this record
    -- not skill_id's course-code prefix. Every repository read/write must
    filter or supply it."""

    model_config = ConfigDict(frozen=True)

    attempt_id: str = Field(min_length=1)
    course_id: str = Field(min_length=1)
    presentation_id: str = Field(min_length=1)
    learner_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    skill_id: str = Field(min_length=1)
    selected_option_id: str = Field(min_length=1)
    correct: bool
    attempt_order: int = Field(ge=0)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator(
        "attempt_id",
        "course_id",
        "presentation_id",
        "learner_id",
        "item_id",
        "skill_id",
        "selected_option_id",
        mode="before",
    )
    @classmethod
    def strip_identifiers(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value


class MasterySnapshot(BaseModel):
    """The mastery state produced after processing an attempt.

    Identity is learner_id + course_id + skill_id -- the same skill_id can
    legitimately recur across two different courses (a fixture collision,
    or in principle a shared skill), and must not be conflated."""

    learner_id: str = Field(min_length=1)
    course_id: str = Field(min_length=1)
    skill_id: str = Field(min_length=1)
    mastery_probability: float = Field(ge=0.0, le=1.0)
    attempt_count: int = Field(ge=1)
    model_version: str = Field(min_length=1)
    source_attempt_id: str = Field(min_length=1)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator(
        "learner_id",
        "course_id",
        "skill_id",
        "model_version",
        "source_attempt_id",
        mode="before",
    )
    @classmethod
    def strip_identifiers(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value


class BKTModelMetadata(BaseModel):
    """Provenance for one fitted pyBKT model, owned by exactly one course."""

    course_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    fitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    training_attempt_count: int = Field(ge=1)
    skill_ids: list[str] = Field(min_length=1)

    @field_validator("course_id", "model_version", mode="before")
    @classmethod
    def strip_model_version(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value

    @field_validator("skill_ids")
    @classmethod
    def validate_skill_ids(cls, value: list[str]) -> list[str]:
        normalised = [skill_id.strip() for skill_id in value]
        if any(not skill_id for skill_id in normalised):
            raise ValueError("skill_ids cannot contain an empty identifier")
        if len(normalised) != len(set(normalised)):
            raise ValueError("skill_ids cannot contain duplicates")
        return normalised
