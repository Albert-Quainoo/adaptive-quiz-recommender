from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


class AttemptEvent(BaseModel):
    """One scored learner response, in the order it was submitted."""

    attempt_id: str = Field(min_length=1)
    learner_id: str = Field(min_length=1)
    skill_id: str = Field(min_length=1)
    correct: bool
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("attempt_id", "learner_id", "skill_id", mode="before")
    @classmethod
    def strip_identifiers(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value


class MasterySnapshot(BaseModel):
    """The mastery state produced after processing an attempt."""

    learner_id: str = Field(min_length=1)
    skill_id: str = Field(min_length=1)
    mastery_probability: float = Field(ge=0.0, le=1.0)
    model_version: str = Field(min_length=1)
    source_attempt_id: str = Field(min_length=1)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator(
        "learner_id", "skill_id", "model_version", "source_attempt_id", mode="before"
    )
    @classmethod
    def strip_identifiers(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value


class BKTModelMetadata(BaseModel):
    """Provenance for one fitted pyBKT model."""

    model_version: str = Field(min_length=1)
    fitted_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    training_attempt_count: int = Field(ge=1)
    skill_ids: list[str] = Field(min_length=1)

    @field_validator("model_version", mode="before")
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
