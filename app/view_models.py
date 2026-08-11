"""Answer-free data contracts passed to the Streamlit UI."""

from pydantic import BaseModel, ConfigDict, Field

from api.schemas import difficulty_level


RECOMMENDATION_REASON_TEXT = {
    "foundational_unseen_skill": "Start with a foundational skill you have not practised yet.",
    "lowest_mastery_eligible_skill": "Practise the eligible skill with the most room to grow.",
    "requested_difficulty_used": "This question matches the requested difficulty.",
    "fallback_difficulty_used": "The nearest available difficulty was selected.",
}


def readable_recommendation_reason(reason: str | None) -> str:
    if reason is None:
        return "No recommendation has been made yet."
    return RECOMMENDATION_REASON_TEXT.get(
        reason, reason.replace("_", " ").capitalize() + "."
    )


class QuestionOptionViewModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    option_id: str = Field(min_length=1)
    text: str = Field(min_length=1)


class QuestionViewModel(BaseModel):
    """A presented question. The answer is deliberately absent."""

    model_config = ConfigDict(frozen=True)

    presentation_id: str = Field(min_length=1)
    item_id: str = Field(min_length=1)
    skill_id: str = Field(min_length=1)
    topic: str = Field(min_length=1)
    concept: str = Field(min_length=1)
    difficulty: difficulty_level
    question_text: str = Field(min_length=1)
    options: list[QuestionOptionViewModel] = Field(min_length=1)
    recommendation_reason: str = Field(min_length=1)


class SubmissionResultViewModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    attempt_id: str = Field(min_length=1)
    correct: bool
    correct_option_id: str = Field(min_length=1)
    correct_answer_text: str = Field(min_length=1)
    explanation: str = Field(min_length=1)
    previous_mastery: float = Field(ge=0.0, le=1.0)
    updated_mastery: float = Field(ge=0.0, le=1.0)
    attempt_count: int = Field(ge=1)


class ProgressViewModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    learner_id: str = Field(min_length=1)
    skill_id: str = Field(min_length=1)
    skill_name: str = Field(min_length=1)
    mastery_probability: float = Field(ge=0.0, le=1.0)
    attempt_count: int = Field(ge=0)
    difficulty: difficulty_level
    readable_recommendation_reason: str = Field(min_length=1)
