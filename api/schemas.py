from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Literal


difficulty_level = Literal[
     "introductory",
     "intermediate",
     "advanced",
]

class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    correct_answer: str
    explanation: str
    concept: str
    difficulty: difficulty_level

    @model_validator(mode='after')
    def validate_correct_answer(self) -> 'QuizQuestion':
        if self.correct_answer not in self.options:
            raise ValueError('Correct answer is not in options')
        return self


    @field_validator('options')
    @classmethod
    def validate_unique_options_text(cls, options: list[str]) -> list[str]:
            normalized_texts = [option.strip().lower() for option in options]

            if len(normalized_texts) != len(set(normalized_texts)):
                raise ValueError("Duplicate option text content found")
            return options


class QuizResponse(BaseModel):
    questions: list[QuizQuestion] = Field(min_length=1, max_length=10)


class QuizGenerationRequest(BaseModel):
    topic: str = Field(min_length=1)
    difficulty: difficulty_level
    learning_objective: str = Field(min_length=1)
    question_count: int = Field(ge=1, le=10)

