from pydantic import BaseModel, Field, model_validator, field_validator
from typing import Literal,


class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    correct_answer: str
    explanation: str
    concept: str
    difficulty: str

    @model_validator(mode='after')
    def validate_correct_answer(self) -> 'QuizQuestion':
        if self.correct_answer not in self.options:
            raise ValueError('Correct answer is not in options')
        return self


    @field_validator('questions')
    @classmethod
        def validate_unique_options_text(cls, questions: list[QuizQuestion]) -> list[QuizQuestion]:
            for idx, question in enumerate(questions):
                normalized_texts = [option.strip().lower() for option in question.options]

                if len(normalized_texts) != len(set(normalized_texts)):
                    raise ValueError(f"Duplicate option text content found inside question at index {idx}")
            return questions


class QuizResponse(BaseModel):
    questions: list[QuizQuestion] = Field(min_length=1)


class QuizGenerationRequest(BaseModel):
    topic: str = Field(min_length=1)
    difficulty: Literal["introductory", "intermediate", "advanced"]
    learning_objectives: str = Field(min_length=1)
    question_count: int = Field(ge=1, le=10)

