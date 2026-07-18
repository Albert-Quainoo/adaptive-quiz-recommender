from pydantic import BaseModel, Field


class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(min_length=4, max_length=4)
    correct_answer: str
    explanation: str
    concept: str
    difficulty: str


class QuizResponse(BaseModel):
    questions: list[QuizQuestion]
