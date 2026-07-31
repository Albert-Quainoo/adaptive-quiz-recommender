from pydantic import BaseModel, Field
from typing import Literal
from api.schemas import QuizGenerationRequest, QuizResponse

EvaluationStatus = Literal[
    "success",
    "json_error",
    "schema_error",
    "count_error",
    "generation_error"
]

class EvaluationCase(BaseModel):
    case_id: str
    request: QuizGenerationRequest
    max_new_tokens: int = Field(default=1200,gt=0)

class EvaluationResult(BaseModel):
    case_id: str
    model_id: str
    request: QuizGenerationRequest
    status: EvaluationStatus
    latency_seconds: float = Field(ge=0)
    json_valid: bool | None = None
    schema_valid: bool | None = None
    count_valid: bool | None = None
    generated_count: int | None = Field(default=None, ge=0)
    raw_response: str | None = None
    validated_response: QuizResponse | None = None
    error_type: str | None = None
    error_message: str | None = None
    max_new_tokens: int
