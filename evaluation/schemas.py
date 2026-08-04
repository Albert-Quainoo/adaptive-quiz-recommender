from pydantic import BaseModel, Field
from typing import Literal
from api.schemas import QuizGenerationRequest, QuizResponse

PipelineStatus = Literal[
    "valid",
    "json_error",
    "schema_error",
    "count_error",
    "generation_error"
]

objective_type = Literal[
    "conceptual",
    "calculation",
]

QualityStatus = Literal[
    "not_reviewed",
    "approved",
    "rejected",
]

class HumanReview(BaseModel):
    quality_status: QualityStatus = "not_reviewed"
    factually_correct: bool | None = None
    objective_aligned: bool | None = None
    difficulty_appropriate: bool | None = None
    requires_target_operation: bool | None = None
    avoids_reference_regurgitation: bool | None = None
    reviewer_notes: str | None = None

class HumanReviewRecord(BaseModel):
    run_id: str
    case_id: str
    review: HumanReview

class EvaluationCase(BaseModel):
    case_id: str
    request: QuizGenerationRequest
    objective_type: objective_type
    max_new_tokens: int | None = Field(default=None,gt=0)

class EvaluationResult(BaseModel):
    case_id: str
    model_id: str
    prompt_version: str
    request: QuizGenerationRequest
    objective_type: objective_type
    pipeline_status: PipelineStatus
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
    human_review: HumanReview | None = None
