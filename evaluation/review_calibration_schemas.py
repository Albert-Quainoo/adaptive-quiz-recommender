"""Typed shapes for the automated-review calibration harness."""

from typing import Literal

from pydantic import BaseModel, Field

from api.schemas import QuizQuestion

CalibrationLabel = Literal["positive", "negative"]


class CalibrationCase(BaseModel):
    case_id: str = Field(min_length=1)
    label: CalibrationLabel
    skill_id: str = Field(min_length=1)
    intent_id: str = Field(min_length=1)
    question: QuizQuestion
    mutation_type: str | None = None


class CalibrationCaseResult(BaseModel):
    case_id: str
    label: CalibrationLabel
    mutation_type: str | None
    recommendation: str
    risk_level: str
    reviewer_calls: int
    parser_failure: bool
    disagreement: bool


class CalibrationReport(BaseModel):
    total_cases: int
    positive_cases: int
    negative_cases: int
    approval_precision: float
    approval_recall: float
    critical_error_detection_rate: float
    false_low_risk_rate: float
    disagreement_rate: float
    parser_failure_rate: float
    reviewer_calls: int
    estimated_cost_usd: float
    results: list[CalibrationCaseResult]
