import pytest
from pydantic import ValidationError

from api.schemas import(
    QuizGenerationRequest,
    QuizQuestion,
    QuizResponse,
)

from evaluation.schemas import EvaluationResult
from evaluation.schemas import EvaluationCase
from evaluation.schemas import HumanReview, HumanReviewRecord

def test_valid_generation_request():
    request = QuizGenerationRequest(
        topic="Stacks",
        difficulty="intermediate",
        learning_objective="Identify intermediate stack operations",
        question_count=2,
    )

    assert request.topic == "Stacks"
    assert request.question_count == 2

def test_invalid_difficulty_is_rejected():
    with pytest.raises(ValidationError):
        QuizGenerationRequest(
            topic="Stacks",
            difficulty="easy",
            learning_objective="Identify easy stack operations",
            question_count=4
        )

@pytest.mark.parametrize("question_count", [0, 11])
def test_question_count_outside_range_is_rejected(question_count):
    with pytest.raises(ValidationError):
        QuizGenerationRequest(
            topic="Queues",
            difficulty="introductory",
            learning_objective="Identify queue operations",
            question_count=question_count,
        )

def test_duplicate_options_are_rejected():
    with pytest.raises(ValidationError):
        QuizQuestion(
            question="Which queue operation removes the front element?",
            options=["Push", "Pop", "Dequeue", "Pop"],
            correct_answer="Dequeue",
            explanation="Dequeue removes the element at the front of a queue.",
            concept="Queue operations",
            difficulty="introductory"
        )

def test_correct_answer_must_be_an_option():
    with pytest.raises(ValidationError):
        QuizQuestion(
            question="Which queue operation removes the front element?",
            options=["Push", "Pop", "Dequeue", "Peek"],
            correct_answer="Enqueue",
            explanation="nothing, since there is no answer",
            concept="Queue operations",
            difficulty="introductory"
        )


def valid_question_data() -> dict:
    return {
        "question": "Which operation removes the front element?",
        "options": ["Push", "Pop", "Dequeue", "Peek"],
        "correct_answer": "Dequeue",
        "explanation": "Dequeue removes the front element.",
        "concept": "Queue operations",
        "difficulty": "introductory",
    }

def test_valid_quiz_response():
    data = valid_question_data()
    quiz_question_one = QuizQuestion(**data)
    response = QuizResponse(
        questions=[quiz_question_one]
    )

    assert len(response.questions) == 1
    assert response.questions[0].correct_answer == "Dequeue"


@pytest.mark.parametrize("invalid_count", [-5, 0, 11, 100])
def test_invalid_question_counts_are_rejected(invalid_count):
    with pytest.raises(ValidationError):
        QuizGenerationRequest(
            topic="Testing",
            difficulty="intermediate",
            learning_objective="Pytest",
            question_count=invalid_count
        )

def test_evaluation_result_tracks_prompt_version():
    case = EvaluationCase(
        case_id="TEST_001",
        objective_type="conceptual",
        request=QuizGenerationRequest(
            topic="Stacks",
            difficulty="introductory",
            learning_objective="Identify stack operations.",
            question_count=1,
        ),
        max_new_tokens=1200,
    )

    result = EvaluationResult(
        case_id=case.case_id,
        model_id="test-model",
        prompt_version="v2",
        request=case.request,
        objective_type=case.objective_type,
        pipeline_status="generation_error",
        latency_seconds=0.1,
        max_new_tokens=case.max_new_tokens,
        error_type="TestError",
        error_message="Intentional test failure",
    )

    assert result.prompt_version == "v2"
    assert result.model_dump()["prompt_version"] == "v2"

def test_evaluation_result_requires_prompt_version():
    case = EvaluationCase(
        case_id="TEST_001",
        objective_type="conceptual",
        request=QuizGenerationRequest(
            topic="Stacks",
            difficulty="introductory",
            learning_objective="Identify stack operations.",
            question_count=1,
        ),
        max_new_tokens=1200,
    )

    with pytest.raises(ValidationError, match="prompt_version"):
        EvaluationResult(
            case_id=case.case_id,
            model_id="test-model",
            request=case.request,
            pipeline_status="generation_error",
            latency_seconds=0.1,
            max_new_tokens=case.max_new_tokens,
            error_type="TestError",
            error_message="Intentional test failure",
        )


def valid_evaluation_result(**overrides) -> EvaluationResult:
    fields = {
        "case_id": "TEST_001",
        "model_id": "test-model",
        "prompt_version": "v3.1",
        "request": QuizGenerationRequest(
            topic="Stacks",
            difficulty="introductory",
            learning_objective="Identify stack operations.",
            question_count=1,
        ),
        "objective_type": "conceptual",
        "pipeline_status": "valid",
        "latency_seconds": 0.1,
        "max_new_tokens": 1200,
    }
    fields.update(overrides)
    return EvaluationResult(**fields)


def test_human_review_defaults_to_unreviewed():
    review = HumanReview()

    assert review.quality_status == "not_reviewed"
    assert review.factually_correct is None
    assert review.objective_aligned is None
    assert review.difficulty_appropriate is None
    assert review.requires_target_operation is None
    assert review.avoids_reference_regurgitation is None
    assert review.reviewer_notes is None


def test_invalid_quality_status_is_rejected():
    with pytest.raises(ValidationError, match="quality_status"):
        HumanReview(quality_status="maybe")


def test_invalid_pipeline_status_is_rejected():
    with pytest.raises(ValidationError, match="pipeline_status"):
        valid_evaluation_result(pipeline_status="success")


def test_evaluation_result_human_review_is_optional():
    assert valid_evaluation_result().human_review is None


def test_evaluation_result_carries_human_review_through_serialisation():
    result = valid_evaluation_result(
        human_review=HumanReview(
            quality_status="rejected",
            factually_correct=False,
            objective_aligned=True,
            reviewer_notes="Distractor is also correct.",
        )
    )

    dumped = result.model_dump()
    assert dumped["human_review"]["quality_status"] == "rejected"
    assert dumped["human_review"]["factually_correct"] is False
    assert dumped["human_review"]["difficulty_appropriate"] is None

    assert EvaluationResult.model_validate_json(result.model_dump_json()) == result


def test_human_review_record_links_run_and_case():
    record = HumanReviewRecord(
        run_id="baseline_20260804_040615",
        case_id="TEST_001",
        review=HumanReview(quality_status="approved", factually_correct=True),
    )

    assert record.run_id == "baseline_20260804_040615"
    assert record.review.quality_status == "approved"


@pytest.mark.parametrize("missing_field", ["run_id", "case_id", "review"])
def test_human_review_record_requires_all_fields(missing_field):
    fields = {
        "run_id": "baseline_20260804_040615",
        "case_id": "TEST_001",
        "review": HumanReview(),
    }
    del fields[missing_field]

    with pytest.raises(ValidationError, match=missing_field):
        HumanReviewRecord(**fields)



