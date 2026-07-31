import json
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from api.model_loader import get_model_id, load_model, load_tokenizer
from api.quiz_generator import generate_raw_quiz
from api.response_parser import parse_quiz_messages
from evaluation.baseline_cases import BASELINE_CASES
from evaluation.schemas import EvaluationCase, EvaluationResult

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def create_output_path() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    return RESULTS_DIR / f"baseline_{timestamp}.jsonl"


def evaluate_case(case: EvaluationCase, model_id: str) -> EvaluationResult:
    start_time = time.perf_counter()

    try:
        raw_response = generate_raw_quiz(
            case.request,
            max_new_tokens=case.max_new_tokens,
        )
    except Exception as error:  # noqa: BLE001 - record the failure and keep evaluating later cases
        return EvaluationResult(
            case_id=case.case_id,
            model_id=model_id,
            request=case.request,
            status="generation_error",
            latency_seconds=time.perf_counter() - start_time,
            error_type=type(error).__name__,
            error_message=str(error),
        )

    try:
        quiz = parse_quiz_messages(raw_response)
    except json.JSONDecodeError as error:
        return EvaluationResult(
            case_id=case.case_id,
            model_id=model_id,
            request=case.request,
            status="json_error",
            latency_seconds=time.perf_counter() - start_time,
            json_valid=False,
            raw_response=raw_response,
            error_type=type(error).__name__,
            error_message=str(error),
        )
    except ValidationError as error:
        return EvaluationResult(
            case_id=case.case_id,
            model_id=model_id,
            request=case.request,
            status="schema_error",
            latency_seconds=time.perf_counter() - start_time,
            json_valid=True,
            schema_valid=False,
            raw_response=raw_response,
            error_type=type(error).__name__,
            error_message=str(error),
        )

    generated_count = len(quiz.questions)
    count_valid = generated_count == case.request.question_count

    return EvaluationResult(
        case_id=case.case_id,
        model_id=model_id,
        request=case.request,
        status="success" if count_valid else "count_error",
        latency_seconds=time.perf_counter() - start_time,
        json_valid=True,
        schema_valid=True,
        count_valid=count_valid,
        generated_count=generated_count,
        raw_response=raw_response,
        validated_response=quiz,
    )


def main() -> None:
    load_tokenizer()
    load_model()

    model_id = get_model_id()
    output_path = create_output_path()

    with output_path.open("w", encoding="utf-8") as result_file:
        for index, case in enumerate(BASELINE_CASES, start=1):
            result = evaluate_case(case, model_id)

            result_file.write(result.model_dump_json() + "\n")
            result_file.flush()

            print(
                f"[{index}/{len(BASELINE_CASES)}] {case.case_id}: "
                f"{result.status} - {result.latency_seconds:.2f} seconds"
            )

    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
