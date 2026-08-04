import json
import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import ValidationError

from api.model_loader import get_model_id, load_model, load_tokenizer
from api.quiz_generator import generate_raw_quiz, token_budget
from api.response_parser import parse_quiz_messages
from evaluation.baseline_cases import BASELINE_CASES
from api.prompt_builder import PROMPT_VERSION
from evaluation.schemas import EvaluationCase, EvaluationResult


RESULTS_DIR = Path(__file__).resolve().parent / "results"

def create_run_id() -> str:
    return datetime.now(timezone.utc).strftime("baseline_%Y%m%d_%H%M%S")


def create_output_path(run_id: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    return RESULTS_DIR / f"{run_id}.jsonl"


def evaluate_case(case: EvaluationCase, model_id: str, run_id: str) -> EvaluationResult:

    max_new_tokens = case.max_new_tokens or token_budget(case.request.question_count)

    result_metadata = {
    "run_id": run_id,
    "case_id": case.case_id,
    "model_id": model_id,
    "prompt_version": PROMPT_VERSION,
    "request": case.request,
    "objective_type": case.objective_type,
    }

    start_time = time.perf_counter()

    try:
        raw_response = generate_raw_quiz(
            case.request,
            max_new_tokens=max_new_tokens,
        )
    except Exception as error:  # noqa: BLE001 - record the failure and keep evaluating later cases
        return EvaluationResult(
            **result_metadata,
            pipeline_status="generation_error",
            latency_seconds=time.perf_counter() - start_time,
            error_type=type(error).__name__,
            error_message=str(error),
            max_new_tokens=max_new_tokens,
        )

    try:
        quiz = parse_quiz_messages(raw_response)
    except json.JSONDecodeError as error:
        return EvaluationResult(
            **result_metadata,
            pipeline_status="json_error",
            latency_seconds=time.perf_counter() - start_time,
            json_valid=False,
            raw_response=raw_response,
            error_type=type(error).__name__,
            error_message=str(error),
            max_new_tokens=max_new_tokens,
        )
    except ValidationError as error:
        return EvaluationResult(
            **result_metadata,
            pipeline_status="schema_error",
            latency_seconds=time.perf_counter() - start_time,
            json_valid=True,
            schema_valid=False,
            raw_response=raw_response,
            error_type=type(error).__name__,
            error_message=str(error),
            max_new_tokens=max_new_tokens,
        )

    generated_count = len(quiz.questions)
    count_valid = generated_count == case.request.question_count

    if count_valid: 
        pipeline_status = "valid"
        error_type = None
        error_message = None
    else:
        pipeline_status = "count_error"
        error_type = "QuestionCountMismatch"
        error_message = (
        f"Requested {case.request.question_count} questions, "
        f"but generated {generated_count}."
    )

    return EvaluationResult(
        **result_metadata,
        max_new_tokens=max_new_tokens,
        pipeline_status=pipeline_status,
        latency_seconds=time.perf_counter() - start_time,
        json_valid=True,
        schema_valid=True,
        count_valid=count_valid,
        generated_count=generated_count,
        raw_response=raw_response,
        validated_response=quiz,
        error_type=error_type,
        error_message=error_message,
)

def main() -> None:
    load_tokenizer()
    load_model()

    model_id = get_model_id()
    run_id = create_run_id()
    output_path = create_output_path(run_id)

    cases = BASELINE_CASES
    results: list[EvaluationResult] = []

    with output_path.open("w", encoding="utf-8") as result_file:
        for index, case in enumerate(cases, start=1):
            result = evaluate_case(case, model_id, run_id)
            results.append(result)

            result_file.write(result.model_dump_json() + "\n")
            result_file.flush()

            print(
                f"[{index}/{len(cases)}] {case.case_id}: "
                f"{result.pipeline_status} - {result.latency_seconds:.2f} seconds"
            )

    print()
    for objective in ("conceptual", "calculation"):
        matching = [result for result in results if result.objective_type == objective]
        succeeded = sum(result.pipeline_status == "valid" for result in matching)
        print(
            f"{objective}: "
            f"{succeeded}/{len(matching)} pipeline-valid"
        )

    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
