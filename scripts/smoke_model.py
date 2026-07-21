import torch

from api.quiz_generator import generate_raw_quiz
from api.schemas import QuizGenerationRequest
from api.response_parser import parse_quiz_messages

def main() -> None:
    request = QuizGenerationRequest(
        topic="Queues",
        difficulty="introductory",
        learning_objective="Identify basic queue operations",
        question_count=1,
    )

    raw_response = generate_raw_quiz(request)
    quiz = parse_quiz_messages(raw_response)

    if len(quiz.questions) != request.question_count:
        raise ValueError(
            f"Expected {request.question_count} questions, "
            f"but received {len(quiz.questions)} questions."

        )
    
    print(quiz.model_dump_json(indent=2))

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3

        print(f"\nGPU memory allocated: {allocated:.2f} GB")
        print(f"GPU memory reserved: {reserved:.2f} GB")


if __name__ == "__main__":
    main()
