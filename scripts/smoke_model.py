import torch

from api.quiz_generator import generate_raw_quiz
from api.schemas import QuizGenerationRequest

def main() -> None:
    request = QuizGenerationRequest(
        topic="Queues",
        difficulty="introductory",
        learning_objective="Identify basic queue operations",
        question_count=1,
    )

    response = generate_raw_quiz(request)
    
    print("\nGenerated response:\n")
    print(response)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3

        print(f"\nGPU memory allocated: {allocated:.2f} GB")
        print(f"GPU memory reserved: {reserved:.2f} GB")


if __name__ == "__main__":
    main()
