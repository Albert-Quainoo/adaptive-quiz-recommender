from api.schemas import QuizQuestion, QuizResponse


def generate_mock_quiz() -> QuizResponse:
    return QuizResponse(
        questions=[
            QuizQuestion(
                question="Which queue operation removes the front element?",
                options=["Push", "Pop", "Dequeue", "Peek"],
                correct_answer="Dequeue",
                explanation="Dequeue removes the element at the front of a queue.",
                concept="Queue operations",
                difficulty="introductory",
            )
        ]
    )
