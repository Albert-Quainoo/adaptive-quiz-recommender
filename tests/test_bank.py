import pytest
from pydantic import ValidationError

from api.bank import BankItem
from api.prompt_builder import build_quiz_messages
from api.schemas import QuizGenerationRequest, QuizQuestion


def valid_question() -> QuizQuestion:
    return QuizQuestion(
        question="Which queue operation removes the front element?",
        options=["Push", "Pop", "Dequeue", "Peek"],
        correct_answer="Dequeue",
        explanation="Dequeue removes the element at the front of a queue.",
        concept="Queue operations",
        difficulty="introductory",
    )


def test_bank_item_records_provenance():
    item = BankItem(question=valid_question(), provenance="templated")

    assert item.provenance == "templated"
    assert item.skill_id is None


def test_unknown_provenance_is_rejected():
    with pytest.raises(ValidationError):
        BankItem(question=valid_question(), provenance="imported")


def test_provenance_is_not_requested_from_the_model():
    request = QuizGenerationRequest(
        topic="Stacks",
        difficulty="introductory",
        learning_objective="Stack operations",
        question_count=2,
    )

    system_content = build_quiz_messages(request)[0]["content"]

    assert "provenance" not in system_content
    assert "skill_id" not in system_content
