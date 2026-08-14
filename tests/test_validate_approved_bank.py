import json

import pytest

from api.bank import BankItem
from api.schemas import QuizQuestion
from scripts.validate_approved_bank import validate_bank


def item(item_id: str, skill_id: str, stem: str) -> BankItem:
    return BankItem(
        item_id=item_id,
        skill_id=skill_id,
        provenance="generated",
        question=QuizQuestion(
            question=stem,
            options=["Correct", "Wrong A", "Wrong B", "Wrong C"],
            correct_answer="Correct",
            explanation="The approved sources support the correct answer.",
            concept="Approved concept",
            difficulty="introductory",
        ),
    )


def write(path, items):
    path.write_text(
        "".join(
            json.dumps(value.model_dump(mode="json"), sort_keys=True) + "\n"
            for value in items
        ),
        encoding="utf-8",
    )


def test_readiness_requires_unique_items_stems_and_required_skill_coverage(tmp_path):
    bank = tmp_path / "bank.jsonl"
    write(
        bank,
        [
            item("one", "AI-FND-01", "First approved question?"),
            item("two", "AI-AGT-01", "Second approved question?"),
        ],
    )

    summary = validate_bank(
        bank,
        course="ai",
        expected_count=2,
        required_skill_ids=["AI-FND-01", "AI-AGT-01"],
    )

    assert summary["status"] == "ready"
    assert summary["unique_item_ids"] == 2
    assert summary["unique_normalized_stems"] == 2


def test_readiness_rejects_duplicate_normalized_stems(tmp_path):
    bank = tmp_path / "bank.jsonl"
    write(
        bank,
        [
            item("one", "AI-FND-01", "Same question?"),
            item("two", "AI-FND-01", "  SAME   QUESTION? "),
        ],
    )

    with pytest.raises(ValueError, match="duplicate normalized stems"):
        validate_bank(
            bank,
            course="ai",
            expected_count=2,
            required_skill_ids=["AI-FND-01"],
        )
