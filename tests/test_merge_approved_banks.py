import json

import pytest

from api.bank import BankItem
from api.schemas import QuizQuestion
from app.bootstrap import load_approved_bank
from scripts.merge_approved_banks import merge_approved_banks


def item(item_id, skill_id):
    return BankItem(
        item_id=item_id,
        skill_id=skill_id,
        provenance="generated",
        question=QuizQuestion(
            question=f"Question for {skill_id}?",
            options=["Correct", "Wrong A", "Wrong B", "Wrong C"],
            correct_answer="Correct",
            explanation="Grounded explanation.",
            concept=skill_id,
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


def test_merge_preserves_approved_items_and_source_files(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    output = tmp_path / "versioned" / "merged.jsonl"
    write(first, [item("one", "skill-a")])
    write(second, [item("two", "skill-b")])
    sources = (first.read_bytes(), second.read_bytes())

    merge_approved_banks([first, second], output)

    assert [value.item_id for value in load_approved_bank(output)] == ["one", "two"]
    assert sources == (first.read_bytes(), second.read_bytes())


def test_merge_rejects_duplicate_item_ids(tmp_path):
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"
    write(first, [item("duplicate", "skill-a")])
    write(second, [item("duplicate", "skill-b")])

    with pytest.raises(ValueError, match="duplicate"):
        merge_approved_banks([first, second], tmp_path / "merged.jsonl")
