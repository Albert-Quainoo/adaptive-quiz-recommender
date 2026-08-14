import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from api.bank import BankItem
from api.schemas import QuizQuestion
from authoring.grounded_review import assert_immutable_source, question_content_hash
from authoring.pilot_curation_v3 import (
    PROPOSED_QUESTIONS,
    UNCHANGED_INTENTS,
    approve_all,
    approved_bank_items,
    build_review,
    load_sources,
)


BATCH_PATH = Path("outputs/grounded-pilot-20260811-v3")
EDITED_AT = datetime(2026, 8, 11, 16, 0, tzinfo=timezone.utc)


def test_review_has_exact_human_review_inventory_and_pending_status():
    review = build_review(BATCH_PATH, edited_at=EDITED_AT)

    assert len(review.items) == 24
    assert len(PROPOSED_QUESTIONS) == 14
    assert len(UNCHANGED_INTENTS) == 10
    assert {item.intent_id for item in review.items if item.revisions} == set(PROPOSED_QUESTIONS)
    assert {item.intent_id for item in review.items if not item.revisions} == UNCHANGED_INTENTS
    assert all(item.final_review_status == "pending" for item in review.items)
    assert all(
        revision.final_review_status == "pending"
        for item in review.items
        for revision in item.revisions
    )


def test_revisions_validate_and_preserve_source_provenance():
    review = build_review(BATCH_PATH, edited_at=EDITED_AT)
    sources = {source.question_id: source for source in load_sources(BATCH_PATH)}

    for item in review.items:
        if not item.revisions:
            continue
        source = sources[item.original_question_id]
        revision = item.revisions[0]
        QuizQuestion.model_validate(revision.question.model_dump())
        BankItem(
            item_id=revision.revision_id,
            skill_id=source.skill_id,
            provenance="generated",
            question=revision.question,
        )
        assert revision.source_batch_id == source.batch_id
        assert revision.intent_id == source.intent_id
        assert revision.skill_id == source.skill_id
        assert revision.reference_ids == source.reference_ids
        assert revision.model_id == source.model_id
        assert revision.model_revision == source.model_revision
        assert revision.prompt_version == source.prompt_version
        assert revision.prompt_hash == source.prompt_hash
        assert revision.changed_fields
        assert revision.content_hash == question_content_hash(revision.question)


def test_source_is_immutable_and_final_distribution_is_unchanged():
    review = build_review(BATCH_PATH, edited_at=EDITED_AT)
    assert_immutable_source(BATCH_PATH, review)
    sources = load_sources(BATCH_PATH)
    counts = Counter((source.skill_id, source.question.difficulty) for source in sources)

    assert counts == Counter(
        {
            ("AI-FND-01", "intermediate"): 3,
            ("AI-AGT-01", "introductory"): 3,
            ("AI-AGT-01", "intermediate"): 3,
            ("AI-SRC-01", "introductory"): 3,
            ("AI-SRC-02", "introductory"): 3,
            ("AI-SRC-03", "introductory"): 3,
            ("AI-SRC-03", "intermediate"): 3,
            ("AI-SRC-08", "introductory"): 3,
        }
    )


def test_original_generated_records_are_not_revision_targets(tmp_path):
    before = {
        path.name: path.read_bytes()
        for path in BATCH_PATH.iterdir()
        if path.is_file()
    }
    review = build_review(BATCH_PATH, edited_at=EDITED_AT)
    after = {
        path.name: path.read_bytes()
        for path in BATCH_PATH.iterdir()
        if path.is_file()
    }

    assert before == after
    assert all(item.original_question_id not in {revision.revision_id for revision in item.revisions} for item in review.items)


def test_all_items_can_be_approved_without_revising_unchanged_sources():
    review = build_review(BATCH_PATH, edited_at=EDITED_AT)
    approved = approve_all(review, reviewer="albert", reviewed_at=EDITED_AT)
    sources = load_sources(BATCH_PATH)
    items = approved_bank_items(approved, sources)
    source_by_intent = {source.intent_id: source for source in sources}

    assert len(items) == 24
    assert len({item.item_id for item in items}) == 24
    assert all(item.final_review_status == "approved" for item in approved.items)
    assert all(item.reviewed_by == "albert" for item in approved.items)
    assert all(item.reviewed_at == EDITED_AT for item in approved.items)
    for curation, bank_item in zip(approved.items, items, strict=True):
        if curation.intent_id in UNCHANGED_INTENTS:
            source = source_by_intent[curation.intent_id]
            assert curation.revisions == []
            assert bank_item.item_id == source.question_id
            assert bank_item.question == source.question
        else:
            revision = curation.revisions[0]
            assert revision.final_review_status == "approved"
            assert revision.reviewed_by == "albert"
            assert revision.reviewed_at == EDITED_AT
            assert bank_item.item_id == revision.revision_id
            assert bank_item.question == revision.question
