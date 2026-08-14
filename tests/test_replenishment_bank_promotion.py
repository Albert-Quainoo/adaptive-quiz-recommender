"""Promotion safety: versioned output, rollback preservation, dedup, atomicity."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from app.bootstrap import load_approved_bank
from authoring.replenishment.jobs import SQLiteReplenishmentJobRepository
from authoring.replenishment.manifest import (
    CourseManifest,
    active_bank_pointer_path,
    load_course_manifest,
)
from authoring.replenishment.worker import _handle_promote_approved_items

FIXED_TIME = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)


def fixed_clock():
    return FIXED_TIME


def _bank_item(item_id: str, skill_id: str = "AI-SRC-08") -> dict:
    return {
        "item_id": item_id,
        "skill_id": skill_id,
        "provenance": "generated",
        "question": {
            "question": f"What does a heuristic estimate? ({item_id})",
            "options": ["a", "b", "c", "d"],
            "correct_answer": "a",
            "explanation": "because",
            "concept": "Heuristics",
            "difficulty": "intermediate",
        },
    }


def _write_review(path: Path, item_ids: list[str]) -> None:
    from authoring.grounded_review import CurationItem, GroundedReview, QuestionRevision
    from api.schemas import QuizQuestion

    items = []
    for item_id in item_ids:
        question = QuizQuestion.model_validate(_bank_item(item_id)["question"])
        revision = QuestionRevision(
            original_question_id=item_id,
            revision_id=item_id,
            editor="albert",
            edited_at=FIXED_TIME,
            changed_fields=["explanation"],
            review_note="ok",
            source_batch_id="b1",
            intent_id="AI-SRC-08-INT-01",
            skill_id="AI-SRC-08",
            reference_ids=["AI-SRC-08-ref1"],
            model_id="fake-model",
            model_revision="fake-rev",
            prompt_version="v3.3",
            prompt_hash="a" * 64,
            final_review_status="approved",
            reviewed_by="albert",
            reviewed_at=FIXED_TIME,
            question=question,
        )
        items.append(
            CurationItem(
                original_question_id=item_id,
                skill_id="AI-SRC-08",
                intent_id="AI-SRC-08-INT-01",
                recommendation="propose_revision",
                recommendation_reason="test",
                final_review_status="approved",
                reviewed_by="albert",
                reviewed_at=FIXED_TIME,
                revisions=[revision],
            )
        )
    review = GroundedReview(batch_id="b1", source_hashes={}, items=items)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(review.model_dump(mode="json")), encoding="utf-8")


def manifest_at(tmp_path, **overrides) -> CourseManifest:
    taxonomy_dir = tmp_path / "taxonomy"
    taxonomy_dir.mkdir(exist_ok=True)
    (taxonomy_dir / "skills.csv").write_text("skill_id,topic,subtopic,name,learning_objective,cognitive_process,generation_strategy,template_id,prerequisite_skill_ids\n", encoding="utf-8")
    (taxonomy_dir / "references.csv").write_text("skill_id,reference_material\n", encoding="utf-8")
    fields = dict(
        course_id="ai", title="t", version="1", taxonomy_path=taxonomy_dir,
        approved_bank_path=tmp_path / "bank" / "ai-bank-v0.jsonl",
        candidate_store_path=tmp_path / "candidates.json",
        review_store_path=tmp_path / "reviews",
        allowed_domains=("example.edu",), low_supply_threshold=3, target_supply=6,
        default_bkt_model_version="v1", status="active",
    )
    fields.update(overrides)
    return CourseManifest(**fields)


@pytest.fixture
def repository():
    repository = SQLiteReplenishmentJobRepository(":memory:")
    repository.initialize_schema()
    yield repository
    repository.close()


def _promote_job(repository, manifest, review_path):
    job = repository.enqueue(course_id="ai", skill_id="AI-SRC-08", requested_count=1, clock=fixed_clock)
    job = repository.claim_next(clock=fixed_clock)
    job = repository.mark_waiting(
        job.job_id, "waiting_for_question_review", job_type="generate_questions",
        metadata={"review_path": str(review_path)},
    )
    repository.mark_queued(job.job_id, job_type="promote_approved_items")
    return repository.claim_next(clock=fixed_clock)


def test_promotion_writes_a_new_versioned_bank_and_pointer(tmp_path, repository):
    manifest = manifest_at(tmp_path)
    review_path = tmp_path / "reviews" / "b1.json"
    _write_review(review_path, ["AI-SRC-08-q1"])
    job = _promote_job(repository, manifest, review_path)

    _handle_promote_approved_items(job, manifest, job_repository=repository, clock=fixed_clock)

    pointer = json.loads(active_bank_pointer_path(manifest).read_text())
    assert pointer["version"] == 1
    bank_path = Path(pointer["path"])
    assert bank_path.name == "ai-approved-bank-v1.jsonl"
    items = load_approved_bank(bank_path)
    assert [item.item_id for item in items] == ["AI-SRC-08-q1"]
    assert repository.get(job.job_id).status == "completed"


def test_second_promotion_chains_from_the_pointer_and_preserves_the_first(tmp_path, repository):
    manifest = manifest_at(tmp_path)
    review_path_1 = tmp_path / "reviews" / "b1.json"
    _write_review(review_path_1, ["AI-SRC-08-q1"])
    job1 = _promote_job(repository, manifest, review_path_1)
    _handle_promote_approved_items(job1, manifest, job_repository=repository, clock=fixed_clock)
    v1_pointer = json.loads(active_bank_pointer_path(manifest).read_text())
    v1_path = Path(v1_pointer["path"])
    v1_contents_before = v1_path.read_text()

    review_path_2 = tmp_path / "reviews" / "b2.json"
    _write_review(review_path_2, ["AI-SRC-08-q2"])
    job2 = _promote_job(repository, manifest, review_path_2)
    _handle_promote_approved_items(job2, manifest, job_repository=repository, clock=fixed_clock)

    v2_pointer = json.loads(active_bank_pointer_path(manifest).read_text())
    assert v2_pointer["version"] == 2
    v2_items = load_approved_bank(Path(v2_pointer["path"]))
    assert {item.item_id for item in v2_items} == {"AI-SRC-08-q1", "AI-SRC-08-q2"}

    # v1 file itself was never touched by the second promotion.
    assert v1_path.read_text() == v1_contents_before
    assert manifest.approved_bank_path.exists() is False  # nothing was ever promoted onto it


def test_duplicate_item_id_across_bank_and_new_items_is_rejected(tmp_path, repository):
    manifest = manifest_at(tmp_path)
    manifest.approved_bank_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.approved_bank_path.write_text(
        json.dumps(_bank_item("AI-SRC-08-q1")) + "\n", encoding="utf-8"
    )
    review_path = tmp_path / "reviews" / "b1.json"
    _write_review(review_path, ["AI-SRC-08-q1"])  # same id as the existing bank
    job = _promote_job(repository, manifest, review_path)

    with pytest.raises(ValueError, match="duplicate"):
        _handle_promote_approved_items(job, manifest, job_repository=repository, clock=fixed_clock)


def test_no_approved_items_is_a_permanent_failure(tmp_path, repository):
    manifest = manifest_at(tmp_path)
    review_path = tmp_path / "reviews" / "b1.json"
    _write_review(review_path, [])
    job = _promote_job(repository, manifest, review_path)

    _handle_promote_approved_items(job, manifest, job_repository=repository, clock=fixed_clock)
    after = repository.get(job.job_id)
    assert after.status == "permanent_failure"
    assert after.error_code == "no_approved_items"


def test_pointer_write_is_atomic_temp_then_replace(tmp_path, repository):
    manifest = manifest_at(tmp_path)
    review_path = tmp_path / "reviews" / "b1.json"
    _write_review(review_path, ["AI-SRC-08-q1"])
    job = _promote_job(repository, manifest, review_path)
    _handle_promote_approved_items(job, manifest, job_repository=repository, clock=fixed_clock)

    pointer_path = active_bank_pointer_path(manifest)
    assert not pointer_path.with_suffix(pointer_path.suffix + ".tmp").exists()
    assert pointer_path.is_file()


def test_the_real_38_item_bank_is_never_touched_by_this_test_suite():
    manifest = load_course_manifest("ai")
    before = manifest.approved_bank_path.read_bytes()
    items = load_approved_bank(manifest.approved_bank_path)
    assert len(items) == 38
    after = manifest.approved_bank_path.read_bytes()
    assert before == after
