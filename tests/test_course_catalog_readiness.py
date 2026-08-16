from datetime import datetime, timezone

import pandas as pd
import pytest

from authoring.course_catalog import readiness as readiness_module
from authoring.course_catalog.readiness import inspect_readiness
from authoring.grounded_review import CurationItem, GroundedReview
from authoring.replenishment.manifest import CourseManifest

SKILL_ID = "XX-PLC-01"


def _manifest_fields(tmp_path, **overrides):
    taxonomy_dir = tmp_path / "taxonomy"
    taxonomy_dir.mkdir(exist_ok=True)
    fields = dict(
        course_id="x",
        title="X",
        version="1",
        taxonomy_path=taxonomy_dir,
        approved_bank_path=tmp_path / "bank.jsonl",
        bkt_model_path=tmp_path / "model.pkl",
        candidate_store_path=tmp_path / "candidates.json",
        review_store_path=tmp_path / "reviews",
        allowed_domains=("example.edu",),
        low_supply_threshold=3,
        target_supply=6,
        default_bkt_model_version="v1",
        status="awaiting_content_approval",
    )
    fields.update(overrides)
    return fields


def _write_taxonomy(taxonomy_dir):
    (taxonomy_dir / "skills.csv").write_text(
        "skill_id,topic,subtopic,name,learning_objective,cognitive_process,"
        "generation_strategy,template_id,prerequisite_skill_ids\n"
        f"{SKILL_ID},Topic,Subtopic,Name,Objective,remember,hand_authored,,\n",
        encoding="utf-8",
    )
    (taxonomy_dir / "references.csv").write_text(
        "skill_id,reference_material\n", encoding="utf-8"
    )


def _write_bank(bank_path):
    item = {
        "item_id": "x-item-1",
        "provenance": "generated",
        "skill_id": SKILL_ID,
        "question": {
            "question": "What is placeholder?",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "explanation": "Because.",
            "concept": "Placeholder",
            "difficulty": "introductory",
        },
    }
    bank_path.parent.mkdir(parents=True, exist_ok=True)
    bank_path.write_text(__import__("json").dumps(item) + "\n", encoding="utf-8")


def _write_approved_review(review_store_path):
    review_store_path.mkdir(parents=True, exist_ok=True)
    review = GroundedReview(
        batch_id="b1",
        source_hashes={},
        items=[
            CurationItem(
                original_question_id="q1",
                skill_id=SKILL_ID,
                intent_id="i1",
                recommendation="approve_as_written",
                recommendation_reason="looks correct",
                final_review_status="approved",
                reviewed_by="reviewer-1",
                reviewed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        ],
    )
    (review_store_path / "b1__x.json").write_text(review.model_dump_json(), encoding="utf-8")


class _FakeFittedModel:
    def __init__(self, skill_id: str, prior: float) -> None:
        index = pd.MultiIndex.from_tuples(
            [(skill_id, "prior"), (skill_id, "learns")], names=["skill", "param"]
        )
        self._frame = pd.DataFrame({"value": [prior, 0.3]}, index=index)

    def get_parameters(self) -> pd.DataFrame:
        return self._frame


def _patch_fitted_model(monkeypatch, *, skill_id=SKILL_ID, prior=0.20):
    monkeypatch.setattr(
        readiness_module,
        "load_fitted_bkt_model",
        lambda path, *, model_version, course_id: _FakeFittedModel(skill_id, prior),
    )


@pytest.fixture
def ready_manifest(tmp_path, monkeypatch):
    fields = _manifest_fields(tmp_path)
    _write_taxonomy(fields["taxonomy_path"])
    _write_bank(fields["approved_bank_path"])
    _write_approved_review(fields["review_store_path"])
    fields["bkt_model_path"].touch()
    _patch_fitted_model(monkeypatch)
    return CourseManifest(**fields)


def test_fully_satisfied_course_reports_ready_with_no_blockers(ready_manifest):
    report = inspect_readiness(ready_manifest)
    assert report.is_ready is True
    assert report.blockers == []
    assert report.taxonomy_version is not None
    assert report.approved_bank_version is not None
    assert report.bkt_model_version == "v1"


def test_missing_bank_file_is_reported_not_raised(tmp_path, monkeypatch):
    fields = _manifest_fields(tmp_path)
    _write_taxonomy(fields["taxonomy_path"])
    _write_approved_review(fields["review_store_path"])
    fields["bkt_model_path"].touch()
    _patch_fitted_model(monkeypatch)
    manifest = CourseManifest(**fields)

    report = inspect_readiness(manifest)
    assert report.is_ready is False
    assert any("approved bank" in blocker for blocker in report.blockers)


def test_unknown_skill_in_bank_is_reported_not_raised(tmp_path, monkeypatch):
    fields = _manifest_fields(tmp_path)
    _write_taxonomy(fields["taxonomy_path"])
    fields["approved_bank_path"].parent.mkdir(parents=True, exist_ok=True)
    bad_item = {
        "item_id": "x-item-2",
        "provenance": "generated",
        "skill_id": "X-UNKNOWN-01",
        "question": {
            "question": "What is unknown?",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "explanation": "Because.",
            "concept": "Unknown",
            "difficulty": "introductory",
        },
    }
    fields["approved_bank_path"].write_text(
        __import__("json").dumps(bad_item) + "\n", encoding="utf-8"
    )
    _write_approved_review(fields["review_store_path"])
    fields["bkt_model_path"].touch()
    _patch_fitted_model(monkeypatch)
    manifest = CourseManifest(**fields)

    report = inspect_readiness(manifest)
    assert report.is_ready is False
    assert any("coverage" in blocker for blocker in report.blockers)


def test_missing_model_file_is_reported_not_raised(tmp_path, monkeypatch):
    fields = _manifest_fields(tmp_path)
    _write_taxonomy(fields["taxonomy_path"])
    _write_bank(fields["approved_bank_path"])
    _write_approved_review(fields["review_store_path"])
    # bkt_model_path deliberately never created
    manifest = CourseManifest(**fields)

    report = inspect_readiness(manifest)
    assert report.is_ready is False
    assert any("BKT model artifact is missing" in blocker for blocker in report.blockers)


def test_pending_review_item_blocks_readiness(tmp_path, monkeypatch):
    fields = _manifest_fields(tmp_path)
    _write_taxonomy(fields["taxonomy_path"])
    _write_bank(fields["approved_bank_path"])
    fields["review_store_path"].mkdir(parents=True, exist_ok=True)
    review = GroundedReview(
        batch_id="b1",
        source_hashes={},
        items=[
            CurationItem(
                original_question_id="q1",
                skill_id=SKILL_ID,
                intent_id="i1",
                recommendation="approve_as_written",
                recommendation_reason="looks correct",
            )
        ],
    )
    (fields["review_store_path"] / "b1__x.json").write_text(
        review.model_dump_json(), encoding="utf-8"
    )
    fields["bkt_model_path"].touch()
    _patch_fitted_model(monkeypatch)
    manifest = CourseManifest(**fields)

    report = inspect_readiness(manifest)
    assert report.is_ready is False
    assert any("awaiting human review" in blocker for blocker in report.blockers)


def test_rejected_review_item_does_not_block_readiness(tmp_path, monkeypatch):
    """Course approval must never override individual question-review
    decisions -- a rejection is a resolved decision, not an open blocker."""
    fields = _manifest_fields(tmp_path)
    _write_taxonomy(fields["taxonomy_path"])
    _write_bank(fields["approved_bank_path"])
    fields["review_store_path"].mkdir(parents=True, exist_ok=True)
    review = GroundedReview(
        batch_id="b1",
        source_hashes={},
        items=[
            CurationItem(
                original_question_id="q1",
                skill_id=SKILL_ID,
                intent_id="i1",
                recommendation="reject",
                recommendation_reason="not accurate",
                final_review_status="rejected",
                reviewed_by="reviewer-1",
                reviewed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
                rejection_reason="factually incorrect",
            )
        ],
    )
    (fields["review_store_path"] / "b1__x.json").write_text(
        review.model_dump_json(), encoding="utf-8"
    )
    fields["bkt_model_path"].touch()
    _patch_fitted_model(monkeypatch)
    manifest = CourseManifest(**fields)

    report = inspect_readiness(manifest)
    assert not any("awaiting human review" in blocker for blocker in report.blockers)


def test_model_prior_mismatch_is_reported_not_raised(tmp_path, monkeypatch):
    fields = _manifest_fields(tmp_path)
    _write_taxonomy(fields["taxonomy_path"])
    _write_bank(fields["approved_bank_path"])
    _write_approved_review(fields["review_store_path"])
    fields["bkt_model_path"].touch()
    _patch_fitted_model(monkeypatch, prior=0.5)
    manifest = CourseManifest(**fields)

    report = inspect_readiness(manifest, initial_mastery_probability=0.20)
    assert report.is_ready is False
    assert any("prior does not match" in blocker for blocker in report.blockers)
