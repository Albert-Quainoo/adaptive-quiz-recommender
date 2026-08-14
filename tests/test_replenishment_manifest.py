import pytest
from pydantic import ValidationError

from authoring.replenishment.manifest import (
    CourseManifest,
    ManifestError,
    load_active_manifests,
    load_course_manifest,
)


def test_load_course_manifest_reads_the_ai_manifest():
    manifest = load_course_manifest("ai")
    assert manifest.course_id == "ai"
    assert manifest.status == "active"
    assert manifest.low_supply_threshold > 0
    assert manifest.target_supply >= manifest.low_supply_threshold
    assert manifest.allowed_domains


def test_load_course_manifest_is_case_insensitive():
    assert load_course_manifest("AI").course_id == "ai"


def test_load_course_manifest_missing_course_raises():
    with pytest.raises(ManifestError):
        load_course_manifest("does-not-exist")


def test_load_active_manifests_only_returns_active_courses():
    manifests = load_active_manifests()
    assert all(manifest.status == "active" for manifest in manifests)
    assert any(manifest.course_id == "ai" for manifest in manifests)


def test_skill_and_reference_paths_are_derived_from_taxonomy_path():
    manifest = load_course_manifest("ai")
    assert manifest.skills_path() == manifest.taxonomy_path / "skills.csv"
    assert manifest.references_path() == manifest.taxonomy_path / "references.csv"
    assert manifest.provenance_path() == manifest.taxonomy_path / "reference_provenance.csv"


def _fields(**overrides):
    fields = dict(
        course_id="ai",
        title="Introduction to AI",
        version="1",
        taxonomy_path="taxonomy/data/ai",
        approved_bank_path="outputs/approved_banks/pilot-approved-bank-38-v1.jsonl",
        candidate_store_path="outputs/reference_candidates.json",
        review_store_path="outputs/replenishment/ai/reviews",
        allowed_domains=("example.edu",),
        low_supply_threshold=3,
        target_supply=6,
        default_bkt_model_version="v1",
        status="active",
    )
    fields.update(overrides)
    return fields


def test_target_supply_must_be_at_least_low_supply_threshold():
    with pytest.raises(ValidationError):
        CourseManifest(**_fields(low_supply_threshold=6, target_supply=3))


def test_thresholds_must_be_positive():
    with pytest.raises(ValidationError):
        CourseManifest(**_fields(low_supply_threshold=0))


def test_allowed_domains_cannot_be_empty():
    with pytest.raises(ValidationError):
        CourseManifest(**_fields(allowed_domains=()))
