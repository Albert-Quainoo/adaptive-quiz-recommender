import pytest
from pydantic import ValidationError

from authoring.replenishment.manifest import (
    CourseManifest,
    ManifestError,
    load_active_manifests,
    load_course_manifest,
)


def test_load_course_manifest_reads_the_intro_ai_manifest():
    manifest = load_course_manifest("intro-ai")
    assert manifest.course_id == "intro-ai"
    assert manifest.status == "active"
    assert manifest.low_supply_threshold > 0
    assert manifest.target_supply >= manifest.low_supply_threshold
    assert manifest.allowed_domains


def test_load_course_manifest_is_case_insensitive():
    assert load_course_manifest("INTRO-AI").course_id == "intro-ai"


def test_load_course_manifest_missing_course_raises():
    with pytest.raises(ManifestError):
        load_course_manifest("does-not-exist")


def test_load_active_manifests_only_returns_active_courses():
    manifests = load_active_manifests()
    assert all(manifest.status == "active" for manifest in manifests)
    assert any(manifest.course_id == "intro-ai" for manifest in manifests)


def test_skill_and_reference_paths_are_derived_from_taxonomy_path():
    manifest = load_course_manifest("intro-ai")
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
        bkt_model_path="outputs/bkt_dev_model_v4.pkl",
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


def test_max_session_questions_defaults_to_unset():
    manifest = CourseManifest(**_fields())
    assert manifest.max_session_questions is None


def test_max_session_questions_must_be_positive():
    with pytest.raises(ValidationError):
        CourseManifest(**_fields(max_session_questions=0))


def test_max_session_questions_can_be_set_explicitly():
    manifest = CourseManifest(**_fields(max_session_questions=20))
    assert manifest.max_session_questions == 20


def test_intro_ai_manifest_leaves_session_length_unbounded():
    manifest = load_course_manifest("intro-ai")
    assert manifest.max_session_questions is None
