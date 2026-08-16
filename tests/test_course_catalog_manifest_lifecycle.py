import pytest
from pydantic import ValidationError

from authoring.replenishment.manifest import (
    CourseManifest,
    load_active_manifests,
    load_all_manifests,
    load_preparation_eligible_manifests,
)


def _fields(**overrides):
    fields = dict(
        course_id="x",
        title="X",
        version="1",
        taxonomy_path="taxonomy/data/x",
        approved_bank_path="outputs/x-bank.jsonl",
        bkt_model_path="outputs/x-model.pkl",
        candidate_store_path="outputs/x-candidates.json",
        review_store_path="outputs/x-reviews",
        allowed_domains=("example.edu",),
        low_supply_threshold=3,
        target_supply=6,
        default_bkt_model_version="v1",
        status="proposed",
    )
    fields.update(overrides)
    return fields


@pytest.mark.parametrize(
    "status",
    [
        "proposed",
        "approved_for_preparation",
        "preparing",
        "awaiting_content_approval",
        "ready",
        "active",
        "archived",
    ],
)
def test_manifest_status_accepts_every_lifecycle_state(status):
    manifest = CourseManifest(**_fields(status=status))
    assert manifest.status == status


def test_manifest_status_rejects_the_old_binary_literal():
    with pytest.raises(ValidationError):
        CourseManifest(**_fields(status="inactive"))


def test_aliases_and_auto_activate_round_trip():
    manifest = CourseManifest(
        **_fields(aliases=("X", "Ecks"), auto_activate_when_ready=False)
    )
    assert manifest.aliases == ("X", "Ecks")
    assert manifest.auto_activate_when_ready is False


def test_aliases_default_to_empty_and_auto_activate_defaults_true():
    manifest = CourseManifest(**_fields())
    assert manifest.aliases == ()
    assert manifest.auto_activate_when_ready is True


def test_bkt_model_path_is_required():
    fields = _fields()
    del fields["bkt_model_path"]
    with pytest.raises(ValidationError):
        CourseManifest(**fields)


def test_load_active_manifests_only_returns_active():
    manifests = load_active_manifests()
    assert manifests
    assert all(manifest.status == "active" for manifest in manifests)
    assert any(manifest.course_id == "intro-ai" for manifest in manifests)


def test_load_preparation_eligible_manifests_excludes_proposed_and_archived():
    manifests = load_preparation_eligible_manifests()
    statuses = {manifest.status for manifest in manifests}
    assert "proposed" not in statuses
    assert "archived" not in statuses
    course_ids = {manifest.course_id for manifest in manifests}
    assert {"intro-ai", "dsa", "linear-algebra", "database-systems"} <= course_ids


def test_load_all_manifests_returns_every_registered_course_regardless_of_status():
    course_ids = {manifest.course_id for manifest in load_all_manifests()}
    assert {"intro-ai", "dsa", "linear-algebra", "database-systems"} <= course_ids
