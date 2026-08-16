from datetime import datetime, timezone

import pytest

from authoring.course_catalog.lifecycle import (
    LifecycleError,
    advance_to_ready_and_activate,
    approve_course_definition,
    archive_course,
    reject_course_definition,
)
from authoring.course_catalog.readiness import ReadinessReport
from authoring.course_catalog.repository import SQLiteCourseApprovalRepository
from authoring.replenishment.manifest import CourseManifest

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def clock_at(when=T0):
    return lambda: when


def manifest_with_status(status: str, **overrides) -> CourseManifest:
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
        status=status,
    )
    fields.update(overrides)
    return CourseManifest(**fields)


@pytest.fixture
def repository():
    repository = SQLiteCourseApprovalRepository(":memory:")
    repository.initialize_schema()
    yield repository
    repository.close()


def test_approve_course_definition_moves_proposed_to_approved_for_preparation(repository):
    manifest = manifest_with_status("proposed")
    updated, record = approve_course_definition(
        manifest, repository, approver="op", clock=clock_at()
    )
    assert updated.status == "approved_for_preparation"
    assert record.decision == "approved"
    assert record.lifecycle_status == "approved_for_preparation"
    assert record.sequence_number == 1


def test_approve_course_definition_rejects_non_proposed_courses(repository):
    manifest = manifest_with_status("preparing")
    with pytest.raises(LifecycleError):
        approve_course_definition(manifest, repository, approver="op", clock=clock_at())
    assert repository.list_for_course("x") == []


def test_reject_course_definition_keeps_status_proposed(repository):
    manifest = manifest_with_status("proposed")
    record = reject_course_definition(
        manifest, repository, approver="op", reason="missing sources", clock=clock_at()
    )
    assert record.decision == "rejected"
    assert record.lifecycle_status == "proposed"


def test_reject_course_definition_rejects_non_proposed_courses(repository):
    manifest = manifest_with_status("active")
    with pytest.raises(LifecycleError):
        reject_course_definition(
            manifest, repository, approver="op", reason="x", clock=clock_at()
        )


def test_advance_to_ready_and_activate_noops_when_not_ready(repository):
    manifest = manifest_with_status("awaiting_content_approval")
    report = ReadinessReport(course_id="x", is_ready=False, blockers=["missing bank"])
    result = advance_to_ready_and_activate(manifest, report, repository, clock=clock_at())
    assert result is None
    assert repository.list_for_course("x") == []


def test_advance_to_ready_and_activate_noops_when_not_in_eligible_status(repository):
    manifest = manifest_with_status("preparing")
    report = ReadinessReport(course_id="x", is_ready=True, blockers=[])
    result = advance_to_ready_and_activate(manifest, report, repository, clock=clock_at())
    assert result is None


def test_advance_to_ready_and_activate_noops_when_auto_activate_disabled(repository):
    manifest = manifest_with_status(
        "awaiting_content_approval", auto_activate_when_ready=False
    )
    report = ReadinessReport(course_id="x", is_ready=True, blockers=[])
    result = advance_to_ready_and_activate(manifest, report, repository, clock=clock_at())
    assert result is None


def test_advance_to_ready_and_activate_transitions_and_writes_one_record(repository):
    manifest = manifest_with_status("awaiting_content_approval")
    report = ReadinessReport(
        course_id="x",
        is_ready=True,
        blockers=[],
        taxonomy_version="tv1",
        approved_bank_version="bv1",
        bkt_model_version="mv1",
    )
    updated, record = advance_to_ready_and_activate(
        manifest, report, repository, clock=clock_at()
    )
    assert updated.status == "active"
    assert record.decision == "activated"
    assert record.lifecycle_status == "active"
    assert record.taxonomy_version == "tv1"
    assert record.approved_bank_version == "bv1"
    assert record.bkt_model_version == "mv1"


def test_archive_course_moves_active_to_archived(repository):
    manifest = manifest_with_status("active")
    updated, record = archive_course(manifest, repository, approver="op", clock=clock_at())
    assert updated.status == "archived"
    assert record.decision == "archived"


def test_archive_course_rejects_non_active_courses(repository):
    manifest = manifest_with_status("preparing")
    with pytest.raises(LifecycleError):
        archive_course(manifest, repository, approver="op", clock=clock_at())


def test_full_lifecycle_produces_an_append_only_audit_trail(repository):
    manifest = manifest_with_status("proposed")
    manifest, _ = approve_course_definition(
        manifest, repository, approver="op", clock=clock_at()
    )
    manifest = manifest.model_copy(update={"status": "awaiting_content_approval"})
    report = ReadinessReport(course_id="x", is_ready=True, blockers=[])
    manifest, _ = advance_to_ready_and_activate(
        manifest, report, repository, clock=clock_at()
    )
    before_archive = repository.list_for_course("x")
    archive_course(manifest, repository, approver="op", clock=clock_at())
    after_archive = repository.list_for_course("x")

    assert len(after_archive) == 3
    assert [record.sequence_number for record in after_archive] == [1, 2, 3]
    # every record present before archiving is unchanged after it
    for record in before_archive:
        assert record in after_archive
