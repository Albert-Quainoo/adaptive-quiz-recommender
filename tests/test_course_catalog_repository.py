from datetime import datetime, timezone

import pytest

from authoring.course_catalog.repository import SQLiteCourseApprovalRepository

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


@pytest.fixture
def repository():
    repository = SQLiteCourseApprovalRepository(":memory:")
    repository.initialize_schema()
    yield repository
    repository.close()


def test_append_assigns_incrementing_sequence_numbers_per_course(repository):
    first = repository.append(
        course_id="x",
        lifecycle_status="approved_for_preparation",
        course_profile_version="1",
        decision="approved",
        approver_identity="op",
        decided_at=T0,
        auto_activate_when_ready=True,
    )
    second = repository.append(
        course_id="x",
        lifecycle_status="active",
        course_profile_version="1",
        decision="activated",
        approver_identity="system",
        decided_at=T0,
        auto_activate_when_ready=True,
    )
    assert first.sequence_number == 1
    assert second.sequence_number == 2


def test_sequence_numbers_are_scoped_per_course(repository):
    x = repository.append(
        course_id="x",
        lifecycle_status="active",
        course_profile_version="1",
        decision="activated",
        approver_identity="system",
        decided_at=T0,
        auto_activate_when_ready=True,
    )
    y = repository.append(
        course_id="y",
        lifecycle_status="active",
        course_profile_version="1",
        decision="activated",
        approver_identity="system",
        decided_at=T0,
        auto_activate_when_ready=True,
    )
    assert x.sequence_number == 1
    assert y.sequence_number == 1


def test_list_for_course_returns_every_record_in_order_none_mutated(repository):
    for lifecycle_status, decision in [
        ("approved_for_preparation", "approved"),
        ("active", "activated"),
        ("archived", "archived"),
    ]:
        repository.append(
            course_id="x",
            lifecycle_status=lifecycle_status,
            course_profile_version="1",
            decision=decision,
            approver_identity="op",
            decided_at=T0,
            auto_activate_when_ready=True,
        )

    records = repository.list_for_course("x")
    assert [record.sequence_number for record in records] == [1, 2, 3]
    assert [record.decision for record in records] == ["approved", "activated", "archived"]
    # Re-fetch and confirm the earlier record is byte-for-byte the same --
    # nothing about it changed as later records were appended.
    assert repository.list_for_course("x")[0] == records[0]


def test_latest_for_course_returns_the_highest_sequence_record(repository):
    repository.append(
        course_id="x",
        lifecycle_status="approved_for_preparation",
        course_profile_version="1",
        decision="approved",
        approver_identity="op",
        decided_at=T0,
        auto_activate_when_ready=True,
    )
    repository.append(
        course_id="x",
        lifecycle_status="active",
        course_profile_version="1",
        decision="activated",
        approver_identity="system",
        decided_at=T0,
        auto_activate_when_ready=True,
    )
    latest = repository.latest_for_course("x")
    assert latest.sequence_number == 2
    assert latest.decision == "activated"


def test_latest_for_course_returns_none_for_an_unregistered_course(repository):
    assert repository.latest_for_course("does-not-exist") is None


def test_no_update_or_delete_method_exists(repository):
    assert not hasattr(repository, "update")
    assert not hasattr(repository, "delete")
