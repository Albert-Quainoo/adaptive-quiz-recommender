"""PostgreSQL integration smoke coverage for the course-approval audit trail.

Skipped locally unless QUIZ_TEST_POSTGRES_URL is set; GitHub Actions always
sets it against the postgres service container. Never touches the Supabase
DSN configured in .streamlit/secrets.toml -- exclusively QUIZ_TEST_POSTGRES_URL.
"""

import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from authoring.course_catalog.repository import SQLiteCourseApprovalRepository
from tests.postgres_test_safety import DSN_ENV_VAR, require_safe_postgres_target

pytestmark = pytest.mark.skipif(
    not os.getenv(DSN_ENV_VAR),
    reason=f"set {DSN_ENV_VAR} to a PostgreSQL DSN to run these integration tests",
)


def _dsn() -> str:
    return os.environ[DSN_ENV_VAR]


@pytest.fixture(autouse=True)
def _clean_database():
    require_safe_postgres_target(_dsn())
    engine_owner = SQLiteCourseApprovalRepository(_dsn())
    with engine_owner._engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS course_approval_records CASCADE"))
    engine_owner.close()
    yield


def test_schema_initialization_is_idempotent_on_postgres():
    repository = SQLiteCourseApprovalRepository(_dsn())
    repository.initialize_schema()
    repository.initialize_schema()
    repository.close()


def test_append_list_and_latest_round_trip_on_postgres():
    dsn = _dsn()
    repository = SQLiteCourseApprovalRepository(dsn)
    repository.initialize_schema()

    first = repository.append(
        course_id="x",
        lifecycle_status="approved_for_preparation",
        course_profile_version="1",
        decision="approved",
        approver_identity="op",
        decided_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
        auto_activate_when_ready=True,
    )
    second = repository.append(
        course_id="x",
        lifecycle_status="active",
        course_profile_version="1",
        decision="activated",
        approver_identity="system",
        decided_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
        auto_activate_when_ready=True,
    )

    assert first.sequence_number == 1
    assert second.sequence_number == 2
    assert [record.sequence_number for record in repository.list_for_course("x")] == [1, 2]
    assert repository.latest_for_course("x") == second
    repository.close()
