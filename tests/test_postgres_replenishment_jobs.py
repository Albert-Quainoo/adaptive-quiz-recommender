"""PostgreSQL integration smoke coverage for the replenishment job queue.

Skipped locally unless QUIZ_TEST_POSTGRES_URL is set; GitHub Actions always
sets it against the postgres service container. See
tests/test_postgres_adaptive_repository.py for the equivalent coverage of
the learner-facing repositories.
"""

import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from authoring.replenishment.jobs import SQLiteReplenishmentJobRepository

DSN_ENV_VAR = "QUIZ_TEST_POSTGRES_URL"

pytestmark = pytest.mark.skipif(
    not os.getenv(DSN_ENV_VAR),
    reason=f"set {DSN_ENV_VAR} to a PostgreSQL DSN to run these integration tests",
)


def _dsn() -> str:
    return os.environ[DSN_ENV_VAR]


@pytest.fixture(autouse=True)
def _clean_database():
    engine_owner = SQLiteReplenishmentJobRepository(_dsn())
    with engine_owner._engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS replenishment_jobs CASCADE"))
    engine_owner.close()
    yield


def clock_at(when: datetime):
    return lambda: when


def test_schema_initialization_is_idempotent_on_postgres():
    repository = SQLiteReplenishmentJobRepository(_dsn())
    repository.initialize_schema()
    repository.initialize_schema()
    repository.close()


def test_enqueue_get_and_list_roundtrip_on_postgres():
    dsn = _dsn()
    repository = SQLiteReplenishmentJobRepository(dsn)
    repository.initialize_schema()
    job = repository.enqueue(course_id="ai", skill_id="AI-PG-01", requested_count=3)

    reopened = SQLiteReplenishmentJobRepository(dsn)
    fetched = reopened.get(job.job_id)
    assert fetched == job
    assert reopened.list_active(course_id="ai", skill_id="AI-PG-01") == [job]
    repository.close()
    reopened.close()


def test_active_job_partial_unique_index_prevents_duplicates_on_postgres():
    repository = SQLiteReplenishmentJobRepository(_dsn())
    repository.initialize_schema()
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)

    first = repository.enqueue(
        course_id="ai", skill_id="AI-PG-01", requested_count=1, clock=clock_at(now)
    )
    second = repository.enqueue(
        course_id="ai", skill_id="AI-PG-01", requested_count=1, clock=clock_at(now)
    )
    assert second.job_id == first.job_id  # returned the existing active job, not a duplicate row

    repository.mark_completed(first.job_id)
    third = repository.enqueue(
        course_id="ai", skill_id="AI-PG-01", requested_count=1, clock=clock_at(now)
    )
    assert third.job_id != first.job_id  # a new active job is allowed once the old one is done
    repository.close()


def test_claim_next_and_lease_recovery_on_postgres():
    repository = SQLiteReplenishmentJobRepository(_dsn())
    repository.initialize_schema()
    repository.enqueue(course_id="ai", skill_id="AI-PG-02", requested_count=1)

    claimed = repository.claim_next(lease_seconds=0)
    assert claimed.status == "running"

    recovered = repository.recover_expired_leases(
        clock=clock_at(datetime.now(timezone.utc))
    )
    assert recovered == 1
    assert repository.get(claimed.job_id).status == "queued"
    repository.close()
