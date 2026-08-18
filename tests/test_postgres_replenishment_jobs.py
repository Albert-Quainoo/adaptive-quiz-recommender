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
from sqlalchemy.exc import DBAPIError

from authoring.replenishment.jobs import SchemaNotReadyError, SQLiteReplenishmentJobRepository
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


# ---------------------------------------------------------------------------
# read_only=True: schema-readiness check and DB-enforced read-only transaction --
# see authoring/replenishment/jobs.py's SQLiteReplenishmentJobRepository and
# scripts/run_replenishment_cycle.py's run_cycle(dry_run=True), the production
# caller this exists for.
# ---------------------------------------------------------------------------


def test_check_schema_ready_raises_schema_not_ready_before_any_initialization_on_postgres():
    """_clean_database (autouse above) already dropped the table -- a read-only
    repository must refuse to proceed, not silently create it."""
    read_only = SQLiteReplenishmentJobRepository(_dsn(), read_only=True)
    with pytest.raises(SchemaNotReadyError, match="schema_not_ready"):
        read_only.check_schema_ready()
    read_only.close()


def test_check_schema_ready_passes_once_a_writable_repository_has_initialized_on_postgres():
    writer = SQLiteReplenishmentJobRepository(_dsn())
    writer.initialize_schema()

    read_only = SQLiteReplenishmentJobRepository(_dsn(), read_only=True)
    read_only.check_schema_ready()  # must not raise
    writer.close()
    read_only.close()


def test_initialize_schema_refuses_on_a_read_only_repository_on_postgres():
    read_only = SQLiteReplenishmentJobRepository(_dsn(), read_only=True)
    with pytest.raises(RuntimeError, match="read_only"):
        read_only.initialize_schema()
    read_only.close()


def test_read_only_repository_reads_real_data_but_postgres_rejects_any_write():
    """The core guarantee: a read-only repository can genuinely scan inventory (real
    SELECTs against real rows), and PostgreSQL itself -- not just this class's own
    discipline -- rejects any write attempted through it, with zero rows changed."""
    writer = SQLiteReplenishmentJobRepository(_dsn())
    writer.initialize_schema()
    seeded = writer.enqueue(course_id="ai", skill_id="AI-PG-03", requested_count=2)

    read_only = SQLiteReplenishmentJobRepository(_dsn(), read_only=True)
    read_only.check_schema_ready()

    # Read: succeeds, returns the real row.
    fetched = read_only.get(seeded.job_id)
    assert fetched == seeded
    assert read_only.list_active(course_id="ai", skill_id="AI-PG-03") == [seeded]
    assert read_only.latest_for_skill("ai", "AI-PG-03") == seeded
    assert read_only.list_waiting() == []

    # Write: PostgreSQL itself rejects it -- not caught/prevented by application code,
    # a real ReadOnlySqlTransaction from the database (verified: DBAPIError, message
    # mentions "read-only").
    with pytest.raises(DBAPIError, match="read-only"):
        read_only.enqueue(course_id="ai", skill_id="AI-PG-04", requested_count=1)

    # Confirm zero rows changed: still exactly the one row the writer created.
    assert writer.list_active(course_id="ai") == [seeded]

    writer.close()
    read_only.close()
