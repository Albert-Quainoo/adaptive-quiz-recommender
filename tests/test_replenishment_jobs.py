from datetime import datetime, timedelta, timezone

import pytest

from authoring.replenishment.jobs import ACTIVE_STATUSES, JobType, SQLiteReplenishmentJobRepository
from database import execute_schema_script

T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def clock_at(when=T0):
    return lambda: when


@pytest.fixture
def repository():
    repository = SQLiteReplenishmentJobRepository(":memory:")
    repository.initialize_schema()
    yield repository
    repository.close()


def test_enqueue_creates_a_queued_replenish_skill_job(repository):
    job = repository.enqueue(
        course_id="ai", skill_id="AI-SRC-01", requested_count=6, clock=clock_at()
    )
    assert job.status == "queued"
    assert job.job_type == "replenish_skill"
    assert job.attempts == 0
    assert job.created_at == T0


def test_enqueue_is_idempotent_for_the_same_active_job(repository):
    first = repository.enqueue(
        course_id="ai", skill_id="AI-SRC-01", requested_count=6, clock=clock_at()
    )
    second = repository.enqueue(
        course_id="ai", skill_id="AI-SRC-01", requested_count=9, clock=clock_at()
    )
    assert first.job_id == second.job_id
    assert second.requested_count == 6  # unchanged: the original job wins


def test_enqueue_allows_a_new_job_once_the_old_one_is_terminal(repository):
    first = repository.enqueue(
        course_id="ai", skill_id="AI-SRC-01", requested_count=6, clock=clock_at()
    )
    repository.mark_permanent_failure(first.job_id, error_code="x", error_message="y")
    second = repository.enqueue(
        course_id="ai", skill_id="AI-SRC-01", requested_count=6, clock=clock_at()
    )
    assert second.job_id != first.job_id


def test_enqueue_is_scoped_per_skill(repository):
    a = repository.enqueue(
        course_id="ai", skill_id="AI-SRC-01", requested_count=6, clock=clock_at()
    )
    b = repository.enqueue(
        course_id="ai", skill_id="AI-SRC-02", requested_count=6, clock=clock_at()
    )
    assert a.job_id != b.job_id


def test_claim_next_atomically_claims_the_oldest_queued_job(repository):
    repository.enqueue(course_id="ai", skill_id="AI-SRC-01", requested_count=6, clock=clock_at())
    later = clock_at(T0 + timedelta(seconds=1))
    repository.enqueue(course_id="ai", skill_id="AI-SRC-02", requested_count=6, clock=later)

    claimed = repository.claim_next(clock=clock_at())
    assert claimed.skill_id == "AI-SRC-01"
    assert claimed.status == "running"
    assert claimed.attempts == 1
    assert claimed.lease_expires_at is not None

    second_claim = repository.claim_next(clock=clock_at())
    assert second_claim.skill_id == "AI-SRC-02"

    assert repository.claim_next(clock=clock_at()) is None


def test_recover_expired_leases_resets_abandoned_running_jobs(repository):
    repository.enqueue(course_id="ai", skill_id="AI-SRC-01", requested_count=6, clock=clock_at())
    claimed = repository.claim_next(lease_seconds=10, clock=clock_at())

    still_leased = clock_at(T0 + timedelta(seconds=5))
    assert repository.recover_expired_leases(clock=still_leased) == 0

    expired = clock_at(T0 + timedelta(seconds=20))
    assert repository.recover_expired_leases(clock=expired) == 1
    assert repository.get(claimed.job_id).status == "queued"
    assert repository.get(claimed.job_id).lease_expires_at is None


def test_retryable_failure_escalates_to_permanent_after_max_attempts(repository):
    repository.enqueue(course_id="ai", skill_id="AI-SRC-01", requested_count=6, clock=clock_at())

    job_id = repository.claim_next(clock=clock_at()).job_id
    first = repository.mark_retryable_failure(
        job_id, error_code="e", error_message="m", max_attempts=2, clock=clock_at()
    )
    assert first.status == "retryable_failure"
    assert first.next_retry_at is not None

    repository.requeue_retryable(clock=clock_at(T0 + timedelta(days=1)))
    repository.claim_next(clock=clock_at())
    second = repository.mark_retryable_failure(
        job_id, error_code="e", error_message="m", max_attempts=2, clock=clock_at()
    )
    assert second.status == "permanent_failure"


def test_requeue_retryable_only_moves_jobs_past_their_backoff(repository):
    repository.enqueue(course_id="ai", skill_id="AI-SRC-01", requested_count=6, clock=clock_at())
    job_id = repository.claim_next(clock=clock_at()).job_id
    repository.mark_retryable_failure(
        job_id, error_code="e", error_message="m", backoff_seconds=100, max_attempts=5, clock=clock_at()
    )

    too_soon = clock_at(T0 + timedelta(seconds=1))
    assert repository.requeue_retryable(clock=too_soon) == 0
    assert repository.get(job_id).status == "retryable_failure"

    later = clock_at(T0 + timedelta(seconds=200))
    assert repository.requeue_retryable(clock=later) == 1
    assert repository.get(job_id).status == "queued"


def test_mark_waiting_updates_status_job_type_and_metadata(repository):
    repository.enqueue(course_id="ai", skill_id="AI-SRC-01", requested_count=6, clock=clock_at())
    job = repository.claim_next(clock=clock_at())
    updated = repository.mark_waiting(
        job.job_id,
        "waiting_for_reference_review",
        job_type="retrieve_references",
        metadata={"pending_candidates": 3},
    )
    assert updated.status == "waiting_for_reference_review"
    assert updated.job_type == "retrieve_references"
    assert updated.metadata == {"pending_candidates": 3}
    assert updated.lease_expires_at is None


def test_mark_completed_and_cancel(repository):
    repository.enqueue(course_id="ai", skill_id="AI-SRC-01", requested_count=6, clock=clock_at())
    job = repository.claim_next(clock=clock_at())
    completed = repository.mark_completed(job.job_id, clock=clock_at())
    assert completed.status == "completed"
    assert completed.completed_at == T0

    repository.enqueue(course_id="ai", skill_id="AI-SRC-02", requested_count=6, clock=clock_at())
    other = repository.claim_next(clock=clock_at())
    cancelled = repository.cancel(other.job_id)
    assert cancelled.status == "cancelled"


def test_list_active_and_latest_for_skill(repository):
    repository.enqueue(course_id="ai", skill_id="AI-SRC-01", requested_count=6, clock=clock_at())
    assert {job.status for job in repository.list_active()} <= set(ACTIVE_STATUSES)

    latest = repository.latest_for_skill("ai", "AI-SRC-01")
    assert latest is not None
    assert repository.latest_for_skill("ai", "AI-SRC-99") is None


# enqueue()'s duplicate detection is scoped by (course_id, skill_id) only -- job_type is
# excluded, so these regression tests holding job_type fixed at "automated_review" on
# both the mark_waiting call and the re-enqueue attempt exercise the status-list-drift
# fix in isolation from the job_type-drift fix covered by the tests further below.
@pytest.mark.parametrize(
    "status", ["waiting_for_full_human_review", "rejected_by_automated_review"]
)
def test_enqueue_is_idempotent_while_parked_in_a_new_waiting_status(repository, status):
    first = repository.enqueue(
        course_id="ai",
        skill_id="AI-SRC-08",
        job_type="automated_review",
        requested_count=6,
        clock=clock_at(),
    )
    repository.mark_waiting(first.job_id, status, job_type="automated_review")

    second = repository.enqueue(
        course_id="ai",
        skill_id="AI-SRC-08",
        job_type="automated_review",
        requested_count=6,
        clock=clock_at(),
    )
    assert second.job_id == first.job_id


def test_initialize_schema_migrates_an_index_created_under_the_old_status_list():
    repository = SQLiteReplenishmentJobRepository(":memory:")
    # Recreate the pre-automated-review index by hand, as it would exist on an
    # already-deployed database file, before calling the real initialize_schema().
    with repository._engine.begin() as connection:
        execute_schema_script(
            connection,
            """
            CREATE TABLE IF NOT EXISTS replenishment_jobs (
                job_id TEXT PRIMARY KEY, course_id TEXT NOT NULL, skill_id TEXT NOT NULL,
                job_type TEXT NOT NULL, status TEXT NOT NULL, requested_count INTEGER NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL, started_at TEXT,
                completed_at TEXT, next_retry_at TEXT, lease_expires_at TEXT, error_code TEXT,
                error_message TEXT, metadata_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE UNIQUE INDEX IF NOT EXISTS ux_replenishment_jobs_active
            ON replenishment_jobs(course_id, skill_id, job_type)
            WHERE status IN (
                'queued', 'running', 'waiting_for_reference_review',
                'waiting_for_model', 'waiting_for_question_review'
            );
            """,
        )
    repository.initialize_schema()

    first = repository.enqueue(
        course_id="ai",
        skill_id="AI-SRC-08",
        job_type="automated_review",
        requested_count=6,
        clock=clock_at(),
    )
    repository.mark_waiting(first.job_id, "rejected_by_automated_review", job_type="automated_review")

    second = repository.enqueue(
        course_id="ai",
        skill_id="AI-SRC-08",
        job_type="automated_review",
        requested_count=6,
        clock=clock_at(),
    )
    assert second.job_id == first.job_id
    repository.close()


# The real bug: every real caller (cli.py's `scan`, app/bootstrap.py's low-supply
# reporter -- which fires on every Streamlit rerun) always calls enqueue() with the
# default job_type="replenish_skill", but the worker advances a job's job_type in
# place as it moves through the pipeline (almost immediately after creation). If
# dedup were keyed on job_type, a re-enqueue for a skill already mid-pipeline would
# no longer match the active row and would silently insert a duplicate. These tests
# hold job_type at its caller-supplied default and vary it only on the row itself,
# the way the worker actually does, to prove that scenario can no longer duplicate.
def test_reenqueue_with_default_job_type_is_idempotent_once_worker_advances_job_type(
    repository,
):
    first = repository.enqueue(
        course_id="ai", skill_id="AI-SRC-08", requested_count=6, clock=clock_at()
    )
    assert first.job_type == "replenish_skill"

    repository.mark_waiting(
        first.job_id, "waiting_for_reference_review", job_type="retrieve_references"
    )

    reenqueued = repository.enqueue(
        course_id="ai", skill_id="AI-SRC-08", requested_count=6, clock=clock_at()
    )
    assert reenqueued.job_id == first.job_id
    assert repository.get(first.job_id).job_type == "retrieve_references"
    assert len(repository.list_active(course_id="ai", skill_id="AI-SRC-08")) == 1


def test_repeated_streamlit_rerun_style_enqueue_never_duplicates_across_every_stage(
    repository,
):
    """Simulates one replenishment episode's full job_type progression, with a
    default-job_type enqueue() call (a Streamlit rerun hitting the low-supply
    reporter, or a repeated `scan`) interleaved after every stage transition."""
    job = repository.enqueue(
        course_id="ai", skill_id="AI-SRC-08", requested_count=6, clock=clock_at()
    )
    job_id = job.job_id

    pipeline_stages: list[JobType] = [
        "retrieve_references",
        "generate_questions",
        "automated_review",
        "automated_revision",
        "automated_review",
        "promote_approved_items",
    ]
    for stage in pipeline_stages:
        repository.mark_queued(job_id, job_type=stage)
        rerun = repository.enqueue(
            course_id="ai", skill_id="AI-SRC-08", requested_count=6, clock=clock_at()
        )
        assert rerun.job_id == job_id

    active = repository.list_active(course_id="ai", skill_id="AI-SRC-08")
    assert len(active) == 1
    assert active[0].job_id == job_id
