"""PostgreSQL integration coverage for migrate_course_ownership --
proves the same idempotent, data-preserving migration tested in
test_course_ownership_migration.py (SQLite) also works against the
production dialect. Skipped locally unless QUIZ_TEST_POSTGRES_URL is set;
never touches the Supabase DSN in .streamlit/secrets.toml.
"""

import os

import pytest
from sqlalchemy import text

from bkt.sqlite_repository import (
    DEFAULT_MIGRATION_COURSE_ID,
    SchemaStatus,
    SQLiteBKTRepository,
    check_schema_status,
    run_course_ownership_migration,
)
from database import create_engine_for
from tests.postgres_test_safety import DSN_ENV_VAR, require_safe_postgres_target

pytestmark = pytest.mark.skipif(
    not os.getenv(DSN_ENV_VAR),
    reason=f"set {DSN_ENV_VAR} to a PostgreSQL DSN to run these integration tests",
)

OLD_SCHEMA = """
CREATE TABLE attempt_events (
    attempt_id TEXT PRIMARY KEY,
    presentation_id TEXT NOT NULL,
    learner_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    selected_option_id TEXT NOT NULL,
    correct INTEGER NOT NULL CHECK (correct IN (0, 1)),
    attempt_order INTEGER NOT NULL,
    occurred_at TEXT NOT NULL,
    UNIQUE (learner_id, skill_id, attempt_order)
);

CREATE TABLE mastery_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    mastery_probability REAL NOT NULL,
    attempt_count INTEGER NOT NULL,
    source_attempt_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (source_attempt_id) REFERENCES attempt_events(attempt_id),
    UNIQUE (source_attempt_id, model_version)
);

CREATE TABLE recommendation_events (
    recommendation_id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    mastery_probability REAL NOT NULL,
    reason TEXT NOT NULL,
    model_version TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE content_gap_events (
    content_gap_id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL,
    completed_skill_id TEXT NOT NULL,
    completed_skill_name TEXT NOT NULL,
    newly_unlocked_skill_id TEXT NOT NULL,
    newly_unlocked_skill_name TEXT NOT NULL,
    missing_approved_content INTEGER NOT NULL CHECK (missing_approved_content IN (0, 1)),
    current_mastery_probability REAL NOT NULL,
    prerequisite_mastery_threshold REAL NOT NULL,
    reason TEXT NOT NULL,
    model_version TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE bkt_model_metadata (
    model_version TEXT PRIMARY KEY,
    fitted_at TEXT NOT NULL,
    training_attempt_count INTEGER NOT NULL,
    skill_ids TEXT NOT NULL
);

CREATE TABLE learner_sessions (
    learner_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL
);

CREATE TABLE question_presentations (
    presentation_id TEXT PRIMARY KEY,
    learner_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    presentation_seed TEXT NOT NULL,
    recommendation_reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (learner_id) REFERENCES learner_sessions(learner_id)
);

INSERT INTO learner_sessions (learner_id, created_at) VALUES
    ('learner-1', '2026-01-01T00:00:00+00:00'),
    ('learner-2', '2026-01-01T00:00:00+00:00');

INSERT INTO question_presentations (presentation_id, learner_id, item_id, presentation_seed, recommendation_reason, created_at) VALUES
    ('pres-1', 'learner-1', 'item-AI-PG-01', '42', 'foundational_unseen_skill', '2026-01-01T00:01:00+00:00'),
    ('pres-2', 'learner-2', 'item-AI-PG-01', '42', 'foundational_unseen_skill', '2026-01-01T00:01:00+00:00');

INSERT INTO attempt_events (attempt_id, presentation_id, learner_id, item_id, skill_id, selected_option_id, correct, attempt_order, occurred_at) VALUES
    ('attempt-1', 'pres-1', 'learner-1', 'item-AI-PG-01', 'AI-PG-01', 'opt-a', 1, 1, '2026-01-01T00:02:00+00:00'),
    ('attempt-2', 'pres-2', 'learner-2', 'item-AI-PG-01', 'AI-PG-01', 'opt-a', 0, 1, '2026-01-01T00:02:00+00:00');

INSERT INTO mastery_snapshots (snapshot_id, learner_id, skill_id, mastery_probability, attempt_count, source_attempt_id, model_version, updated_at) VALUES
    ('snap-1', 'learner-1', 'AI-PG-01', 0.6, 1, 'attempt-1', 'bkt-test-v1', '2026-01-01T00:02:01+00:00'),
    ('snap-2', 'learner-2', 'AI-PG-01', 0.3, 1, 'attempt-2', 'bkt-test-v1', '2026-01-01T00:02:01+00:00');

INSERT INTO recommendation_events (recommendation_id, learner_id, skill_id, item_id, difficulty, mastery_probability, reason, model_version, policy_version, created_at) VALUES
    ('rec-1', 'learner-1', 'AI-PG-01', 'item-AI-PG-01', 'introductory', 0.6, 'foundational_unseen_skill', 'bkt-test-v1', 'recommendation-policy-v1', '2026-01-01T00:01:30+00:00');

INSERT INTO content_gap_events (content_gap_id, learner_id, completed_skill_id, completed_skill_name, newly_unlocked_skill_id, newly_unlocked_skill_name, missing_approved_content, current_mastery_probability, prerequisite_mastery_threshold, reason, model_version, policy_version, created_at) VALUES
    ('gap-1', 'learner-1', 'AI-PG-01', 'Skill one', 'AI-PG-02', 'Skill two', 1, 0.8, 0.75, 'missing_approved_content_for_unlocked_skill', 'bkt-test-v1', 'recommendation-policy-v1', '2026-01-01T00:03:00+00:00');

INSERT INTO bkt_model_metadata (model_version, fitted_at, training_attempt_count, skill_ids) VALUES
    ('bkt-test-v1', '2025-12-01T00:00:00+00:00', 1000, '["AI-PG-01"]');
"""

ALL_TABLES = [
    "attempt_events",
    "mastery_snapshots",
    "recommendation_events",
    "content_gap_events",
    "bkt_model_metadata",
    "learner_sessions",
    "question_presentations",
]


def _dsn() -> str:
    return os.environ[DSN_ENV_VAR]


@pytest.fixture(autouse=True)
def _legacy_database():
    require_safe_postgres_target(_dsn())
    engine = create_engine_for(_dsn())
    with engine.begin() as connection:
        connection.execute(
            text(
                "DROP TABLE IF EXISTS question_presentations, learner_sessions, "
                "content_gap_events, recommendation_events, mastery_snapshots, "
                "bkt_model_metadata, attempt_events, schema_migrations CASCADE"
            )
        )
        for statement in OLD_SCHEMA.split(";"):
            statement = statement.strip()
            if statement:
                connection.execute(text(statement))
    engine.dispose()
    yield


def test_migration_preserves_rows_and_backfills_course_id_on_postgres():
    engine = create_engine_for(_dsn())
    with engine.connect() as connection:
        before_counts = {
            table: connection.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            for table in ALL_TABLES
        }
    engine.dispose()

    backfilled = run_course_ownership_migration(_dsn(), course_id=DEFAULT_MIGRATION_COURSE_ID)

    assert backfilled == {
        "mastery_snapshots": 2,
        "attempt_events": 2,
        "recommendation_events": 1,
        "content_gap_events": 1,
        "bkt_model_metadata": 1,
        "question_presentations": 2,
        "learner_sessions": 2,
    }

    engine = create_engine_for(_dsn())
    with engine.connect() as connection:
        after_counts = {
            table: connection.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar()
            for table in ALL_TABLES
        }
        for table in ALL_TABLES:
            course_ids = connection.execute(
                text(f"SELECT DISTINCT course_id FROM {table}")
            ).fetchall()
            assert course_ids == [(DEFAULT_MIGRATION_COURSE_ID,)]
    engine.dispose()

    assert before_counts == after_counts


def test_migration_is_idempotent_on_postgres():
    first_run = run_course_ownership_migration(_dsn(), course_id=DEFAULT_MIGRATION_COURSE_ID)
    second_run = run_course_ownership_migration(_dsn(), course_id=DEFAULT_MIGRATION_COURSE_ID)

    assert first_run
    assert second_run == {}


def test_runtime_works_after_migration_with_foreign_keys_enabled_on_postgres():
    run_course_ownership_migration(_dsn(), course_id=DEFAULT_MIGRATION_COURSE_ID)

    repository = SQLiteBKTRepository(_dsn(), course_id=DEFAULT_MIGRATION_COURSE_ID)
    status = repository.initialize_schema()
    assert status is SchemaStatus.CURRENT
    repository.close()


def test_schema_status_ignores_same_named_table_in_another_schema_on_postgres():
    """Regression: Supabase (and Postgres generally) ships its own
    same-named tables in other schemas -- e.g. auth.schema_migrations,
    realtime.schema_migrations. check_schema_status must not mistake one of
    those for this application's own public.schema_migrations and must
    still report MIGRATION_REQUIRED on the legacy schema the autouse
    fixture set up, rather than crashing on an unqualified query against a
    table that doesn't exist in the public schema."""
    engine = create_engine_for(_dsn())
    with engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS other_schema"))
        connection.execute(text("CREATE TABLE other_schema.schema_migrations (version TEXT)"))
    try:
        with engine.connect() as connection:
            status = check_schema_status(connection)
        assert status is SchemaStatus.MIGRATION_REQUIRED
    finally:
        with engine.begin() as connection:
            connection.execute(text("DROP SCHEMA other_schema CASCADE"))
    engine.dispose()


def test_migrated_primary_keys_on_postgres():
    run_course_ownership_migration(_dsn(), course_id=DEFAULT_MIGRATION_COURSE_ID)

    engine = create_engine_for(_dsn())
    with engine.connect() as connection:
        def pk_columns(table):
            rows = connection.execute(
                text(
                    f"""
                    SELECT a.attname FROM pg_index i
                    JOIN pg_attribute a ON a.attrelid = i.indrelid AND a.attnum = ANY(i.indkey)
                    WHERE i.indrelid = '{table}'::regclass AND i.indisprimary
                    """
                )
            ).fetchall()
            return {row[0] for row in rows}

        assert pk_columns("learner_sessions") == {"learner_id", "course_id"}
        assert pk_columns("bkt_model_metadata") == {"course_id", "model_version"}
    engine.dispose()


def test_migrated_foreign_keys_are_composite_on_postgres():
    run_course_ownership_migration(_dsn(), course_id=DEFAULT_MIGRATION_COURSE_ID)

    engine = create_engine_for(_dsn())
    with engine.connect() as connection:
        def fk_columns(table):
            # Columns must be paired by ordinal position, not joined on
            # constraint_name alone -- the latter produces a cross-product
            # for a composite FK (2 referencing cols x 2 referenced cols).
            rows = connection.execute(
                text(
                    """
                    SELECT kcu.column_name AS from_column, kcu2.column_name AS to_column
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON kcu.constraint_name = tc.constraint_name
                        AND kcu.table_name = tc.table_name
                    JOIN information_schema.referential_constraints rc
                        ON rc.constraint_name = tc.constraint_name
                    JOIN information_schema.key_column_usage kcu2
                        ON kcu2.constraint_name = rc.unique_constraint_name
                        AND kcu2.ordinal_position = kcu.position_in_unique_constraint
                    WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_name = :table
                    """
                ),
                {"table": table},
            ).fetchall()
            return {(row[0], row[1]) for row in rows}

        assert fk_columns("mastery_snapshots") == {
            ("course_id", "course_id"),
            ("source_attempt_id", "attempt_id"),
        }
        assert fk_columns("question_presentations") == {
            ("learner_id", "learner_id"),
            ("course_id", "course_id"),
        }
    engine.dispose()


def test_orphan_mastery_snapshot_is_rejected_on_postgres():
    from datetime import datetime, timezone

    from sqlalchemy.exc import IntegrityError

    from bkt.schemas import MasterySnapshot

    run_course_ownership_migration(_dsn(), course_id=DEFAULT_MIGRATION_COURSE_ID)
    repository = SQLiteBKTRepository(_dsn(), course_id=DEFAULT_MIGRATION_COURSE_ID)
    repository.initialize_schema()

    orphan = MasterySnapshot(
        learner_id="learner-1",
        course_id=DEFAULT_MIGRATION_COURSE_ID,
        skill_id="AI-PG-01",
        mastery_probability=0.5,
        attempt_count=1,
        source_attempt_id="does-not-exist",
        model_version="v1",
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    with pytest.raises(IntegrityError):
        repository.save_mastery(orphan)
    repository.close()


def test_orphan_question_presentation_is_rejected_on_postgres():
    from sqlalchemy.exc import IntegrityError

    run_course_ownership_migration(_dsn(), course_id=DEFAULT_MIGRATION_COURSE_ID)
    repository = SQLiteBKTRepository(_dsn(), course_id=DEFAULT_MIGRATION_COURSE_ID)
    repository.initialize_schema()

    with pytest.raises(IntegrityError):
        with repository._engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO question_presentations
                        (presentation_id, course_id, learner_id, item_id,
                         presentation_seed, recommendation_reason, created_at)
                    VALUES
                        ('pres-orphan', :course_id, 'no-such-learner', 'item-1',
                         '1', 'reason', '2026-01-01T00:00:00+00:00')
                    """
                ),
                {"course_id": DEFAULT_MIGRATION_COURSE_ID},
            )
    repository.close()


def test_child_record_cannot_reference_a_parent_from_another_course_on_postgres():
    from datetime import datetime, timezone

    from sqlalchemy.exc import IntegrityError

    from bkt.schemas import MasterySnapshot

    run_course_ownership_migration(_dsn(), course_id=DEFAULT_MIGRATION_COURSE_ID)

    other_course_repo = SQLiteBKTRepository(_dsn(), course_id="other-course")
    cross_course_snapshot = MasterySnapshot(
        learner_id="learner-1",
        course_id="other-course",
        skill_id="AI-PG-01",
        mastery_probability=0.5,
        attempt_count=1,
        source_attempt_id="attempt-1",  # real attempt, but under intro-ai
        model_version="v1",
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    with pytest.raises(IntegrityError):
        other_course_repo.save_mastery(cross_course_snapshot)
    other_course_repo.close()
