"""Proves migrate_course_ownership (bkt/sqlite_repository.py) preserves
every existing row exactly while backfilling course_id, is idempotent, and
produces the expected final schema shape -- the same checks run manually
against a production-shaped copy before this feature ships, kept here as a
permanent regression test.
"""

import sqlite3

import pytest
from sqlalchemy import text

from bkt.sqlite_repository import (
    DEFAULT_MIGRATION_COURSE_ID,
    SchemaStatus,
    SQLiteBKTRepository,
    migrate_course_ownership,
    run_course_ownership_migration,
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


def _build_legacy_database(path, *, learner_count: int = 5, skills=("AI-SRC-01", "AI-SRC-02")):
    connection = sqlite3.connect(str(path))
    connection.executescript(OLD_SCHEMA)

    learners = [f"learner-{i}" for i in range(1, learner_count + 1)]
    for learner in learners:
        connection.execute(
            "INSERT INTO learner_sessions (learner_id, created_at) VALUES (?, ?)",
            (learner, "2026-01-01T00:00:00+00:00"),
        )

    counts = {"attempts": 0, "mastery": 0, "presentations": 0, "recommendations": 0}
    for learner in learners:
        for order, skill in enumerate(skills, start=1):
            presentation_id = f"pres-{learner}-{skill}"
            attempt_id = f"attempt-{learner}-{skill}"
            item_id = f"item-{skill}-a"
            connection.execute(
                """INSERT INTO question_presentations
                   (presentation_id, learner_id, item_id, presentation_seed,
                    recommendation_reason, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (presentation_id, learner, item_id, "42", "foundational_unseen_skill",
                 "2026-01-01T00:01:00+00:00"),
            )
            counts["presentations"] += 1
            connection.execute(
                """INSERT INTO attempt_events
                   (attempt_id, presentation_id, learner_id, item_id, skill_id,
                    selected_option_id, correct, attempt_order, occurred_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (attempt_id, presentation_id, learner, item_id, skill, "opt-a", 1, order,
                 "2026-01-01T00:02:00+00:00"),
            )
            counts["attempts"] += 1
            connection.execute(
                """INSERT INTO mastery_snapshots
                   (snapshot_id, learner_id, skill_id, mastery_probability, attempt_count,
                    source_attempt_id, model_version, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (f"snap-{attempt_id}", learner, skill, 0.55 + order * 0.01, order,
                 attempt_id, "bkt-test-v1", "2026-01-01T00:02:01+00:00"),
            )
            counts["mastery"] += 1
            connection.execute(
                """INSERT INTO recommendation_events
                   (recommendation_id, learner_id, skill_id, item_id, difficulty,
                    mastery_probability, reason, model_version, policy_version, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (f"rec-{attempt_id}", learner, skill, item_id, "introductory", 0.55,
                 "foundational_unseen_skill", "bkt-test-v1", "recommendation-policy-v1",
                 "2026-01-01T00:01:30+00:00"),
            )
            counts["recommendations"] += 1

    connection.execute(
        """INSERT INTO content_gap_events
           (content_gap_id, learner_id, completed_skill_id, completed_skill_name,
            newly_unlocked_skill_id, newly_unlocked_skill_name, missing_approved_content,
            current_mastery_probability, prerequisite_mastery_threshold, reason,
            model_version, policy_version, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("gap-1", "learner-1", skills[0], "Skill one", skills[1], "Skill two", 1, 0.8, 0.75,
         "missing_approved_content_for_unlocked_skill", "bkt-test-v1",
         "recommendation-policy-v1", "2026-01-01T00:03:00+00:00"),
    )
    connection.execute(
        """INSERT INTO bkt_model_metadata
           (model_version, fitted_at, training_attempt_count, skill_ids)
           VALUES (?, ?, ?, ?)""",
        ("bkt-test-v1", "2025-12-01T00:00:00+00:00", 1000,
         '["AI-SRC-01", "AI-SRC-02"]'),
    )
    connection.commit()
    connection.close()
    return counts


def _table_contents_excluding_course_id(path):
    connection = sqlite3.connect(str(path))
    snapshot = {}
    for table in ALL_TABLES:
        columns = [
            row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()
            if row[1] != "course_id"
        ]
        rows = connection.execute(
            f"SELECT {','.join(columns)} FROM {table} ORDER BY {columns[0]}"
        ).fetchall()
        snapshot[table] = rows
    connection.close()
    return snapshot


@pytest.fixture
def legacy_database(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    counts = _build_legacy_database(path)
    return path, counts


def test_migration_preserves_every_row_exactly(legacy_database):
    path, counts = legacy_database
    before = _table_contents_excluding_course_id(path)

    run_course_ownership_migration(path, course_id=DEFAULT_MIGRATION_COURSE_ID)

    after = _table_contents_excluding_course_id(path)
    assert before == after


def test_migration_reports_correct_backfill_counts(legacy_database):
    path, counts = legacy_database
    backfilled = run_course_ownership_migration(path, course_id=DEFAULT_MIGRATION_COURSE_ID)

    assert backfilled["attempt_events"] == counts["attempts"]
    assert backfilled["mastery_snapshots"] == counts["mastery"]
    assert backfilled["recommendation_events"] == counts["recommendations"]
    assert backfilled["question_presentations"] == counts["presentations"]
    assert backfilled["content_gap_events"] == 1
    assert backfilled["bkt_model_metadata"] == 1
    assert backfilled["learner_sessions"] == 5


def test_migration_backfills_every_row_to_the_same_course_id(legacy_database):
    path, _ = legacy_database
    run_course_ownership_migration(path, course_id=DEFAULT_MIGRATION_COURSE_ID)

    connection = sqlite3.connect(str(path))
    for table in ALL_TABLES:
        values = connection.execute(f"SELECT DISTINCT course_id FROM {table}").fetchall()
        assert values == [(DEFAULT_MIGRATION_COURSE_ID,)]
    connection.close()


def test_migration_is_idempotent(legacy_database):
    path, _ = legacy_database
    first_run = run_course_ownership_migration(path, course_id=DEFAULT_MIGRATION_COURSE_ID)
    second_run = run_course_ownership_migration(path, course_id=DEFAULT_MIGRATION_COURSE_ID)

    assert first_run  # something was actually migrated the first time
    assert second_run == {}  # nothing left to migrate the second time


def test_a_fresh_database_initializes_directly_without_migration(tmp_path):
    path = tmp_path / "fresh.sqlite3"
    repository = SQLiteBKTRepository(path, course_id="dsa")
    status = repository.initialize_schema()
    repository.close()

    assert status is SchemaStatus.EMPTY
    connection = sqlite3.connect(str(path))
    for table in ALL_TABLES:
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
        assert "course_id" in columns
    connection.close()


def test_migrated_schema_has_expected_primary_keys_and_indexes(legacy_database):
    path, _ = legacy_database
    run_course_ownership_migration(path, course_id=DEFAULT_MIGRATION_COURSE_ID)

    connection = sqlite3.connect(str(path))

    def pk_columns(table):
        info = connection.execute(f"PRAGMA table_info({table})").fetchall()
        return [row[1] for row in sorted(info, key=lambda row: row[5]) if row[5] > 0]

    assert pk_columns("learner_sessions") == ["learner_id", "course_id"]
    assert pk_columns("bkt_model_metadata") == ["course_id", "model_version"]
    assert pk_columns("attempt_events") == ["attempt_id"]
    assert pk_columns("question_presentations") == ["presentation_id"]

    index_names = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ).fetchall()
    }
    assert index_names == {
        "idx_attempt_events_course_learner",
        "idx_mastery_snapshots_identity",
        "idx_recommendation_events_course_learner",
        "idx_content_gap_events_course_learner",
        "idx_question_presentations_course_learner",
    }
    connection.close()


def test_migrate_course_ownership_is_dialect_explicit_not_inspector_based(legacy_database):
    """Regression guard for the collection-time bug this migration hit
    during development: CREATE INDEX statements must never run against a
    table that migrate_course_ownership hasn't gotten to yet.

    migrate_course_ownership() itself never touches the foreign_keys
    pragma (see its docstring) -- callers going around
    SQLiteBKTRepository.initialize_schema() must disable it themselves
    first, exactly like initialize_schema() does, or the DROP TABLE on a
    still-FK-referenced legacy table fails."""
    path, _ = legacy_database
    from sqlalchemy import create_engine

    from bkt.sqlite_repository import _set_sqlite_foreign_keys

    engine = create_engine(f"sqlite:///{path}")
    _set_sqlite_foreign_keys(engine, enabled=False)
    with engine.begin() as connection:
        backfilled = migrate_course_ownership(connection)
    _set_sqlite_foreign_keys(engine, enabled=True)
    engine.dispose()
    assert backfilled


def test_migrated_foreign_keys_are_composite_and_course_scoped(legacy_database):
    path, _ = legacy_database
    run_course_ownership_migration(path, course_id=DEFAULT_MIGRATION_COURSE_ID)

    connection = sqlite3.connect(str(path))
    mastery_fks = {
        (row[3], row[4])  # (from, to) column names
        for row in connection.execute("PRAGMA foreign_key_list(mastery_snapshots)").fetchall()
    }
    assert mastery_fks == {("course_id", "course_id"), ("source_attempt_id", "attempt_id")}

    presentation_fks = {
        (row[3], row[4])
        for row in connection.execute(
            "PRAGMA foreign_key_list(question_presentations)"
        ).fetchall()
    }
    assert presentation_fks == {("learner_id", "learner_id"), ("course_id", "course_id")}
    connection.close()


def test_migration_reports_zero_foreign_key_violations(legacy_database):
    path, _ = legacy_database
    run_course_ownership_migration(path, course_id=DEFAULT_MIGRATION_COURSE_ID)

    connection = sqlite3.connect(str(path))
    connection.execute("PRAGMA foreign_keys=ON")
    violations = connection.execute("PRAGMA foreign_key_check").fetchall()
    connection.close()

    assert violations == []


def test_runtime_works_after_migration_with_foreign_keys_enabled(legacy_database):
    """After the explicit migration, a freshly constructed repository (the
    same shape build_controller() constructs on every course selection)
    must see SchemaStatus.CURRENT via the runtime-safe initialize_schema()
    path, and its own engine's connection must have foreign_keys ON --
    create_engine_for's connect listener sets this by default, and nothing
    in the CURRENT/EMPTY paths ever needs to override it."""
    path, _ = legacy_database
    run_course_ownership_migration(path, course_id=DEFAULT_MIGRATION_COURSE_ID)

    repository = SQLiteBKTRepository(path, course_id=DEFAULT_MIGRATION_COURSE_ID)
    status = repository.initialize_schema()
    assert status is SchemaStatus.CURRENT

    with repository._engine.connect() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).fetchone()[0] == 1
    repository.close()


def test_orphan_mastery_snapshot_is_rejected(legacy_database):
    from sqlalchemy.exc import IntegrityError

    from bkt.schemas import MasterySnapshot
    from datetime import datetime, timezone

    path, _ = legacy_database
    run_course_ownership_migration(path, course_id=DEFAULT_MIGRATION_COURSE_ID)
    repository = SQLiteBKTRepository(path, course_id=DEFAULT_MIGRATION_COURSE_ID)
    repository.initialize_schema()

    orphan = MasterySnapshot(
        learner_id="learner-1",
        course_id=DEFAULT_MIGRATION_COURSE_ID,
        skill_id="AI-SRC-01",
        mastery_probability=0.5,
        attempt_count=1,
        source_attempt_id="does-not-exist",
        model_version="v1",
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        repository.save_mastery(orphan)
    repository.close()


def test_orphan_question_presentation_is_rejected(legacy_database):
    from sqlalchemy.exc import IntegrityError

    path, _ = legacy_database
    run_course_ownership_migration(path, course_id=DEFAULT_MIGRATION_COURSE_ID)
    repository = SQLiteBKTRepository(path, course_id=DEFAULT_MIGRATION_COURSE_ID)
    repository.initialize_schema()

    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
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


def test_child_record_cannot_reference_a_parent_from_another_course(legacy_database):
    """The composite FK is what makes this rejection possible: attempt_id
    'attempt-learner-1-AI-SRC-01' genuinely exists (under intro-ai), but
    (course_id='other-course', attempt_id='attempt-learner-1-AI-SRC-01')
    does not -- a plain single-column FK on attempt_id alone would have let
    this through."""
    from sqlalchemy.exc import IntegrityError

    from bkt.schemas import MasterySnapshot
    from datetime import datetime, timezone

    path, _ = legacy_database
    run_course_ownership_migration(path, course_id=DEFAULT_MIGRATION_COURSE_ID)

    other_course_repo = SQLiteBKTRepository(path, course_id="other-course")
    cross_course_snapshot = MasterySnapshot(
        learner_id="learner-1",
        course_id="other-course",
        skill_id="AI-SRC-01",
        mastery_probability=0.5,
        attempt_count=1,
        source_attempt_id="attempt-learner-1-AI-SRC-01",
        model_version="v1",
        updated_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        other_course_repo.save_mastery(cross_course_snapshot)
    other_course_repo.close()
