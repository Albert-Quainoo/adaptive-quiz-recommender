"""Proves the migration-control contract added to bkt/sqlite_repository.py
and app/bootstrap.py: production schema migration never runs from
build_controller(), initialize_schema(), course selection, login, or any
learner request. The only way to actually migrate a legacy database is the
explicit, protected run_course_ownership_migration() entry point (invoked
only via scripts/migrate_course_ownership.py) -- initialize_schema() only
ever creates-if-empty, verifies-if-current, or raises
SchemaMigrationRequiredError without mutating anything.
"""

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import text

from bkt.sqlite_repository import (
    DEFAULT_MIGRATION_COURSE_ID,
    SchemaMigrationRequiredError,
    SchemaStatus,
    SQLiteBKTRepository,
    check_schema_status,
    run_course_ownership_migration,
)
from database import create_engine_for
from tests.test_course_ownership_migration import (
    ALL_TABLES,
    _build_legacy_database,
    _table_contents_excluding_course_id,
)

BANK_PATH = Path("outputs/approved_banks/pilot-approved-bank-43-v1.jsonl")


def _learner_path_env(tmp_path, name: str) -> dict[str, str]:
    return {
        "QUIZ_APPROVED_BANK_PATH": str(BANK_PATH),
        "QUIZ_BKT_MODEL_PATH": "outputs/bkt_dev_model_v4.pkl",
        "QUIZ_BKT_MODEL_VERSION": "bkt-synthetic-v4",
        "QUIZ_INITIAL_MASTERY_PROBABILITY": "0.20",
        "QUIZ_DATABASE_PATH": str(tmp_path / f"{name}.sqlite3"),
    }


def test_fresh_empty_database_initializes_without_raising(tmp_path):
    path = tmp_path / "fresh.sqlite3"
    repository = SQLiteBKTRepository(path, course_id="intro-ai")

    status = repository.initialize_schema()

    assert status is SchemaStatus.EMPTY
    with repository._engine.connect() as connection:
        assert check_schema_status(connection) is SchemaStatus.CURRENT
    repository.close()


def test_legacy_database_reports_migration_required_without_changing_anything(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    _build_legacy_database(path)

    before = _table_contents_excluding_course_id(path)

    repository = SQLiteBKTRepository(path, course_id=DEFAULT_MIGRATION_COURSE_ID)
    with pytest.raises(SchemaMigrationRequiredError) as raised:
        repository.initialize_schema()
    repository.close()

    assert "maintenance mode" in raised.value.user_message.lower()
    after = _table_contents_excluding_course_id(path)
    assert before == after, "initialize_schema() must never mutate a legacy database"

    connection = sqlite3.connect(str(path))
    for table in ALL_TABLES:
        columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})").fetchall()}
        assert "course_id" not in columns
    connection.close()


def test_explicit_migration_succeeds_and_is_idempotent(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    _build_legacy_database(path)

    first_run = run_course_ownership_migration(path, course_id=DEFAULT_MIGRATION_COURSE_ID)
    assert first_run  # something was actually migrated

    second_run = run_course_ownership_migration(path, course_id=DEFAULT_MIGRATION_COURSE_ID)
    assert second_run == {}  # nothing left to migrate, true no-op

    engine = create_engine_for(path)
    with engine.connect() as connection:
        assert check_schema_status(connection) is SchemaStatus.CURRENT
    engine.dispose()


def test_runtime_works_correctly_after_explicit_migration(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    _build_legacy_database(path)
    run_course_ownership_migration(path, course_id=DEFAULT_MIGRATION_COURSE_ID)

    repository = SQLiteBKTRepository(path, course_id=DEFAULT_MIGRATION_COURSE_ID)
    status = repository.initialize_schema()
    assert status is SchemaStatus.CURRENT

    # A real query against the migrated schema, through the ordinary
    # runtime API -- proves the migrated shape is actually usable, not
    # just structurally present.
    attempts = repository.list_attempts(learner_id="learner-1")
    assert attempts
    assert all(attempt.course_id == DEFAULT_MIGRATION_COURSE_ID for attempt in attempts)
    repository.close()


def test_migration_failure_rolls_back_completely(tmp_path, monkeypatch):
    path = tmp_path / "legacy.sqlite3"
    _build_legacy_database(path)
    before = _table_contents_excluding_course_id(path)

    import bkt.sqlite_repository as module

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated failure during migration")

    # _record_schema_version runs last, inside the same migration
    # transaction as the table rebuild -- forcing it to fail proves the
    # whole transaction (including the table rebuild that already ran)
    # rolls back, not just the last statement.
    monkeypatch.setattr(module, "_record_schema_version", _boom)

    with pytest.raises(RuntimeError, match="simulated failure during migration"):
        module.run_course_ownership_migration(path, course_id=DEFAULT_MIGRATION_COURSE_ID)

    after = _table_contents_excluding_course_id(path)
    assert before == after, "a failed migration must leave every row exactly as it was"

    engine = create_engine_for(path)
    with engine.connect() as connection:
        assert check_schema_status(connection) is SchemaStatus.MIGRATION_REQUIRED
    engine.dispose()


def test_no_migration_occurs_during_an_ordinary_learner_session(monkeypatch, tmp_path):
    """Drives a full learner session -- login, course selection, submit,
    next question -- through the real Streamlit app entrypoint against a
    fresh (empty) database, and proves run_course_ownership_migration /
    migrate_course_ownership are never called. A fresh database only ever
    needs initialize_schema()'s EMPTY path (a plain CREATE), never the
    migration path -- this test's spy would also catch a regression that
    made initialize_schema() migrate again."""
    for key, value in _learner_path_env(tmp_path, "no-migration-learner-path").items():
        monkeypatch.setenv(key, value)

    import bkt.sqlite_repository as module

    call_count = [0]
    original = module.migrate_course_ownership

    def spy(*args, **kwargs):
        call_count[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "migrate_course_ownership", spy)

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("app/main.py").run(timeout=20)
    assert not app.exception

    app.text_input[0].set_value("learner-migration-check")
    next(button for button in app.button if button.label == "Start").click()
    app.run(timeout=20)

    app.text_input[0].set_value("AI")
    next(button for button in app.button if button.label == "Continue").click()
    app.run(timeout=20)

    app.radio[0].set_value(app.radio[0].options[0])
    next(button for button in app.button if button.label == "Submit").click()
    app.run(timeout=20)

    next_button = next(
        (button for button in app.button if button.label == "Next Question"), None
    )
    if next_button is not None:
        next_button.click()
        app.run(timeout=20)

    assert not app.exception
    assert call_count[0] == 0, (
        f"migrate_course_ownership was called {call_count[0]} time(s) during an "
        "ordinary learner session -- migration must only run via the explicit "
        "scripts/migrate_course_ownership.py CLI"
    )


def test_maintenance_mode_setting_defaults_to_disabled():
    from app.bootstrap import AppSettings

    settings = AppSettings.from_sources(
        {
            "QUIZ_APPROVED_BANK_PATH": "x",
            "QUIZ_BKT_MODEL_PATH": "y",
            "QUIZ_BKT_MODEL_VERSION": "z",
        }
    )
    assert settings.maintenance_mode is False


def test_maintenance_mode_setting_is_only_enabled_by_explicit_opt_in():
    from app.bootstrap import AppSettings

    for value in ("true", "1", "yes", "on", "TRUE", "On"):
        settings = AppSettings.from_sources(
            {
                "QUIZ_APPROVED_BANK_PATH": "x",
                "QUIZ_BKT_MODEL_PATH": "y",
                "QUIZ_BKT_MODEL_VERSION": "z",
                "QUIZ_MAINTENANCE_MODE": value,
            }
        )
        assert settings.maintenance_mode is True, f"failed for {value!r}"

    for value in ("false", "0", "no", "off", "", "garbage"):
        settings = AppSettings.from_sources(
            {
                "QUIZ_APPROVED_BANK_PATH": "x",
                "QUIZ_BKT_MODEL_PATH": "y",
                "QUIZ_BKT_MODEL_VERSION": "z",
                "QUIZ_MAINTENANCE_MODE": value,
            }
        )
        assert settings.maintenance_mode is False, f"failed for {value!r}"


def test_maintenance_mode_blocks_learner_session_but_allows_startup(monkeypatch, tmp_path):
    """When enabled, learner writes must be blocked and a maintenance
    message shown, while settings/catalogue loading (basic startup
    diagnostics) still runs -- proven by the absence of app.exception."""
    for key, value in _learner_path_env(tmp_path, "maintenance-mode-blocks").items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("QUIZ_MAINTENANCE_MODE", "true")

    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file("app/main.py").run(timeout=20)

    assert not app.exception
    assert any("maintenance" in warning.value.lower() for warning in app.warning)
    # No login form should be rendered -- the maintenance gate returns
    # before get_session_state()/render_login() ever runs.
    assert not app.text_input
