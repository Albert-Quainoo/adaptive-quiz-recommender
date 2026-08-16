"""Proves the learner path's expensive resources -- the SQLAlchemy engine,
the approved bank/taxonomy/BKT-model artifacts, and the schema-status
check -- are each acquired/loaded exactly once per process, not once per
Streamlit rerun. A full multi-rerun learner session (login, course
selection, submit, next question) drives app/main.py through
streamlit.testing.v1.AppTest exactly like
tests/test_learner_path_admin_status_disabled.py; spies on the underlying
functions assert each fires exactly once despite several reruns and two
separate learner actions.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

BANK_PATH = Path("outputs/approved_banks/pilot-approved-bank-43-v1.jsonl")


def _learner_path_env(tmp_path, name: str) -> dict[str, str]:
    return {
        "QUIZ_APPROVED_BANK_PATH": str(BANK_PATH),
        "QUIZ_BKT_MODEL_PATH": "outputs/bkt_dev_model_v4.pkl",
        "QUIZ_BKT_MODEL_VERSION": "bkt-synthetic-v4",
        "QUIZ_INITIAL_MASTERY_PROBABILITY": "0.20",
        "QUIZ_DATABASE_PATH": str(tmp_path / f"{name}.sqlite3"),
    }


def _spy(monkeypatch, module, name: str) -> list[int]:
    count = [0]
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        count[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(module, name, wrapper)
    return count


def _run_full_learner_session(app: AppTest) -> None:
    app.text_input[0].set_value("resource-caching-check")
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


def test_engine_and_expensive_artifacts_load_exactly_once_across_reruns(
    monkeypatch, tmp_path
):
    for key, value in _learner_path_env(tmp_path, "resource-caching").items():
        monkeypatch.setenv(key, value)

    import app.bootstrap as bootstrap_module
    import bkt.sqlite_repository as sqlite_repository_module

    # bkt/sqlite_repository.py does `from database import create_engine_for`,
    # binding its own local name at import time -- patch that name, not
    # database.create_engine_for, or the spy would never see any calls.
    engine_calls = _spy(monkeypatch, sqlite_repository_module, "create_engine_for")
    skills_calls = _spy(monkeypatch, bootstrap_module, "load_skills")
    bank_calls = _spy(monkeypatch, bootstrap_module, "load_approved_bank")
    model_calls = _spy(monkeypatch, bootstrap_module, "load_fitted_bkt_model")
    schema_status_calls = _spy(
        monkeypatch, sqlite_repository_module, "check_schema_status"
    )

    app = AppTest.from_file("app/main.py").run(timeout=20)
    assert not app.exception

    _run_full_learner_session(app)

    assert not app.exception
    assert engine_calls[0] == 1, (
        f"create_engine_for called {engine_calls[0]} times across a multi-rerun "
        "learner session -- the SQLAlchemy engine is being recreated instead of "
        "reused via CourseCatalogController's cached ApplicationController"
    )
    assert skills_calls[0] == 1, f"load_skills called {skills_calls[0]} times"
    assert bank_calls[0] == 1, f"load_approved_bank called {bank_calls[0]} times"
    assert model_calls[0] == 1, f"load_fitted_bkt_model called {model_calls[0]} times"
    assert schema_status_calls[0] == 1, (
        f"check_schema_status called {schema_status_calls[0]} times -- it should "
        "run once per course per process (inside initialize_schema on the first "
        "cold controller build), not on every learner action"
    )


def test_repeated_controller_resolution_returns_the_same_engine_instance(
    monkeypatch, tmp_path
):
    for key, value in _learner_path_env(tmp_path, "engine-identity").items():
        monkeypatch.setenv(key, value)

    from app.bootstrap import AppSettings
    from app.multi_course import build_course_catalog

    settings = AppSettings.from_sources(
        {
            "QUIZ_APPROVED_BANK_PATH": str(BANK_PATH),
            "QUIZ_BKT_MODEL_PATH": "outputs/bkt_dev_model_v4.pkl",
            "QUIZ_BKT_MODEL_VERSION": "bkt-synthetic-v4",
            "QUIZ_INITIAL_MASTERY_PROBABILITY": "0.20",
            "QUIZ_DATABASE_PATH": str(tmp_path / "engine-identity.sqlite3"),
        }
    )
    catalogue = build_course_catalog(settings)

    first = catalogue.resolve_active("intro-ai")
    second = catalogue.resolve_active("intro-ai")
    third = catalogue.resolve_active("intro-ai")

    assert first is second is third
    assert first.repository._engine is second.repository._engine is third.repository._engine
