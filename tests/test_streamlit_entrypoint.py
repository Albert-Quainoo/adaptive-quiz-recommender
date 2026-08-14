import subprocess
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from app.bootstrap import AppSettings, BootstrapError, build_controller, load_approved_bank


BANK_PATH = Path("outputs/approved_banks/pilot-approved-bank-38-v1.jsonl")


def test_entrypoint_resolves_package_imports_outside_repository_cwd(tmp_path):
    entrypoint = Path("app/main.py").resolve()
    command = (
        "import runpy; "
        f"runpy.run_path({str(entrypoint)!r}, run_name='streamlit_import_test')"
    )

    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "ModuleNotFoundError" not in completed.stderr


def test_configured_approved_bank_contains_38_items():
    items = load_approved_bank(BANK_PATH)

    assert len(items) == 38
    assert len({item.item_id for item in items}) == 38


@pytest.mark.parametrize(
    ("bank_path", "model_path", "message"),
    [
        (Path("missing-bank.jsonl"), Path("outputs/bkt_dev_model_v4.pkl"), "bank"),
        (BANK_PATH, Path("missing-model.pkl"), "BKT model"),
    ],
)
def test_bootstrap_reports_missing_runtime_artifacts(tmp_path, bank_path, model_path, message):
    settings = AppSettings(
        database_path=tmp_path / "app.sqlite3",
        approved_bank_path=bank_path,
        bkt_model_path=model_path,
        skills_path=Path("taxonomy/data/ai/skills.csv"),
        references_path=Path("taxonomy/data/ai/references.csv"),
        model_version="bkt-synthetic-v4",
        policy_version="recommendation-policy-v1",
    )

    with pytest.raises(BootstrapError, match=message):
        build_controller(settings)


def test_bootstrap_rejects_model_without_every_bank_skill(tmp_path):
    settings = AppSettings(
        database_path=tmp_path / "coverage.sqlite3",
        approved_bank_path=BANK_PATH,
        bkt_model_path=Path("outputs/bkt_dev_model_v2.pkl"),
        skills_path=Path("taxonomy/data/ai/skills.csv"),
        references_path=Path("taxonomy/data/ai/references.csv"),
        model_version="bkt-synthetic-v2",
        policy_version="recommendation-policy-v1",
        initial_mastery_probability=0.0,
    )

    with pytest.raises(BootstrapError, match="AI-AGT-01, AI-SRC-03"):
        build_controller(settings)


def test_bootstrap_rejects_initial_mastery_that_differs_from_model_prior(tmp_path):
    settings = AppSettings(
        database_path=tmp_path / "prior.sqlite3",
        approved_bank_path=BANK_PATH,
        bkt_model_path=Path("outputs/bkt_dev_model_v4.pkl"),
        skills_path=Path("taxonomy/data/ai/skills.csv"),
        references_path=Path("taxonomy/data/ai/references.csv"),
        model_version="bkt-synthetic-v4",
        policy_version="recommendation-policy-v1",
        initial_mastery_probability=0.0,
    )

    with pytest.raises(BootstrapError, match="initial mastery"):
        build_controller(settings)


def test_streamlit_entrypoint_starts_without_interaction(monkeypatch, tmp_path):
    monkeypatch.setenv("QUIZ_APPROVED_BANK_PATH", str(BANK_PATH))
    monkeypatch.setenv("QUIZ_BKT_MODEL_PATH", "outputs/bkt_dev_model_v4.pkl")
    monkeypatch.setenv("QUIZ_BKT_MODEL_VERSION", "bkt-synthetic-v4")
    monkeypatch.setenv("QUIZ_INITIAL_MASTERY_PROBABILITY", "0.20")
    monkeypatch.setenv("QUIZ_DATABASE_PATH", str(tmp_path / "startup.sqlite3"))

    app = AppTest.from_file("app/main.py").run(timeout=20)

    assert not app.exception
    assert app.title[0].value == "Adaptive Quiz"
    assert app.text_input[0].label == "Learner ID"


def test_streamlit_submission_locks_question_and_shows_feedback(monkeypatch, tmp_path):
    monkeypatch.setenv("QUIZ_APPROVED_BANK_PATH", str(BANK_PATH))
    monkeypatch.setenv("QUIZ_BKT_MODEL_PATH", "outputs/bkt_dev_model_v4.pkl")
    monkeypatch.setenv("QUIZ_BKT_MODEL_VERSION", "bkt-synthetic-v4")
    monkeypatch.setenv("QUIZ_INITIAL_MASTERY_PROBABILITY", "0.20")
    monkeypatch.setenv("QUIZ_DATABASE_PATH", str(tmp_path / "submission.sqlite3"))
    app = AppTest.from_file("app/main.py").run(timeout=20)
    app.text_input[0].set_value("streamlit-submit-test")
    next(button for button in app.button if button.label == "Start").click()
    app.run(timeout=20)
    app.radio[0].set_value(app.radio[0].options[0])
    next(button for button in app.button if button.label == "Submit").click()
    app.run(timeout=20)

    submit = next(button for button in app.button if button.label == "Submit")
    assert not app.exception
    assert submit.disabled is True
    assert len(app.success) + len(app.error) >= 1
    assert {metric.label for metric in app.metric} == {
        "Previous mastery",
        "Updated mastery",
    }
    assert any(button.label == "Next Question" for button in app.button)


def test_switch_button_returns_to_login_and_activates_a_new_learner(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("QUIZ_APPROVED_BANK_PATH", str(BANK_PATH))
    monkeypatch.setenv("QUIZ_BKT_MODEL_PATH", "outputs/bkt_dev_model_v4.pkl")
    monkeypatch.setenv("QUIZ_BKT_MODEL_VERSION", "bkt-synthetic-v4")
    monkeypatch.setenv("QUIZ_INITIAL_MASTERY_PROBABILITY", "0.20")
    monkeypatch.setenv("QUIZ_DATABASE_PATH", str(tmp_path / "switch.sqlite3"))
    app = AppTest.from_file("app/main.py").run(timeout=20)
    app.text_input[0].set_value("learner-a")
    next(button for button in app.button if button.label == "Start").click()
    app.run(timeout=20)

    next(button for button in app.button if button.label == "Switch learner").click()
    app.run(timeout=20)

    session = app.session_state["adaptive_quiz_session"]
    assert session.learner_id is None
    assert session.question is None
    assert app.text_input[0].label == "Learner ID"
    assert any(button.label == "Start" for button in app.button)

    app.text_input[0].set_value("learner-b")
    next(button for button in app.button if button.label == "Start").click()
    app.run(timeout=20)

    session = app.session_state["adaptive_quiz_session"]
    assert not app.exception
    assert session.learner_id == "learner-b"
    assert session.question is not None
    assert any(
        "Learner: learner-b" in markdown.value for markdown in app.markdown
    )
