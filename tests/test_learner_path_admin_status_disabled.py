"""Proves the public learner interface never calls into the replenishment
inventory scan by default. AppSettings.admin_status_enabled defaults to
False (see app/bootstrap.py); this drives an ordinary learner session --
login, course selection, submit, next question, across several reruns --
and asserts compute_course_inventory (the function that reads every
prep-eligible course's taxonomy and approved bank off disk) is never
invoked, without needing QUIZ_ADMIN_STATUS_ENABLED set at all.
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
        # Deliberately not set: QUIZ_ADMIN_STATUS_ENABLED. The default
        # (disabled) is exactly what this test is proving the consequence
        # of.
    }


def _count_calls(monkeypatch, target: str) -> list[int]:
    """Returns a single-element list holding the call count, updated by a
    wrapper installed at `target` -- a mutable cell survives inside the
    Streamlit subprocess-free AppTest run alongside monkeypatch's own
    teardown."""
    count = [0]
    import authoring.replenishment.inventory as inventory_module

    original = inventory_module.compute_course_inventory

    def spy(*args, **kwargs):
        count[0] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(inventory_module, "compute_course_inventory", spy)
    return count


def test_ordinary_learner_session_never_scans_replenishment_inventory(
    monkeypatch, tmp_path
):
    for key, value in _learner_path_env(tmp_path, "learner-path-no-scan").items():
        monkeypatch.setenv(key, value)
    call_count = _count_calls(monkeypatch, "compute_course_inventory")

    app = AppTest.from_file("app/main.py").run(timeout=20)
    assert not app.exception

    # Login
    app.text_input[0].set_value("learner-scan-check")
    next(button for button in app.button if button.label == "Start").click()
    app.run(timeout=20)

    # Course selection
    app.text_input[0].set_value("AI")
    next(button for button in app.button if button.label == "Continue").click()
    app.run(timeout=20)

    # Answer the question (submit) -- this is the path that writes an
    # attempt and updates mastery, the highest-traffic learner action.
    app.radio[0].set_value(app.radio[0].options[0])
    next(button for button in app.button if button.label == "Submit").click()
    app.run(timeout=20)

    # Advance to the next question
    next_button = next(
        (button for button in app.button if button.label == "Next Question"), None
    )
    if next_button is not None:
        next_button.click()
        app.run(timeout=20)

    assert not app.exception
    assert call_count[0] == 0, (
        f"compute_course_inventory was called {call_count[0]} time(s) during an "
        "ordinary learner session with admin status disabled by default"
    )


def test_admin_status_setting_defaults_to_disabled():
    from app.bootstrap import AppSettings

    settings = AppSettings.from_sources(
        {
            "QUIZ_APPROVED_BANK_PATH": "x",
            "QUIZ_BKT_MODEL_PATH": "y",
            "QUIZ_BKT_MODEL_VERSION": "z",
        }
    )
    assert settings.admin_status_enabled is False


def test_admin_status_setting_is_only_enabled_by_explicit_opt_in():
    from app.bootstrap import AppSettings

    for value in ("true", "1", "yes", "on", "TRUE", "On"):
        settings = AppSettings.from_sources(
            {
                "QUIZ_APPROVED_BANK_PATH": "x",
                "QUIZ_BKT_MODEL_PATH": "y",
                "QUIZ_BKT_MODEL_VERSION": "z",
                "QUIZ_ADMIN_STATUS_ENABLED": value,
            }
        )
        assert settings.admin_status_enabled is True, f"failed for {value!r}"

    for value in ("false", "0", "no", "off", "", "garbage"):
        settings = AppSettings.from_sources(
            {
                "QUIZ_APPROVED_BANK_PATH": "x",
                "QUIZ_BKT_MODEL_PATH": "y",
                "QUIZ_BKT_MODEL_VERSION": "z",
                "QUIZ_ADMIN_STATUS_ENABLED": value,
            }
        )
        assert settings.admin_status_enabled is False, f"failed for {value!r}"
