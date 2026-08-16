"""Learner-path isolation and admin-view behavior for the replenishment pipeline.

The learner-facing app must never import retrieval or model-loading code, and
the new admin status view must render without ever touching either -- this is
checked in a genuinely fresh subprocess so no other test's imports can hide a
violation, plus a full AppTest run against the real entrypoint.
"""

import subprocess
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

BANK_PATH = Path("outputs/approved_banks/pilot-approved-bank-38-v1.jsonl")


def _env(tmp_path, name: str) -> dict[str, str]:
    return {
        "QUIZ_APPROVED_BANK_PATH": str(BANK_PATH),
        "QUIZ_BKT_MODEL_PATH": "outputs/bkt_dev_model_v4.pkl",
        "QUIZ_BKT_MODEL_VERSION": "bkt-synthetic-v4",
        "QUIZ_INITIAL_MASTERY_PROBABILITY": "0.20",
        "QUIZ_DATABASE_PATH": str(tmp_path / f"{name}.sqlite3"),
        # These tests specifically exercise the admin sidebar's own
        # behavior (rendering, idempotency, failure resilience), so they
        # must explicitly opt in -- it defaults to disabled for every
        # ordinary learner-path render (see AppSettings.admin_status_enabled
        # and tests/test_learner_path_admin_status_disabled.py, which proves
        # the default-off behavior this constant is the exception to).
        "QUIZ_ADMIN_STATUS_ENABLED": "true",
    }


NETWORK_OR_MODEL_MODULES = (
    "authoring.retrieval.brave",  # Brave Search HTTP client
    "authoring.retrieval.fetcher",  # page-fetching HTTP client
    "api.model_loader",  # torch/transformers model loading
)


def test_importing_the_learner_app_never_pulls_in_retrieval_or_model_loader():
    # The read-only inventory path legitimately imports pure data/schema
    # modules from authoring.retrieval (CandidateStore, ReferenceCandidate)
    # to count pending/approved references -- that's allowed (constraint 8:
    # "may read inventory"). What must never load is anything that actually
    # performs network I/O or loads the model.
    entrypoint = Path("app/main.py").resolve()
    probe = (
        "import runpy, sys; "
        f"runpy.run_path({str(entrypoint)!r}, run_name='isolation_probe'); "
        f"watched = {NETWORK_OR_MODEL_MODULES!r}; "
        "leaked = [name for name in sys.modules if name in watched]; "
        "print('LEAKED:' + ','.join(leaked))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    leaked_line = next(line for line in completed.stdout.splitlines() if line.startswith("LEAKED:"))
    assert leaked_line == "LEAKED:"


def test_admin_status_view_renders(monkeypatch, tmp_path):
    # Module-level import isolation is checked by the subprocess test above,
    # in a genuinely fresh interpreter -- sys.modules here is shared with
    # every other test in this session, so it can't prove non-import.
    for key, value in _env(tmp_path, "admin-isolation").items():
        monkeypatch.setenv(key, value)

    app = AppTest.from_file("app/main.py").run(timeout=20)

    assert not app.exception
    expander_labels = [expander.label for expander in app.expander]
    assert any("Content status" in label for label in expander_labels)


def test_admin_status_view_is_idempotent_across_reruns(monkeypatch, tmp_path):
    for key, value in _env(tmp_path, "admin-rerun").items():
        monkeypatch.setenv(key, value)

    app = AppTest.from_file("app/main.py").run(timeout=20)
    first_captions = [caption.value for caption in app.caption]

    app.run(timeout=20)
    second_captions = [caption.value for caption in app.caption]

    assert not app.exception
    assert first_captions == second_captions


def test_admin_status_failure_never_breaks_the_learner_quiz(monkeypatch, tmp_path):
    def explode(manifest, job_repository, **kwargs):
        raise RuntimeError("simulated replenishment outage")

    # render_admin_status imports compute_course_inventory lazily (inside its
    # own guarded try/except) so the core app never takes a module-level
    # dependency on authoring.replenishment -- patch it at its source module,
    # which the lazy import re-resolves on every call.
    monkeypatch.setattr(
        "authoring.replenishment.inventory.compute_course_inventory", explode
    )
    for key, value in _env(tmp_path, "admin-inventory-outage").items():
        monkeypatch.setenv(key, value)

    app = AppTest.from_file("app/main.py").run(timeout=20)

    assert not app.exception
    assert app.title[0].value == "Adaptive Quiz"
    assert app.text_input[0].label == "Learner ID"
