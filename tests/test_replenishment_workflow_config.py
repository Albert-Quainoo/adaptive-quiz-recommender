"""Structural checks on .github/workflows/replenishment.yml.

GitHub Actions' own scheduling/concurrency semantics cannot be exercised
from pytest (there is no real runner here), so this only verifies the
workflow *declares* the safety properties the requirements ask for:
manual-dispatch-first rollout (schedule trigger left disabled), a
concurrency lock that queues rather than cancels (preventing concurrent
runs from writing to content-ops at the same time), every required secret
referenced by name, and that generated content is only ever pushed to the
content-ops branch, never to main.
"""

from pathlib import Path

import yaml

WORKFLOW_PATH = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "replenishment.yml"


def _load() -> dict:
    return yaml.safe_load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _triggers(config: dict) -> dict:
    # PyYAML parses the bare `on:` key as the boolean True (YAML 1.1), not
    # the string "on" -- both are checked so this test survives either.
    return config.get("on", config.get(True))


def test_workflow_file_exists_and_parses():
    assert WORKFLOW_PATH.is_file()
    config = _load()
    assert config["name"] == "Content Replenishment"


def test_concurrency_lock_queues_rather_than_cancels():
    config = _load()
    concurrency = config["concurrency"]
    assert concurrency["group"] == "replenishment-cycle"
    # Must queue a second run, not kill an in-flight branch write mid-commit.
    assert concurrency["cancel-in-progress"] is False


def test_manual_dispatch_is_enabled_and_schedule_is_not_yet():
    triggers = _triggers(_load())
    assert "workflow_dispatch" in triggers
    assert "schedule" not in triggers  # left commented out pending rollout review

    raw = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "# schedule:" in raw
    assert "cron:" in raw


def test_workflow_dispatch_defaults_to_dry_run():
    triggers = _triggers(_load())
    dry_run_input = triggers["workflow_dispatch"]["inputs"]["dry_run"]
    assert dry_run_input["default"] is True


def test_every_required_secret_is_referenced_by_name():
    raw = WORKFLOW_PATH.read_text(encoding="utf-8")
    for secret in (
        "BRAVE_SEARCH_API_KEY",
        "QUIZ_DATABASE_URL",
        "MODAL_INFERENCE_ENDPOINT",
        "MODAL_PROXY_TOKEN_ID",
        "MODAL_PROXY_TOKEN_SECRET",
        "MODEL_REPOSITORY",
    ):
        assert f"secrets.{secret}" in raw, f"missing secrets.{secret} reference"


def test_no_secret_value_is_ever_echoed_or_printed():
    raw = WORKFLOW_PATH.read_text(encoding="utf-8")
    for line in raw.splitlines():
        stripped = line.strip()
        if stripped.startswith("echo") or stripped.startswith("print"):
            assert "secrets." not in stripped


def test_generated_content_is_pushed_only_to_content_ops_never_main():
    raw = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "CONTENT_OPS_BRANCH" in raw
    assert "content-ops/replenishment" in raw
    push_lines = [line for line in raw.splitlines() if "git push" in line]
    assert push_lines, "expected at least one git push step"
    for line in push_lines:
        assert "$CONTENT_OPS_BRANCH" in line or "CONTENT_OPS_BRANCH" in line
        assert "main" not in line


def test_pr_body_disclaims_approval_or_promotion():
    raw = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "does **not** approve or promote" in raw


def test_review_artifacts_are_uploaded_from_the_run_scoped_staging_directory():
    """The upload step must point at the run-scoped staging directory built
    by scripts.stage_run_artifacts, never at outputs/replenishment/ itself --
    that tree also holds every earlier run's job-scoped artifacts, checked
    out from the content-ops branch, which must never be re-uploaded as if
    they belonged to this run."""
    config = _load()
    steps = config["jobs"]["replenish"]["steps"]
    upload_steps = [
        step for step in steps if step.get("uses", "").startswith("actions/upload-artifact")
    ]
    assert upload_steps
    assert upload_steps[0]["with"]["path"] == "outputs/replenishment_run_artifact/"


def test_run_scoped_staging_step_runs_before_upload_and_always():
    config = _load()
    steps = config["jobs"]["replenish"]["steps"]
    stage_index = next(
        index for index, step in enumerate(steps)
        if "stage_run_artifacts" in step.get("run", "")
    )
    upload_index = next(
        index for index, step in enumerate(steps)
        if step.get("uses", "").startswith("actions/upload-artifact")
    )
    assert stage_index < upload_index
    assert steps[stage_index]["if"] == "always()"
