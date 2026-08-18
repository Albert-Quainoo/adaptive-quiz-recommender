"""Tests for scripts/reconcile_demand_already_satisfied.py and its backing
repository method, SQLiteReplenishmentJobRepository.
reconcile_legacy_demand_already_satisfied() (authoring/replenishment/jobs.py).

Mirrors the minimal fixture shape tests/test_replenishment_cycle.py and
tests/test_replenishment_worker.py already established for a single-skill,
single-intent blueprint whose one slot is already an approved bank item --
the exact shape a pre-fix worker would have recorded as
mark_permanent_failure(error_code="demand_already_satisfied").
"""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest
from sqlalchemy import text

import authoring.question_intents as question_intents
import scripts.reconcile_demand_already_satisfied as reconcile
from api.bank import BankItem
from api.schemas import QuizQuestion
from authoring.grounded_batch import question_id
from authoring.question_intents import PilotBlueprint, QuestionIntent
from authoring.replenishment.jobs import JobConflictError, SQLiteReplenishmentJobRepository
from authoring.replenishment.manifest import CourseManifest, active_bank_path
from tests.postgres_test_safety import DSN_ENV_VAR, require_safe_postgres_target

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXED_TIME = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
SKILL_ID = "AI-SRC-08"
BATCH_ID = "test-legacy-batch-01"


@pytest.fixture
def manifest(tmp_path):
    taxonomy_dir = tmp_path / "taxonomy"
    taxonomy_dir.mkdir()
    shutil.copy(REPO_ROOT / "taxonomy/data/ai/skills.csv", taxonomy_dir / "skills.csv")
    (taxonomy_dir / "references.csv").write_text("skill_id,reference_material\n", encoding="utf-8")
    return CourseManifest(
        course_id="intro-ai",
        title="test",
        version="1",
        taxonomy_path=taxonomy_dir,
        approved_bank_path=tmp_path / "bank" / "intro-ai-bank-v0.jsonl",
        bkt_model_path=tmp_path / "model.pkl",
        candidate_store_path=tmp_path / "reference_candidates.json",
        review_store_path=tmp_path / "reviews",
        allowed_domains=("example.edu",),
        low_supply_threshold=1,
        target_supply=1,
        default_bkt_model_version="test-v1",
        status="active",
    )


@pytest.fixture
def reviewed_blueprint(tmp_path, monkeypatch):
    blueprint_dir = tmp_path / "blueprints"
    blueprint_dir.mkdir()
    intent = QuestionIntent(
        intent_id=f"{SKILL_ID}-INT-01", skill_id=SKILL_ID,
        assessment_focus="What a heuristic estimates", question_archetype="definition recall",
        preferred_reference_ids=["placeholder-ref"], required_concepts=["heuristic"],
        prohibited_conflations=["heuristic equals exact cost"], difficulty="intermediate",
    )
    blueprint = PilotBlueprint(
        batch_id=BATCH_ID, prompt_version=question_intents.PILOT_PROMPT_VERSION,
        review_status="blueprint-approved", reviewer_id="albert", reviewed_at=FIXED_TIME,
        base_seed=1, intents=[intent],
    )
    (blueprint_dir / f"{BATCH_ID}.json").write_text(
        json.dumps(blueprint.model_dump(mode="json")), encoding="utf-8"
    )
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", blueprint_dir)
    return blueprint


@pytest.fixture
def satisfied_bank(manifest, reviewed_blueprint):
    """The blueprint's one declared slot is already an approved bank item --
    the exact demand-satisfied shape a legacy job's error_code recorded."""
    path = active_bank_path(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    item = BankItem(
        item_id=question_id(BATCH_ID, SKILL_ID, 0),
        skill_id=SKILL_ID,
        provenance="generated",
        question=QuizQuestion(
            question="Placeholder question?", options=["A", "B", "C", "D"],
            correct_answer="A", explanation="Placeholder.", concept="placeholder",
            difficulty="introductory",
        ),
    )
    path.write_text(json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n", encoding="utf-8")


@pytest.fixture
def database(request, tmp_path):
    backend = getattr(request, "param", "sqlite")
    if backend == "sqlite":
        return tmp_path / "jobs.sqlite3"
    dsn = os.getenv(DSN_ENV_VAR)
    if not dsn:
        pytest.skip(f"set {DSN_ENV_VAR} to run this test against PostgreSQL")
    require_safe_postgres_target(dsn)
    cleanup = SQLiteReplenishmentJobRepository(dsn)
    with cleanup._engine.begin() as connection:
        connection.execute(text("DROP TABLE IF EXISTS replenishment_jobs CASCADE"))
    cleanup.close()
    return dsn


def _legacy_job(database, manifest, *, error_code="demand_already_satisfied"):
    """A job in exactly the shape a pre-fix worker would have left it:
    mark_permanent_failure(error_code=...), never mark_no_longer_needed()."""
    repository = SQLiteReplenishmentJobRepository(database)
    repository.initialize_schema()
    job = repository.enqueue(course_id=manifest.course_id, skill_id=SKILL_ID, requested_count=1)
    repository.mark_permanent_failure(
        job.job_id,
        error_code=error_code,
        error_message=(
            f"every intent slot in {BATCH_ID} for {SKILL_ID} is already an approved "
            "bank item; nothing to retrieve or generate"
        ),
    )
    repository.close()
    return job.job_id


@pytest.mark.parametrize("database", ["sqlite", "postgres"], indirect=True)
def test_dry_run_reports_the_plan_and_writes_nothing(
    tmp_path, manifest, reviewed_blueprint, satisfied_bank, database, monkeypatch, capsys
):
    job_id = _legacy_job(database, manifest)
    monkeypatch.setattr(reconcile, "load_preparation_eligible_manifests", lambda: [manifest])

    exit_code = reconcile.main([job_id, "--database", str(database)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "DRY RUN" in output
    assert "demand_fingerprint" in output

    repository = SQLiteReplenishmentJobRepository(database)
    unchanged = repository.get(job_id)
    assert unchanged.status == "permanent_failure"
    assert unchanged.error_code == "demand_already_satisfied"
    assert "demand_fingerprint" not in unchanged.metadata
    repository.close()


@pytest.mark.parametrize("database", ["sqlite", "postgres"], indirect=True)
def test_confirm_reconciles_and_preserves_the_historical_error(
    tmp_path, manifest, reviewed_blueprint, satisfied_bank, database, monkeypatch
):
    job_id = _legacy_job(database, manifest)
    monkeypatch.setattr(reconcile, "load_preparation_eligible_manifests", lambda: [manifest])

    exit_code = reconcile.main([job_id, "--database", str(database), "--confirm"])
    assert exit_code == 0

    repository = SQLiteReplenishmentJobRepository(database)
    updated = repository.get(job_id)
    repository.close()

    assert updated.status == "no_longer_needed"
    # Same error_code mark_no_longer_needed() itself now records for this outcome --
    # a reconciled row and an organically-produced one are indistinguishable by it.
    assert updated.error_code == "demand_already_satisfied"
    assert updated.metadata["historical_error_code"] == "demand_already_satisfied"
    assert updated.metadata["historical_error_message"] == updated.error_message
    assert updated.metadata["demand_fingerprint"]
    assert updated.metadata["reconciled_at"]


@pytest.mark.parametrize("database", ["sqlite", "postgres"], indirect=True)
def test_refuses_a_job_that_is_not_the_legacy_shape(
    tmp_path, manifest, reviewed_blueprint, satisfied_bank, database, monkeypatch
):
    """A real, unrelated permanent_failure (e.g. ambiguous_blueprint_selection) must
    never be touched by this command -- it targets exactly one error_code."""
    job_id = _legacy_job(database, manifest, error_code="ambiguous_blueprint_selection")
    monkeypatch.setattr(reconcile, "load_preparation_eligible_manifests", lambda: [manifest])

    exit_code = reconcile.main([job_id, "--database", str(database), "--confirm"])
    assert exit_code == 2

    repository = SQLiteReplenishmentJobRepository(database)
    unchanged = repository.get(job_id)
    repository.close()
    assert unchanged.status == "permanent_failure"
    assert unchanged.error_code == "ambiguous_blueprint_selection"


def test_refuses_an_unknown_job_id(tmp_path, manifest, monkeypatch):
    database = tmp_path / "jobs.sqlite3"
    repository = SQLiteReplenishmentJobRepository(database)
    repository.initialize_schema()
    repository.close()
    monkeypatch.setattr(reconcile, "load_preparation_eligible_manifests", lambda: [manifest])

    exit_code = reconcile.main(["not-a-real-job-id", "--database", str(database), "--confirm"])
    assert exit_code == 2


@pytest.mark.parametrize("database", ["sqlite", "postgres"], indirect=True)
def test_repository_method_refuses_a_second_application_transactionally(
    tmp_path, manifest, database
):
    """Reconciling the same job_id twice must never silently double-apply --
    the second call's guarded UPDATE matches zero rows (the first call already
    moved status off permanent_failure) and raises, touching nothing."""
    job_id = _legacy_job(database, manifest)
    repository = SQLiteReplenishmentJobRepository(database)

    first = repository.reconcile_legacy_demand_already_satisfied(
        job_id, demand_fingerprint="fp-1"
    )
    assert first.status == "no_longer_needed"

    with pytest.raises(JobConflictError, match="refusing to reconcile"):
        repository.reconcile_legacy_demand_already_satisfied(job_id, demand_fingerprint="fp-2")

    unchanged = repository.get(job_id)
    assert unchanged.metadata["demand_fingerprint"] == "fp-1"  # the second call wrote nothing
    repository.close()
