"""CLI subcommands driven end to end with fakes, mirroring
tests/test_retrieval_store.py's CLI-fake pattern.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

import authoring.replenishment.cli as cli
from authoring.replenishment.jobs import SQLiteReplenishmentJobRepository
from authoring.replenishment.manifest import CourseManifest
from authoring.retrieval.models import SearchResult, approve
from authoring.retrieval.search import FetchedPage
from authoring.retrieval.store import CandidateStore

FIXED_TIME = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)


def manifest_at(tmp_path, **overrides) -> CourseManifest:
    taxonomy_dir = tmp_path / "taxonomy"
    taxonomy_dir.mkdir(exist_ok=True)
    (taxonomy_dir / "skills.csv").write_text(
        "skill_id,topic,subtopic,name,learning_objective,cognitive_process,"
        "generation_strategy,template_id,prerequisite_skill_ids\n"
        "AI-SRC-08,Search,Informed,Heuristics,Explain heuristics,understand,generated,,\n",
        encoding="utf-8",
    )
    (taxonomy_dir / "references.csv").write_text("skill_id,reference_material\n", encoding="utf-8")
    fields = dict(
        course_id="ai", title="t", version="1", taxonomy_path=taxonomy_dir,
        approved_bank_path=tmp_path / "bank" / "ai-bank-v0.jsonl",
        bkt_model_path=tmp_path / "model.pkl",
        candidate_store_path=tmp_path / "candidates.json",
        review_store_path=tmp_path / "reviews",
        allowed_domains=("example.edu",), low_supply_threshold=3, target_supply=6,
        default_bkt_model_version="v1", status="active",
    )
    fields.update(overrides)
    return CourseManifest(**fields)


@pytest.fixture
def isolated_manifest(tmp_path, monkeypatch):
    manifest = manifest_at(tmp_path)
    monkeypatch.setattr(cli, "load_preparation_eligible_manifests", lambda: [manifest])
    # Isolated from the real authoring/blueprints/ directory: these tests exercise
    # retrieval/scan/status behavior, not blueprint resolution, and the real
    # directory now legitimately carries more than one blueprint covering AI-SRC-08.
    import authoring.question_intents as question_intents

    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", tmp_path / "no-blueprints")
    return manifest


class FakeSearchProvider:
    def __init__(self, results):
        self.results = results

    def search(self, schedule, diagnostics, budget):
        for step in schedule:
            if not budget.may_request():
                return
            budget.spend_request()
            diagnostics.record_query(step.domain)
            for result in self.results:
                yield step, result


class FakePageFetcher:
    def __init__(self, pages):
        self.pages = pages

    def fetch(self, url):
        return FetchedPage(url=url, text=self.pages[url])


PASSAGE = (
    "A heuristic function estimates the cost of the cheapest path from a given "
    "state to a goal state. It lets an informed search order the frontier by "
    "how promising a state looks rather than by how far it already is."
)


def test_scan_enqueues_low_supply_skills(tmp_path, isolated_manifest, capsys):
    database = tmp_path / "jobs.sqlite3"
    exit_code = cli.main(["--database", str(database), "scan"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "AI-SRC-08" in output
    assert "1 job(s) enqueued." in output


def test_scan_is_idempotent_across_reruns(tmp_path, isolated_manifest, capsys):
    database = tmp_path / "jobs.sqlite3"
    cli.main(["--database", str(database), "scan"])
    capsys.readouterr()
    cli.main(["--database", str(database), "scan"])
    output = capsys.readouterr().out
    assert "0 job(s) enqueued." in output


def test_status_reports_readiness_for_every_skill(tmp_path, isolated_manifest, capsys):
    database = tmp_path / "jobs.sqlite3"
    # status is read-only (see authoring/replenishment/cli.py) and never creates
    # schema -- in production it always already exists, so seed it here exactly like
    # a real pre-provisioned database, without scan()'s enqueue() side effect (which
    # would change the very readiness this test asserts).
    SQLiteReplenishmentJobRepository(database).initialize_schema()
    exit_code = cli.main(["--database", str(database), "status"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "AI-SRC-08" in output
    assert "taxonomy_only" in output


def test_status_against_uninitialized_database_reports_schema_not_ready(
    tmp_path, isolated_manifest
):
    """status must never silently create schema -- a genuinely uninitialized
    database (this test's database fixture is a path that does not exist yet) is a
    stop-and-report condition, not something it repairs."""
    from authoring.replenishment.jobs import SchemaNotReadyError

    database = tmp_path / "jobs.sqlite3"
    with pytest.raises(SchemaNotReadyError, match="schema_not_ready"):
        cli.main(["--database", str(database), "status"])


def test_worker_once_processes_a_single_job_with_fakes(
    tmp_path, isolated_manifest, monkeypatch, capsys
):
    url = "https://example.edu/heuristics.html"
    provider = FakeSearchProvider([SearchResult(title="Heuristics", url=url, snippet="")])
    fetcher = FakePageFetcher({url: PASSAGE})
    monkeypatch.setattr(cli, "_search_provider_factory", lambda manifest: provider)
    monkeypatch.setattr(cli, "_fetcher_factory", lambda manifest: fetcher)

    database = tmp_path / "jobs.sqlite3"
    cli.main(["--database", str(database), "scan"])
    capsys.readouterr()

    exit_code = cli.main(["--database", str(database), "worker", "--once"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "waiting_for_reference_review" in output

    candidates = CandidateStore(isolated_manifest.candidate_store_path).load()
    assert len(candidates) == 1
    assert candidates[0].skill_id == "AI-SRC-08"


def test_worker_once_with_no_queued_jobs_reports_nothing_to_do(tmp_path, isolated_manifest, capsys):
    database = tmp_path / "jobs.sqlite3"
    exit_code = cli.main(["--database", str(database), "worker", "--once"])
    assert exit_code == 0
    assert "No claimable job." in capsys.readouterr().out


def test_model_factory_defaults_to_the_local_adapter(monkeypatch):
    monkeypatch.delenv("QUIZ_REPLENISHMENT_INFERENCE_PROVIDER", raising=False)
    assert cli._model_factory() is cli.LlamaBatchModel


def test_model_factory_selects_modal_when_configured(monkeypatch):
    monkeypatch.setenv("QUIZ_REPLENISHMENT_INFERENCE_PROVIDER", "modal")
    assert cli._model_factory() is cli.ModalBatchModel


def test_model_factory_rejects_an_unknown_provider(monkeypatch):
    monkeypatch.setenv("QUIZ_REPLENISHMENT_INFERENCE_PROVIDER", "bogus")
    with pytest.raises(SystemExit, match="bogus"):
        cli._model_factory()


def test_reviewer_factory_defaults_to_the_local_adapter(monkeypatch):
    monkeypatch.delenv("QUIZ_REVIEW_REVIEWER_PROVIDER", raising=False)
    assert cli._reviewer_factory() is cli.REVIEW_PROVIDERS["local"]


def test_reviewer_factory_selects_modal_when_configured(monkeypatch):
    monkeypatch.setenv("QUIZ_REVIEW_REVIEWER_PROVIDER", "modal")
    assert cli._reviewer_factory() is cli.REVIEW_PROVIDERS["modal"]


def test_reviewer_factory_rejects_an_unknown_provider(monkeypatch):
    monkeypatch.setenv("QUIZ_REVIEW_REVIEWER_PROVIDER", "bogus")
    with pytest.raises(SystemExit, match="bogus"):
        cli._reviewer_factory()


def test_reviewer_factory_never_offers_the_fake_provider(monkeypatch):
    monkeypatch.delenv("QUIZ_REVIEW_REVIEWER_PROVIDER", raising=False)
    assert "fake" not in cli.REVIEW_PROVIDERS


def test_scan_pauses_a_skill_at_its_pending_question_cap(
    tmp_path, isolated_manifest, monkeypatch, capsys
):
    from authoring.grounded_review import CurationItem, GroundedReview, GroundedReviewStore

    monkeypatch.setenv("QUIZ_REVIEW_MAX_PENDING_PER_SKILL", "1")
    review = GroundedReview(
        batch_id="b1",
        source_hashes={},
        items=[
            CurationItem(
                original_question_id="q-1",
                skill_id="AI-SRC-08",
                intent_id="AI-SRC-08-INT-01",
                recommendation="propose_revision",
                recommendation_reason="pending",
            )
        ],
    )
    isolated_manifest.review_store_path.mkdir(parents=True, exist_ok=True)
    GroundedReviewStore(isolated_manifest.review_store_path / "b1__AI-SRC-08.json").save(review)

    database = tmp_path / "jobs.sqlite3"
    exit_code = cli.main(["--database", str(database), "scan"])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "0 job(s) enqueued." in output
