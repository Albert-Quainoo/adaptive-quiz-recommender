"""Tests for the bounded, reportable replenishment cycle
(scripts/run_replenishment_cycle.py) that backs the scheduled GitHub Actions
workflow. Mirrors the fixture shapes already established in
tests/test_replenishment_worker.py (DeterministicFakeModel, reviewed
blueprint + approved candidate) and tests/test_replenishment_cli.py
(monkeypatching cli._search_provider_factory/_model_factory/etc.).
"""

import json
import shutil
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlalchemy import event
from sqlalchemy.engine import Engine

import authoring.grounding_briefs as grounding_briefs
import authoring.question_intents as question_intents
import authoring.replenishment.cli as cli
import scripts.run_replenishment_cycle as cycle
from authoring.grounded_batch import PendingQuestion, question_id, read_jsonl
from authoring.grounded_review import (
    GroundedReviewStore,
    RevisionProvenance,
    approve_revision,
    propose_revision,
)
from authoring.grounding_briefs import CanonicalGroundingBrief
from authoring.question_intents import PilotBlueprint, QuestionIntent
from authoring.replenishment.budget import CycleBudgetConfig
from authoring.replenishment.demand import compute_demand_fingerprint
from authoring.replenishment.inventory import compute_course_inventory
from authoring.replenishment.jobs import SQLiteReplenishmentJobRepository
from authoring.replenishment.manifest import CourseManifest, active_bank_path
from authoring.replenishment.snapshot import archive_stale_jobs
from api.bank import BankItem
from authoring.retrieval.models import ReferenceCandidate, SearchResult, approve, new_candidate
from authoring.retrieval.search import FetchedPage
from authoring.retrieval.store import CandidateStore
from authoring.review.models import (
    AnswerAssessment,
    DifficultyAssessment,
    DuplicateAssessment,
    GroundingAssessment,
    ObjectiveAssessment,
    SemanticReviewResult,
)
from authoring.review.config import ReviewPolicyConfig
from authoring.review.reviewer import FakeContentReviewer
from api.schemas import QuizQuestion
from scripts.stage_run_artifacts import stage_run_artifacts

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXED_TIME = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
SKILL_ID = "AI-SRC-08"

PASSAGE = (
    "A heuristic function estimates the cost of the cheapest path from a given "
    "state to a goal state. It lets an informed search order the frontier by "
    "how promising a state looks rather than by how far it already is."
)


def fixed_clock():
    return FIXED_TIME


class FakeSearchProvider:
    def __init__(self, results=()):
        self.results = list(results)
        self.call_count = 0

    def search(self, schedule, diagnostics, budget):
        self.call_count += 1
        for step in schedule:
            if not budget.may_request():
                return
            budget.spend_request()
            diagnostics.record_query(step.domain)
            for result in self.results:
                yield step, result


class FakePageFetcher:
    def __init__(self, pages: dict[str, str]):
        self.pages = pages

    def fetch(self, url: str) -> FetchedPage:
        return FetchedPage(url=url, text=self.pages[url])


class DeterministicFakeModel:
    model_id = "fake-grounded-model"
    model_revision = "fake-revision-1"

    def generate(self, messages, seed, generation_parameters):
        return json.dumps(
            {
                "questions": [
                    {
                        "question": "What does a heuristic estimate?",
                        "options": [
                            "Remaining cost to goal",
                            "Total memory used",
                            "Number of nodes",
                            "Branching factor",
                        ],
                        "correct_answer": "Remaining cost to goal",
                        "explanation": "A heuristic estimates the cheapest remaining path cost.",
                        "concept": "Heuristics",
                        "difficulty": "intermediate",
                    }
                ]
            }
        )


def _clean_review_result(correct_answer: str) -> SemanticReviewResult:
    return SemanticReviewResult(
        grounding_assessment=GroundingAssessment(
            grounded=True, independently_supported_answer=True, grounding_confidence=0.95
        ),
        answer_assessment=AnswerAssessment(
            selected_option_text=correct_answer,
            matches_declared_answer=True,
            multiple_defensible_answers=False,
            obviously_signalled_answer=False,
            answer_confidence=0.95,
        ),
        objective_assessment=ObjectiveAssessment(
            measures_declared_skill=True,
            satisfies_intent_blueprint=True,
            matches_objective_verb=True,
            cognitive_demand="understand",
            duplicates_another_intent=False,
        ),
        difficulty_assessment=DifficultyAssessment(
            difficulty_justified=True,
            explanation_depth_matches_difficulty=True,
            is_definition_recall_only=False,
        ),
        duplicate_assessment=DuplicateAssessment(),
        reviewer_model_id="fake-reviewer",
        reviewer_model_revision="fake-rev-1",
        reviewer_prompt_version="review-v1",
        reviewer_prompt_template_hash="d" * 64,
        rendered_review_request_hash="d" * 64,
    )


def _semantic_reject_review_result() -> SemanticReviewResult:
    return _clean_review_result("Remaining cost to goal").model_copy(
        update={
            "answer_assessment": AnswerAssessment(
                selected_option_text="Total memory used",
                matches_declared_answer=False,
                multiple_defensible_answers=False,
                obviously_signalled_answer=False,
                answer_confidence=0.9,
            )
        }
    )


def clean_reviewer_factory():
    return FakeContentReviewer(
        {"What does a heuristic estimate?": _clean_review_result("Remaining cost to goal")}
    )


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
def isolated(manifest, monkeypatch):
    """Only this one scratch manifest is visible to the cycle, and only a
    scratch blueprint directory -- never the real authoring/blueprints/ or
    authoring/replenishment/manifests/ content."""
    monkeypatch.setattr(cycle, "load_preparation_eligible_manifests", lambda: [manifest])
    return manifest


@pytest.fixture
def database(tmp_path):
    return tmp_path / "jobs.sqlite3"


@pytest.fixture
def approved_candidate(manifest) -> ReferenceCandidate:
    candidate = new_candidate(
        SKILL_ID, "Heuristics", "https://example.edu/pathfinding/heuristics.html",
        "example.edu", PASSAGE, FIXED_TIME, relevance_score=20,
        matched_terms=["heuristic", "frontier"],
    )
    store = CandidateStore(manifest.candidate_store_path)
    store.add([candidate])
    store.replace(approve(candidate, "albert", reviewed_at=FIXED_TIME))
    return candidate


@pytest.fixture
def reviewed_blueprint(tmp_path, monkeypatch, approved_candidate):
    blueprint_dir = tmp_path / "blueprints"
    blueprint_dir.mkdir()
    intent = QuestionIntent(
        intent_id=f"{SKILL_ID}-INT-01", skill_id=SKILL_ID,
        assessment_focus="What a heuristic estimates", question_archetype="definition recall",
        preferred_reference_ids=[approved_candidate.candidate_id],
        required_concepts=["heuristic", "estimate"],
        prohibited_conflations=["heuristic equals exact cost"], difficulty="intermediate",
    )
    blueprint = PilotBlueprint(
        batch_id="test-batch-01", prompt_version=question_intents.PILOT_PROMPT_VERSION,
        review_status="blueprint-approved", reviewer_id="albert", reviewed_at=FIXED_TIME,
        base_seed=1, intents=[intent],
    )
    (blueprint_dir / "test-batch-01.json").write_text(
        json.dumps(blueprint.model_dump(mode="json")), encoding="utf-8"
    )
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", blueprint_dir)
    monkeypatch.setattr(
        grounding_briefs, "PILOT_GROUNDING_BRIEFS",
        {SKILL_ID: CanonicalGroundingBrief(
            skill_id=SKILL_ID, version="test-v1",
            statements=["A heuristic estimates the remaining cost to the goal."],
        )},
    )
    return blueprint


def _approve_first_pending_question(review_path: Path, output_dir: Path) -> str:
    review = GroundedReviewStore(review_path).load()
    item = review.items[0]
    source = next(
        q for q in read_jsonl(output_dir / "pending_questions.jsonl", PendingQuestion)
        if q.question_id == item.original_question_id
    )
    revised = QuizQuestion.model_validate(
        source.question.model_dump() | {"explanation": source.question.explanation + " (reviewed)"}
    )
    proposed = propose_revision(
        item, source.question, revised, "albert", "reviewer restated wording",
        edited_at=FIXED_TIME, provenance=RevisionProvenance.from_source(source),
    )
    GroundedReviewStore(review_path).replace_item(proposed)
    approved = approve_revision(
        proposed, proposed.revisions[0].revision_id, "albert", reviewed_at=FIXED_TIME
    )
    GroundedReviewStore(review_path).replace_item(approved)
    return item.original_question_id


def _run(database, snapshot_root, *, dry_run=False, budget=None, retention_days=14):
    return cycle.run_cycle(
        database=database, snapshot_root=snapshot_root, dry_run=dry_run,
        budget_config=budget or CycleBudgetConfig(), retention_days=retention_days,
        clock=fixed_clock,
    )


def _seed_schema(database) -> None:
    """A dry run is read-only and never creates schema (see
    authoring/replenishment/jobs.py's check_schema_ready()) -- in production the
    schema always already exists by the time a dry run runs, so every dry-run test
    below that isn't specifically testing the missing-schema case seeds it first,
    exactly like a real pre-provisioned database."""
    repository = SQLiteReplenishmentJobRepository(database)
    repository.initialize_schema()
    repository.close()


def _dump_jobs(database) -> list[dict]:
    """Every column of every row in replenishment_jobs, for proving a dry run
    leaves existing job data completely untouched -- not just row counts."""
    connection = sqlite3.connect(database)
    connection.row_factory = sqlite3.Row
    try:
        rows = connection.execute("SELECT * FROM replenishment_jobs ORDER BY job_id").fetchall()
        return [dict(row) for row in rows]
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# Dry run: zero external calls, zero database writes, deterministic report
# ---------------------------------------------------------------------------

def test_dry_run_reports_proposed_jobs_without_writing_anything(
    tmp_path, isolated, database, reviewed_blueprint, monkeypatch
):
    def _boom(*args, **kwargs):
        raise AssertionError("dry run must never construct a search/model/reviewer provider")

    monkeypatch.setattr(cli, "_search_provider_factory", _boom)
    monkeypatch.setattr(cli, "_fetcher_factory", _boom)
    monkeypatch.setattr(cli, "_model_factory", _boom)
    monkeypatch.setattr(cli, "_reviewer_factory", _boom)
    _seed_schema(database)

    report = _run(database, tmp_path / "snapshots", dry_run=True)

    assert report.dry_run is True
    assert report.job_outcomes == []
    assert report.pending_approvals == []
    assert report.budget == {
        "planned_new_candidates": 1,
        "max_new_candidates": 3,
        "max_cost_usd": 5.0,
        "estimated_cost_usd": 0.0,
    }
    assert report.archived_job_dirs == []
    assert report.queue_rows_written == 0

    proposed = [row for row in report.deficiencies if row.skill_id == SKILL_ID]
    assert len(proposed) == 1
    row = proposed[0]
    assert row.course_id == "intro-ai"
    # A dry run never calls repository.enqueue() -- "proposed", not
    # "enqueued", so the decision label never overstates job-queue state.
    assert row.decision == "proposed"
    assert row.difficulty == "intermediate"  # resolved from reviewed_blueprint
    assert row.reason
    assert row.proposed_job_key == cycle.report.deterministic_job_key("intro-ai", SKILL_ID)

    assert len(report.execution_plan) == 1
    assert report.execution_plan[0].skill_id == SKILL_ID
    assert report.execution_plan[0].rank == 1

    # Zero writes: the queue table holds no rows (schema init alone is not a
    # data mutation), and no snapshot/branch artifact directory was created.
    assert _dump_jobs(database) == []
    assert not (tmp_path / "snapshots").exists()


def test_dry_run_blocks_a_deficiency_with_no_resolvable_difficulty(
    tmp_path, isolated, database, monkeypatch
):
    """No reviewed blueprint exists for SKILL_ID in this scenario (the
    `reviewed_blueprint` fixture is not requested), so the deficiency scan
    can only resolve difficulty="unknown" for it -- must never be reported
    (or, in a live run, enqueued) as a plannable job in that state."""

    def _boom(*args, **kwargs):
        raise AssertionError("dry run must never construct a search/model/reviewer provider")

    monkeypatch.setattr(cli, "_search_provider_factory", _boom)
    monkeypatch.setattr(cli, "_fetcher_factory", _boom)
    monkeypatch.setattr(cli, "_model_factory", _boom)
    monkeypatch.setattr(cli, "_reviewer_factory", _boom)
    _seed_schema(database)

    report = _run(database, tmp_path / "snapshots", dry_run=True)

    row = next(row for row in report.deficiencies if row.skill_id == SKILL_ID)
    assert row.decision == "blocked"
    assert row.difficulty == "unknown"
    assert row.proposed_job_key == "-"
    assert not any(planned.skill_id == SKILL_ID for planned in report.execution_plan)
    assert report.queue_rows_written == 0


def test_dry_run_leaves_existing_job_rows_logically_unchanged(
    tmp_path, isolated, database
):
    repository = SQLiteReplenishmentJobRepository(database)
    repository.initialize_schema()
    repository.enqueue(
        course_id="intro-ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock
    )
    before = _dump_jobs(database)
    assert len(before) == 1

    _run(database, tmp_path / "snapshots", dry_run=True)

    assert _dump_jobs(database) == before


def test_repeated_dry_runs_are_side_effect_free(tmp_path, isolated, database):
    _seed_schema(database)
    first = _run(database, tmp_path / "snapshots", dry_run=True)
    second = _run(database, tmp_path / "snapshots", dry_run=True)

    assert _dump_jobs(database) == []
    # Nothing persisted between runs, so the second dry run re-derives the
    # exact same (still-deficient) state and proposes the identical job.
    first_row = next(row for row in first.deficiencies if row.skill_id == SKILL_ID)
    second_row = next(row for row in second.deficiencies if row.skill_id == SKILL_ID)
    assert first_row == second_row


# ---------------------------------------------------------------------------
# Dry run: schema-readiness (no DDL/DML), never repairs anything
# ---------------------------------------------------------------------------


def _table_names(database) -> set[str]:
    connection = sqlite3.connect(database)
    try:
        rows = connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        ).fetchall()
        return {row[0] for row in rows}
    finally:
        connection.close()


def test_dry_run_against_uninitialized_database_reports_schema_not_ready(
    tmp_path, isolated, database
):
    """`database` (the fixture above) is a path to a SQLite file that does not exist
    yet -- exactly the case check_schema_ready() must refuse to silently repair. A
    real production run never hits this (the schema is always already provisioned by
    the time a dry run runs); this proves the stop-and-report contract for the case
    where it somehow isn't."""
    with pytest.raises(cycle.SchemaNotReadyError, match="schema_not_ready"):
        _run(database, tmp_path / "snapshots", dry_run=True)

    # Never repaired: no table was created as a side effect of the failed check.
    assert _table_names(database) == set()


def test_dry_run_against_database_missing_the_active_job_index_reports_schema_not_ready(
    tmp_path, isolated, database
):
    """A table that exists but is missing the active-job uniqueness index (e.g. an
    older or hand-created database) must also be refused, not silently patched."""
    connection = sqlite3.connect(database)
    try:
        connection.execute(
            "CREATE TABLE replenishment_jobs ("
            "job_id TEXT PRIMARY KEY, course_id TEXT, skill_id TEXT, job_type TEXT, "
            "status TEXT, requested_count INTEGER, attempts INTEGER, created_at TEXT, "
            "started_at TEXT, completed_at TEXT, next_retry_at TEXT, "
            "lease_expires_at TEXT, error_code TEXT, error_message TEXT, "
            "metadata_json TEXT)"
        )
        connection.commit()
    finally:
        connection.close()

    with pytest.raises(cycle.SchemaNotReadyError, match="ux_replenishment_jobs_active"):
        _run(database, tmp_path / "snapshots", dry_run=True)


def test_dry_run_issues_no_ddl_or_dml(tmp_path, isolated, database, reviewed_blueprint):
    """SQL-capture: every statement a dry run sends to the database, captured via
    SQLAlchemy's before_cursor_execute event at the Engine class level (so it's
    caught regardless of which engine instance run_cycle() constructs internally),
    must be a read -- never INSERT/UPDATE/DELETE/CREATE/DROP/ALTER/REPLACE/TRUNCATE."""
    _seed_schema(database)  # schema already exists, as in production

    captured: list[str] = []

    def _capture(conn, cursor, statement, parameters, context, executemany):
        captured.append(statement)

    event.listen(Engine, "before_cursor_execute", _capture)
    try:
        report = _run(database, tmp_path / "snapshots", dry_run=True)
    finally:
        event.remove(Engine, "before_cursor_execute", _capture)

    assert report.dry_run is True
    assert captured, "the scan should have issued at least one query to capture"
    forbidden_prefixes = (
        "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER", "REPLACE", "TRUNCATE",
    )
    offending = [
        statement for statement in captured
        if statement.strip().upper().startswith(forbidden_prefixes)
    ]
    assert offending == [], f"dry run issued write statement(s): {offending}"


def test_deterministic_planning_still_works_without_queue_writes(
    tmp_path, isolated, database, reviewed_blueprint
):
    """The execution plan (proposed_job_key, rank, deterministic ordering) is derived
    entirely from decide_replenishment()'s pure decisions plus
    deterministic_job_key() -- never from a row enqueue() wrote -- so it must be
    identical whether or not the repository can write at all."""
    _seed_schema(database)

    report = _run(database, tmp_path / "snapshots", dry_run=True)

    assert report.queue_rows_written == 0
    assert len(report.execution_plan) == 1
    planned = report.execution_plan[0]
    assert planned.skill_id == SKILL_ID
    assert planned.rank == 1
    proposed = next(row for row in report.deficiencies if row.skill_id == SKILL_ID)
    assert proposed.proposed_job_key == cycle.report.deterministic_job_key("intro-ai", SKILL_ID)
    assert proposed.proposed_job_key != "-"


def test_non_dry_run_still_enqueues_idempotently(
    tmp_path, isolated, database, monkeypatch
):
    provider = FakeSearchProvider([])  # no results -> lands in waiting_for_reference_review
    monkeypatch.setattr(cli, "_search_provider_factory", lambda manifest: provider)
    monkeypatch.setattr(cli, "_fetcher_factory", lambda manifest: FakePageFetcher({}))

    snapshot_root = tmp_path / "snapshots"
    _run(database, snapshot_root)

    repository = SQLiteReplenishmentJobRepository(database)
    first_ids = {
        job.skill_id: job.job_id for job in repository.list_active() if job.skill_id == SKILL_ID
    }
    assert len(first_ids) == 1

    # A second, live scan sees the same skill already in flight and enqueues
    # nothing new for it -- unchanged from cli.py scan's existing idempotent
    # behavior; the partial unique index would reject a duplicate regardless.
    _run(database, snapshot_root)
    second_ids = {
        job.skill_id: job.job_id for job in repository.list_active() if job.skill_id == SKILL_ID
    }
    assert second_ids == first_ids


# ---------------------------------------------------------------------------
# Enqueue ordering: an eligible (planned) deficiency must never lose the
# worker's FIFO claim to a blocked deficiency that merely sorts earlier in
# the taxonomy scan.
# ---------------------------------------------------------------------------

def test_worker_claims_the_planned_eligible_skill_over_an_earlier_blocked_one(
    tmp_path, isolated, database, reviewed_blueprint, monkeypatch
):
    """AI-FND-02 (real intro-ai taxonomy, scanned via the `manifest` fixture's
    skills.csv) sorts before SKILL_ID and has no reviewed blueprint here, so
    it reports as "blocked" -- exactly the shape that let a blocked skill
    silently claim a live run's single new-candidate slot ahead of the run's
    actual planned target (enqueue() used to write every should_enqueue
    decision's row in raw scan order, regardless of blocked_reason, so a
    blocked skill sorting earlier got an earlier created_at and won the
    worker's oldest-first claim)."""
    monkeypatch.setattr(cli, "_search_provider_factory", lambda manifest: FakeSearchProvider([]))
    monkeypatch.setattr(cli, "_fetcher_factory", lambda manifest: FakePageFetcher({}))
    monkeypatch.setattr(cli, "_model_factory", lambda: DeterministicFakeModel)
    monkeypatch.setattr(cli, "_reviewer_factory", lambda: clean_reviewer_factory)

    snapshot_root = tmp_path / "snapshots"
    budget = CycleBudgetConfig(max_new_candidates=1, max_generation_calls=10)
    cycle_report = _run(database, snapshot_root, budget=budget)

    # Confirm the fixture really does produce the shape this test relies on
    # before asserting on the claim order it produces.
    assert SKILL_ID in {row.skill_id for row in cycle_report.execution_plan}
    blocked_skill_ids = {
        row.skill_id for row in cycle_report.deficiencies if row.decision == "blocked"
    }
    assert "AI-FND-02" in blocked_skill_ids

    processed_skill_ids = {row.skill_id for row in cycle_report.job_outcomes}
    assert processed_skill_ids == {SKILL_ID}

    repository = SQLiteReplenishmentJobRepository(database)
    blocked_job = next(job for job in repository.list_active() if job.skill_id == "AI-FND-02")
    target_job = next(job for job in repository.list_active() if job.skill_id == SKILL_ID)
    # Still enqueued (blocked skills may still progress through reference
    # curation), but never claimed by this run's single new-candidate slot.
    assert blocked_job.status == "queued"
    assert blocked_job.attempts == 0
    assert target_job.attempts >= 1
    assert blocked_job.created_at > target_job.created_at


# ---------------------------------------------------------------------------
# New-candidate cap + interrupted-run resume
# ---------------------------------------------------------------------------

def test_new_candidate_cap_stops_after_one_and_resumes_on_the_next_run(
    tmp_path, isolated, database, monkeypatch
):
    provider = FakeSearchProvider([])  # no results -> both skills land in reference review
    monkeypatch.setattr(cli, "_search_provider_factory", lambda manifest: provider)
    monkeypatch.setattr(cli, "_fetcher_factory", lambda manifest: FakePageFetcher({}))

    snapshot_root = tmp_path / "snapshots"
    budget = CycleBudgetConfig(max_new_candidates=1)
    first = _run(database, snapshot_root, budget=budget)

    assert len(first.job_outcomes) == 1
    assert "new-candidate cap reached" in first.stop_reason

    repository = SQLiteReplenishmentJobRepository(database)
    remaining_queued = [job for job in repository.list_active() if job.status == "queued"]
    assert len(remaining_queued) >= 1  # the second deficient skill is untouched, not lost

    # A later scheduled run resumes purely from PostgreSQL/SQLite state --
    # the same database file, a fresh process-level CycleBudgetTracker.
    second = _run(database, snapshot_root, budget=budget)
    assert len(second.job_outcomes) == 1
    assert second.job_outcomes[0].job_id != first.job_outcomes[0].job_id


# ---------------------------------------------------------------------------
# Full episode reaches awaiting_content_approval, never auto-promoted
# ---------------------------------------------------------------------------

def test_candidate_reaches_awaiting_approval_and_is_snapshotted_not_promoted(
    tmp_path, isolated, database, reviewed_blueprint, monkeypatch
):
    monkeypatch.setattr(cli, "_search_provider_factory", lambda manifest: FakeSearchProvider([]))
    monkeypatch.setattr(cli, "_fetcher_factory", lambda manifest: FakePageFetcher({}))
    monkeypatch.setattr(cli, "_model_factory", lambda: DeterministicFakeModel)
    monkeypatch.setattr(cli, "_reviewer_factory", lambda: clean_reviewer_factory)

    snapshot_root = tmp_path / "snapshots"
    repository = SQLiteReplenishmentJobRepository(database)
    repository.initialize_schema()
    repository.enqueue(course_id="intro-ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)

    report = _run(database, snapshot_root, budget=CycleBudgetConfig(max_generation_calls=10))

    outcomes = {row.status for row in report.job_outcomes}
    assert "waiting_for_question_review" in outcomes
    awaiting = [row for row in report.pending_approvals if row.status == "waiting_for_question_review"]
    assert len(awaiting) == 1

    job_dir = snapshot_root / "intro-ai" / awaiting[0].job_id
    assert (job_dir / "job.json").is_file()
    assert (job_dir / "candidates.json").is_file()
    assert (job_dir / "review.json").is_file()
    assert (job_dir / "content_hashes.json").is_file()
    assert (job_dir / "budget.json").is_file()

    # Never auto-promoted: no bank file or active-bank pointer exists yet.
    assert not (tmp_path / "bank").is_dir() or not any((tmp_path / "bank").glob("*.jsonl"))
    assert not any(tmp_path.rglob("*-active-bank.json"))

    # Idempotent re-run: still awaiting, no duplicate job, snapshot unchanged in shape.
    again = _run(database, snapshot_root, budget=CycleBudgetConfig(max_generation_calls=10))
    still_awaiting = [row for row in again.pending_approvals if row.status == "waiting_for_question_review"]
    assert len(still_awaiting) == 1
    assert still_awaiting[0].job_id == awaiting[0].job_id


# ---------------------------------------------------------------------------
# Rejected candidates: reported, never retried into a new episode
# ---------------------------------------------------------------------------

def test_semantic_rejection_is_reported_and_terminal(
    tmp_path, isolated, database, reviewed_blueprint, monkeypatch
):
    monkeypatch.setattr(cli, "_search_provider_factory", lambda manifest: FakeSearchProvider([]))
    monkeypatch.setattr(cli, "_fetcher_factory", lambda manifest: FakePageFetcher({}))
    monkeypatch.setattr(cli, "_model_factory", lambda: DeterministicFakeModel)
    reviewer = FakeContentReviewer(
        {"What does a heuristic estimate?": _semantic_reject_review_result()}
    )
    monkeypatch.setattr(cli, "_reviewer_factory", lambda: (lambda: reviewer))
    monkeypatch.setattr(cli, "_review_config", lambda: ReviewPolicyConfig(shadow_mode=True))

    repository = SQLiteReplenishmentJobRepository(database)
    repository.initialize_schema()
    repository.enqueue(course_id="intro-ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)

    report = _run(database, tmp_path / "snapshots", budget=CycleBudgetConfig(max_generation_calls=10))

    rejected = [row for row in report.job_outcomes if row.status == "rejected_by_automated_review"]
    assert len(rejected) == 1
    assert rejected[0].outcome_category == "rejected"

    # Never retried into a fresh episode: a second run makes no further progress
    # on this skill (still rejected, no new job created for it).
    again = _run(database, tmp_path / "snapshots", budget=CycleBudgetConfig(max_generation_calls=10))
    assert not any(row.skill_id == SKILL_ID for row in again.job_outcomes)


# ---------------------------------------------------------------------------
# Generation-call budget exhaustion mid-episode
# ---------------------------------------------------------------------------

def test_generation_call_budget_stops_before_automated_review(
    tmp_path, isolated, database, reviewed_blueprint, monkeypatch
):
    monkeypatch.setattr(cli, "_search_provider_factory", lambda manifest: FakeSearchProvider([]))
    monkeypatch.setattr(cli, "_fetcher_factory", lambda manifest: FakePageFetcher({}))
    monkeypatch.setattr(cli, "_model_factory", lambda: DeterministicFakeModel)
    monkeypatch.setattr(cli, "_reviewer_factory", lambda: clean_reviewer_factory)

    repository = SQLiteReplenishmentJobRepository(database)
    repository.initialize_schema()
    repository.enqueue(course_id="intro-ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)

    report = _run(database, tmp_path / "snapshots", budget=CycleBudgetConfig(max_generation_calls=1))

    assert "call cap reached" in report.stop_reason
    job = repository.get(report.job_outcomes[-1].job_id)
    assert job.job_type == "automated_review"
    assert job.status == "queued"  # ready for the next run, not lost or retried from scratch


# ---------------------------------------------------------------------------
# Retention / archival
# ---------------------------------------------------------------------------

def test_archive_stale_jobs_compacts_old_terminal_snapshots(tmp_path):
    repository = SQLiteReplenishmentJobRepository(tmp_path / "jobs.sqlite3")
    repository.initialize_schema()
    old_time = FIXED_TIME - timedelta(days=30)
    job = repository.enqueue(
        course_id="intro-ai", skill_id=SKILL_ID, requested_count=1, clock=lambda: old_time
    )
    repository.claim_next(clock=lambda: old_time)
    repository.mark_permanent_failure(job.job_id, error_code="x", error_message="y")

    snapshot_root = tmp_path / "snapshots"
    job_dir = snapshot_root / "intro-ai" / job.job_id
    job_dir.mkdir(parents=True)
    (job_dir / "candidates.json").write_text("[]", encoding="utf-8")
    (job_dir / "job.json").write_text("{}", encoding="utf-8")

    archived = archive_stale_jobs(
        snapshot_root, repository, retention_days=14, clock=fixed_clock
    )
    assert archived == [job_dir]
    assert {p.name for p in job_dir.iterdir()} == {"archived.json"}
    summary = json.loads((job_dir / "archived.json").read_text())
    assert summary["final_status"] == "permanent_failure"

    # Idempotent: a second pass leaves the already-archived directory alone.
    again = archive_stale_jobs(snapshot_root, repository, retention_days=14, clock=fixed_clock)
    assert again == []


# ---------------------------------------------------------------------------
# Run-scoped artifact manifest: excludes stale pre-existing content-ops
# artifacts, preserves them on disk untouched
# ---------------------------------------------------------------------------

def test_dry_run_manifest_excludes_preexisting_snapshot_directory(
    tmp_path, isolated, database, monkeypatch
):
    """A dry run never processes any job (job_outcomes is always empty), so
    its manifest must list only the two report files -- never a job-scoped
    directory left on disk from an earlier live run's commit to the
    content-ops branch, even though that directory sits right there under
    the same snapshot_root this run reads and writes."""

    def _boom(*args, **kwargs):
        raise AssertionError("dry run must never construct a search/model/reviewer provider")

    monkeypatch.setattr(cli, "_search_provider_factory", _boom)
    monkeypatch.setattr(cli, "_fetcher_factory", _boom)
    monkeypatch.setattr(cli, "_model_factory", _boom)
    monkeypatch.setattr(cli, "_reviewer_factory", _boom)
    _seed_schema(database)

    snapshot_root = tmp_path / "snapshots"
    stale_job_dir = snapshot_root / "dsa" / "stale-job-from-an-earlier-run"
    stale_job_dir.mkdir(parents=True)
    (stale_job_dir / "job.json").write_text('{"job_id": "stale"}', encoding="utf-8")

    exit_code = cycle.main([
        "--database", str(database),
        "--snapshot-root", str(snapshot_root),
        "--report-dir", str(snapshot_root / "_reports"),
        "--dry-run",
    ])
    assert exit_code == 0

    manifest_data = json.loads(
        (snapshot_root / "_reports" / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest_data["dry_run"] is True
    assert manifest_data["paths"] == ["_reports/latest.json", "_reports/latest.md"]
    assert "dsa/stale-job-from-an-earlier-run" not in manifest_data["paths"]

    # Never touched: the stale directory is exactly as it was before this run.
    assert (stale_job_dir / "job.json").read_text(encoding="utf-8") == '{"job_id": "stale"}'


def test_live_run_manifest_and_staged_output_include_only_this_runs_job(
    tmp_path, isolated, database, reviewed_blueprint, monkeypatch
):
    """A live run's manifest (and the staged copy built from it) must include
    this run's own job-scoped directory but exclude an unrelated job-scoped
    directory already present under snapshot_root from an earlier run --
    proving both the manifest computation and scripts.stage_run_artifacts
    exclude stale pre-existing files end to end."""
    monkeypatch.setattr(cli, "_search_provider_factory", lambda manifest: FakeSearchProvider([]))
    monkeypatch.setattr(cli, "_fetcher_factory", lambda manifest: FakePageFetcher({}))
    monkeypatch.setattr(cli, "_model_factory", lambda: DeterministicFakeModel)
    monkeypatch.setattr(cli, "_reviewer_factory", lambda: clean_reviewer_factory)

    snapshot_root = tmp_path / "snapshots"
    stale_job_dir = snapshot_root / "dsa" / "stale-job-from-an-earlier-run"
    stale_job_dir.mkdir(parents=True)
    (stale_job_dir / "job.json").write_text('{"job_id": "stale"}', encoding="utf-8")

    repository = SQLiteReplenishmentJobRepository(database)
    repository.initialize_schema()
    repository.enqueue(course_id="intro-ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)

    # Cap at exactly one new candidate so only SKILL_ID's pre-enqueued job
    # (oldest created_at, claimed first) is ever started this run -- the
    # real intro-ai taxonomy has other deficient skills too, and this test
    # only cares about isolating a single job's directory.
    budget = CycleBudgetConfig(max_new_candidates=1, max_generation_calls=10)
    cycle_report = _run(database, snapshot_root, budget=budget)
    assert cycle_report.job_outcomes, "expected at least one tick for the capped candidate"
    job_ids = {row.job_id for row in cycle_report.job_outcomes}
    assert len(job_ids) == 1  # the max_new_candidates=1 cap allows only one job_id
    this_run_job_id = next(iter(job_ids))

    report_dir = snapshot_root / "_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "latest.json").write_text("{}", encoding="utf-8")
    (report_dir / "latest.md").write_text("# report\n", encoding="utf-8")
    manifest_paths = cycle._run_manifest_paths(cycle_report, report_dir, snapshot_root)
    cycle._write_run_manifest(report_dir, snapshot_root, dry_run=False, paths=manifest_paths)

    assert f"intro-ai/{this_run_job_id}" in manifest_paths
    assert "dsa/stale-job-from-an-earlier-run" not in manifest_paths

    dest = tmp_path / "staged"
    staged = stage_run_artifacts(report_dir / "run_manifest.json", snapshot_root, dest)
    assert dest / "intro-ai" / this_run_job_id in staged
    assert (dest / "intro-ai" / this_run_job_id / "job.json").is_file()
    assert (dest / "_reports" / "run_manifest.json").is_file()
    assert not (dest / "dsa").exists()

    # Never touched: the stale directory on the real content-ops checkout
    # is exactly as it was before this run, both in and outside the manifest.
    assert (stale_job_dir / "job.json").read_text(encoding="utf-8") == '{"job_id": "stale"}'


# ---------------------------------------------------------------------------
# Blueprint-capacity-aware planning: the intro-ai/AI-FND-03 incident and its
# generalizations (worker.py's demand_already_satisfied -> no_longer_needed,
# never permanent_failure; the scanner must never enqueue a skill whose one
# resolvable blueprint has zero unsatisfied intent slots left).
# ---------------------------------------------------------------------------

def _isolate_with_thresholds(manifest, monkeypatch, *, low_supply_threshold, target_supply):
    """Like the `isolated` fixture, but with course-wide policy thresholds
    this test controls directly, and scoped to a single-skill taxonomy
    (SKILL_ID only) rather than the shared `manifest` fixture's copy of the
    real, ~40-skill intro-ai skills.csv. Two independent reasons for the
    swap: (1) the shared fixture's low_supply_threshold=target_supply=1 can
    never reproduce the AI-FND-03 shape (a skill deficient by the
    skill-level bank count, yet already at its blueprint's ceiling) -- that
    needs target_supply strictly greater than low_supply_threshold, so
    supply can sit between the two; (2) with a lower threshold, every other
    real skill with zero bank items also turns deficient-and-blocked, and
    (being enqueued earlier in scan order) would win these tests'
    max_new_candidates=1 worker slot instead of SKILL_ID. Reuses every
    other path from `manifest` (bank, candidate store, review store) so
    fixtures built against it (reviewed_blueprint, approved_candidate)
    still point at valid, matching locations."""
    taxonomy_dir = manifest.taxonomy_path.parent / "minimal_taxonomy"
    taxonomy_dir.mkdir(exist_ok=True)
    (taxonomy_dir / "skills.csv").write_text(
        "skill_id,topic,subtopic,name,learning_objective,cognitive_process,"
        "generation_strategy,template_id,prerequisite_skill_ids\n"
        f"{SKILL_ID},Search and Problem Solving,Informed search,Heuristic "
        "function,Explain how a heuristic estimates the remaining cost from "
        "a state to the goal.,understand,generated,,\n",
        encoding="utf-8",
    )
    (taxonomy_dir / "references.csv").write_text(
        "skill_id,reference_material\n", encoding="utf-8"
    )
    scoped = manifest.model_copy(
        update={
            "taxonomy_path": taxonomy_dir,
            "low_supply_threshold": low_supply_threshold,
            "target_supply": target_supply,
        }
    )
    monkeypatch.setattr(cycle, "load_preparation_eligible_manifests", lambda: [scoped])
    return scoped


def _write_bank(manifest, item_ids: list[str], *, skill_id: str = SKILL_ID) -> None:
    """A minimal approved-bank JSONL with exactly these item_ids -- simulates
    "these blueprint slots are already fulfilled" without running promotion
    for real. Mirrors tests/test_replenishment_worker.py's helper of the
    same name."""
    path = active_bank_path(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for item_id in item_ids:
            item = BankItem(
                item_id=item_id,
                skill_id=skill_id,
                provenance="generated",
                question=QuizQuestion(
                    question=f"Placeholder question for {item_id}?",
                    options=["A", "B", "C", "D"],
                    correct_answer="A",
                    explanation="Placeholder explanation.",
                    concept="placeholder",
                    difficulty="introductory",
                ),
            )
            handle.write(json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n")


def test_capacity_exhausted_skill_is_never_enqueued_and_reported_distinctly(
    tmp_path, manifest, database, reviewed_blueprint, monkeypatch
):
    """The exact intro-ai/AI-FND-03 production shape (run 32094990672 /
    32096315756): a blueprint whose only declared intent for SKILL_ID is
    already an approved bank item, while the course-wide policy
    (target_supply=6, threshold=3) still reports the skill as deficient by
    the skill-level bank count alone. Must be reported as
    "capacity_exhausted" -- distinct from "blocked"/"no_deficiency"/
    "proposed" -- and never enqueued, in both a dry run and a live run."""
    scoped_manifest = _isolate_with_thresholds(
        manifest, monkeypatch, low_supply_threshold=3, target_supply=6
    )
    _write_bank(scoped_manifest, [question_id(reviewed_blueprint.batch_id, SKILL_ID, 0)])
    _seed_schema(database)

    dry_report = _run(database, tmp_path / "snapshots", dry_run=True)
    row = next(r for r in dry_report.deficiencies if r.skill_id == SKILL_ID)
    assert row.decision == "capacity_exhausted"
    assert "capacity_exhausted:" in row.reason
    assert "demand_fingerprint=" in row.reason
    assert row.proposed_job_key == "-"
    assert not any(planned.skill_id == SKILL_ID for planned in dry_report.execution_plan)

    # The real intro-ai skills.csv this fixture copies has other genuinely
    # deficient (blocked, no blueprint) skills too -- those still get
    # enqueued and processed by design, so fakes are needed for the live
    # run below; the assertion of interest is scoped to SKILL_ID alone.
    monkeypatch.setattr(cli, "_search_provider_factory", lambda manifest: FakeSearchProvider([]))
    monkeypatch.setattr(cli, "_fetcher_factory", lambda manifest: FakePageFetcher({}))
    monkeypatch.setattr(cli, "_model_factory", lambda: DeterministicFakeModel)
    monkeypatch.setattr(cli, "_reviewer_factory", lambda: clean_reviewer_factory)

    live_report = _run(database, tmp_path / "snapshots", dry_run=False)
    assert not any(
        row.skill_id == SKILL_ID and row.decision in ("enqueued", "proposed")
        for row in live_report.deficiencies
    )
    repository = SQLiteReplenishmentJobRepository(database)
    assert repository.latest_for_skill(scoped_manifest.course_id, SKILL_ID) is None


def test_multi_intent_capacity_exhausted_pattern_mirrors_the_latent_course_case(
    tmp_path, manifest, database, monkeypatch
):
    """The DSA/Linear-Algebra/Database-Systems latent pattern found during
    investigation: a blueprint declaring several (not just one) intents for
    a skill, all already approved. supply(4) sits below a stricter
    threshold(5) here specifically to force deficiency while still at
    blueprint capacity -- proving the mechanism generalizes beyond the
    single-intent AI-FND-03 case, not a special-cased N=1 check."""
    scoped_manifest = _isolate_with_thresholds(
        manifest, monkeypatch, low_supply_threshold=5, target_supply=6
    )
    blueprint_dir = tmp_path / "blueprints-multi"
    blueprint_dir.mkdir()
    intents = [
        QuestionIntent(
            intent_id=f"{SKILL_ID}-INT-{n:02d}", skill_id=SKILL_ID,
            assessment_focus=f"facet {n}", question_archetype="definition recall",
            preferred_reference_ids=["placeholder-ref"], required_concepts=["heuristic"],
            prohibited_conflations=["placeholder conflation"], difficulty="introductory",
        )
        for n in range(1, 5)
    ]
    blueprint = PilotBlueprint(
        batch_id="test-batch-multi", prompt_version=question_intents.PILOT_PROMPT_VERSION,
        review_status="blueprint-approved", reviewer_id="albert", reviewed_at=FIXED_TIME,
        base_seed=1, intents=intents,
    )
    (blueprint_dir / "test-batch-multi.json").write_text(
        json.dumps(blueprint.model_dump(mode="json")), encoding="utf-8"
    )
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", blueprint_dir)
    monkeypatch.setattr(
        grounding_briefs, "PILOT_GROUNDING_BRIEFS",
        {SKILL_ID: CanonicalGroundingBrief(
            skill_id=SKILL_ID, version="test-v1",
            statements=["A heuristic estimates the remaining cost to the goal."],
        )},
    )
    _write_bank(
        scoped_manifest,
        [question_id("test-batch-multi", SKILL_ID, index) for index in range(4)],
    )
    _seed_schema(database)

    dry_report = _run(database, tmp_path / "snapshots", dry_run=True)
    row = next(r for r in dry_report.deficiencies if r.skill_id == SKILL_ID)
    assert row.decision == "capacity_exhausted"
    assert "4 intent slot(s)" in row.reason

    monkeypatch.setattr(cli, "_search_provider_factory", lambda manifest: FakeSearchProvider([]))
    monkeypatch.setattr(cli, "_fetcher_factory", lambda manifest: FakePageFetcher({}))
    monkeypatch.setattr(cli, "_model_factory", lambda: DeterministicFakeModel)
    monkeypatch.setattr(cli, "_reviewer_factory", lambda: clean_reviewer_factory)

    live_report = _run(database, tmp_path / "snapshots", dry_run=False)
    assert not any(
        row.skill_id == SKILL_ID and row.decision in ("enqueued", "proposed")
        for row in live_report.deficiencies
    )
    repository = SQLiteReplenishmentJobRepository(database)
    assert repository.latest_for_skill(scoped_manifest.course_id, SKILL_ID) is None


def test_capacity_exhausted_skill_becomes_reenqueueable_after_blueprint_gains_capacity(
    tmp_path, manifest, database, reviewed_blueprint, monkeypatch
):
    """A skill correctly withheld as capacity_exhausted must become eligible
    again, without any code path treating the earlier outcome as a
    permanent lockout, once a human authors an additional intent for it --
    proving the fingerprint-relevant inputs (blueprint content) actually
    drive re-eligibility, not a cached decision."""
    scoped_manifest = _isolate_with_thresholds(
        manifest, monkeypatch, low_supply_threshold=3, target_supply=6
    )
    approved_item_ids = {question_id(reviewed_blueprint.batch_id, SKILL_ID, 0)}
    _write_bank(scoped_manifest, list(approved_item_ids))
    _seed_schema(database)
    monkeypatch.setattr(cli, "_search_provider_factory", lambda manifest: FakeSearchProvider([]))
    monkeypatch.setattr(cli, "_fetcher_factory", lambda manifest: FakePageFetcher({}))
    monkeypatch.setattr(cli, "_model_factory", lambda: DeterministicFakeModel)
    monkeypatch.setattr(cli, "_reviewer_factory", lambda: clean_reviewer_factory)

    snapshot_root = tmp_path / "snapshots"
    first = _run(database, snapshot_root, dry_run=False)
    row = next(r for r in first.deficiencies if r.skill_id == SKILL_ID)
    assert row.decision == "capacity_exhausted"
    repository = SQLiteReplenishmentJobRepository(database)
    assert repository.latest_for_skill(scoped_manifest.course_id, SKILL_ID) is None

    blueprint_path = (
        question_intents.BLUEPRINT_DIRECTORY / f"{reviewed_blueprint.batch_id}.json"
    )
    grown = reviewed_blueprint.model_copy(update={
        "intents": [
            *reviewed_blueprint.intents,
            QuestionIntent(
                intent_id=f"{SKILL_ID}-INT-02", skill_id=SKILL_ID,
                assessment_focus="second facet", question_archetype="definition recall",
                # Reuses the original intent's own reference id -- generation
                # validates preferred_reference_ids against approved
                # candidates, and approved_candidate only ever approved one.
                preferred_reference_ids=reviewed_blueprint.intents[0].preferred_reference_ids,
                required_concepts=["heuristic"],
                # Must match reviewed_blueprint's original intent's difficulty
                # ("intermediate") -- _blueprint_generation_difficulty requires
                # every intent for a skill to declare the same one explicitly.
                prohibited_conflations=["placeholder conflation"], difficulty="intermediate",
            ),
        ],
    })
    blueprint_path.write_text(json.dumps(grown.model_dump(mode="json")), encoding="utf-8")

    fp_before = compute_demand_fingerprint(
        reviewed_blueprint, SKILL_ID, approved_item_ids,
        difficulty="intermediate", target_supply=6,
    )
    fp_after = compute_demand_fingerprint(
        grown, SKILL_ID, approved_item_ids,
        difficulty="intermediate", target_supply=6,
    )
    assert fp_before != fp_after

    second = _run(database, snapshot_root, dry_run=True)
    row2 = next(r for r in second.deficiencies if r.skill_id == SKILL_ID)
    assert row2.decision == "proposed"

    third = _run(database, snapshot_root, dry_run=False)
    assert repository.latest_for_skill(scoped_manifest.course_id, SKILL_ID) is not None
    assert any(job.skill_id == SKILL_ID for job in repository.list_active())


def test_genuine_permanent_failure_still_permanently_blocks_reenqueue(
    tmp_path, manifest, database, reviewed_blueprint, monkeypatch
):
    """A genuine error (two blueprints ambiguously covering the same skill,
    unrelated to demand/capacity) must keep the skill locked out of
    re-enqueueing across scans exactly as before -- only the new
    no_longer_needed outcome is exempt from this lockout, never a real
    failure like ambiguous_blueprint_selection."""
    scoped_manifest = _isolate_with_thresholds(
        manifest, monkeypatch, low_supply_threshold=3, target_supply=6
    )
    blueprint_dir = question_intents.BLUEPRINT_DIRECTORY
    second_blueprint = PilotBlueprint(
        batch_id="test-batch-ambiguous", prompt_version=question_intents.PILOT_PROMPT_VERSION,
        review_status="blueprint-approved", reviewer_id="albert", reviewed_at=FIXED_TIME,
        base_seed=2,
        intents=[QuestionIntent(
            intent_id=f"{SKILL_ID}-INT-02", skill_id=SKILL_ID,
            assessment_focus="second", question_archetype="definition recall",
            preferred_reference_ids=["placeholder-ref"], required_concepts=["heuristic"],
            prohibited_conflations=["placeholder conflation"], difficulty="introductory",
        )],
    )
    (blueprint_dir / "test-batch-ambiguous.json").write_text(
        json.dumps(second_blueprint.model_dump(mode="json")), encoding="utf-8"
    )
    monkeypatch.setattr(cli, "_search_provider_factory", lambda manifest: FakeSearchProvider([]))
    monkeypatch.setattr(cli, "_fetcher_factory", lambda manifest: FakePageFetcher({}))
    monkeypatch.setattr(cli, "_model_factory", lambda: DeterministicFakeModel)
    monkeypatch.setattr(cli, "_reviewer_factory", lambda: clean_reviewer_factory)

    snapshot_root = tmp_path / "snapshots"
    budget = CycleBudgetConfig(max_new_candidates=1, max_generation_calls=10)
    first = _run(database, snapshot_root, budget=budget)

    job = next(row for row in first.job_outcomes if row.skill_id == SKILL_ID)
    assert job.status == "permanent_failure"
    repository = SQLiteReplenishmentJobRepository(database)
    stored = repository.get(job.job_id)
    assert stored.error_code == "ambiguous_blueprint_selection"

    inventory = compute_course_inventory(scoped_manifest, repository)
    assert inventory[SKILL_ID].readiness == "replenishment_failed"

    second = _run(database, snapshot_root, budget=budget)
    assert not any(row.skill_id == SKILL_ID for row in second.job_outcomes)
    latest = repository.latest_for_skill(scoped_manifest.course_id, SKILL_ID)
    assert latest.job_id == job.job_id  # no new job created; still the same failed one
    assert latest.status == "permanent_failure"


def test_scan_to_worker_race_resolves_safely_across_two_runs(
    tmp_path, manifest, database, reviewed_blueprint, monkeypatch
):
    """Capacity is available when this run's scan enqueues and first ticks
    the job (retrieve_references succeeds via the already-approved
    reference candidate, advancing to generate_questions); the blueprint's
    last slot is then filled by someone else before the next run's tick.
    That next run's generate_questions-stage recheck must resolve to
    no_longer_needed with zero model calls -- never permanent_failure, and
    never a wasted generation call."""
    scoped_manifest = _isolate_with_thresholds(
        manifest, monkeypatch, low_supply_threshold=3, target_supply=6
    )
    monkeypatch.setattr(cli, "_search_provider_factory", lambda manifest: FakeSearchProvider([]))
    monkeypatch.setattr(cli, "_fetcher_factory", lambda manifest: FakePageFetcher({}))

    def _boom_model_factory():
        raise AssertionError("model must not be called once demand is satisfied")

    monkeypatch.setattr(cli, "_model_factory", lambda: _boom_model_factory)
    monkeypatch.setattr(cli, "_reviewer_factory", lambda: clean_reviewer_factory)

    snapshot_root = tmp_path / "snapshots"
    first = _run(
        database, snapshot_root,
        budget=CycleBudgetConfig(max_new_candidates=1, max_ticks=1),
    )
    assert len(first.job_outcomes) == 1
    job_id = first.job_outcomes[0].job_id
    repository = SQLiteReplenishmentJobRepository(database)
    assert repository.get(job_id).job_type == "generate_questions"  # advanced past retrieval

    _write_bank(scoped_manifest, [question_id(reviewed_blueprint.batch_id, SKILL_ID, 0)])

    second = _run(
        database, snapshot_root,
        budget=CycleBudgetConfig(max_new_candidates=1, max_generation_calls=10),
    )
    outcome = next(row for row in second.job_outcomes if row.job_id == job_id)
    assert outcome.status == "no_longer_needed"
    stored = repository.get(job_id)
    assert stored.error_code == "demand_already_satisfied"
    assert stored.metadata["demand_fingerprint"]
