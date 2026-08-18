"""End-to-end stage walk through the replenishment worker, fakes only.

Mirrors the fixture shapes already established in tests/test_retrieval.py
(FakeSearchProvider/FakePageFetcher) and tests/test_grounded_batch.py
(DeterministicFakeModel implementing the BatchModel protocol), so nothing
here reaches the network or a real model.
"""

import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pytest

import authoring.grounding_briefs as grounding_briefs
import authoring.question_intents as question_intents
from api.bank import BankItem
from api.schemas import QuizQuestion
from authoring.grounded_batch import (
    BatchConfig,
    BatchGenerationError,
    PendingQuestion,
    generate_batch,
    question_id,
    read_jsonl,
)
from authoring.grounded_review import (
    GroundedReviewStore,
    RevisionProvenance,
    approve_as_written,
    approve_revision,
    approved_item_provenance,
    build_pending_review,
    list_pending,
    load_source_questions,
    propose_revision,
    question_content_hash,
    reject_item,
)
from authoring.grounding_briefs import CanonicalGroundingBrief
from authoring.question_intents import PilotBlueprint, QuestionIntent
from authoring.replenishment.jobs import ReplenishmentJob, SQLiteReplenishmentJobRepository
from authoring.replenishment.manifest import CourseManifest, active_bank_path
from authoring.replenishment.worker import (
    AmbiguousBlueprintError,
    ModelUnavailableError,
    _blueprint_generation_difficulty,
    blueprints_covering_skill,
    process_job,
    process_one,
    ready_to_resume,
    resolve_job_blueprint,
)
from authoring.review.config import ReviewPolicyConfig
from authoring.review.models import (
    AnswerAssessment,
    AutomatedReviewReport,
    DeterministicChecks,
    DifficultyAssessment,
    DuplicateAssessment,
    GroundingAssessment,
    ObjectiveAssessment,
    SemanticReviewResult,
)
from authoring.review.reports import AutomatedReviewReportStore, review_report_path
from authoring.review.reviewer import FakeContentReviewer, ReviewerUnavailableError
from scripts.import_reference_candidates import import_candidates
from authoring.retrieval.models import ReferenceCandidate, SearchResult, approve, new_candidate
from authoring.retrieval.search import FetchedPage
from authoring.retrieval.store import CandidateStore

FIXED_TIME = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
SKILL_ID = "AI-SRC-08"
REPO_ROOT = Path(__file__).resolve().parents[1]

PASSAGE = (
    "A heuristic function estimates the cost of the cheapest path from a given "
    "state to a goal state. It lets an informed search order the frontier by "
    "how promising a state looks rather than by how far it already is, which "
    "is what separates informed search from uninformed search."
)


def fixed_clock():
    return FIXED_TIME


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
    def __init__(self, pages: dict[str, str]):
        self.pages = pages

    def fetch(self, url: str) -> FetchedPage:
        return FetchedPage(url=url, text=self.pages[url])


class DeterministicFakeModel:
    model_id = "fake-grounded-model"
    model_revision = "fake-revision-1"

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def generate(self, messages, seed, generation_parameters):
        self.calls.append({"messages": messages, "seed": seed})
        if self.responses:
            response = self.responses.pop(0)
            return response(seed) if callable(response) else response
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
                        "explanation": (
                            "A heuristic estimates the cheapest remaining path cost."
                        ),
                        "concept": "Heuristics",
                        "difficulty": "intermediate",
                    }
                ]
            }
        )


def _clean_review_result(correct_answer: str) -> SemanticReviewResult:
    """A reviewer pass that finds nothing wrong -- used wherever a test only cares
    about the pipeline reaching human review, not automated review's own judgment."""
    return SemanticReviewResult(
        grounding_assessment=GroundingAssessment(
            grounded=True,
            independently_supported_answer=True,
            supporting_reference_ids=[],
            grounding_confidence=0.95,
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


def clean_reviewer_factory():
    """Reviewer that finds nothing wrong with DeterministicFakeModel's default question."""
    return FakeContentReviewer(
        {"What does a heuristic estimate?": _clean_review_result("Remaining cost to goal")}
    )


class ModelDownFakeModel:
    model_id = "fake-grounded-model"
    model_revision = "fake-revision-1"

    def generate(self, messages, seed, generation_parameters):
        raise ModelUnavailableError("no GPU available in this test environment")


@pytest.fixture
def manifest(tmp_path):
    taxonomy_dir = tmp_path / "taxonomy"
    taxonomy_dir.mkdir()
    shutil.copy(REPO_ROOT / "taxonomy/data/ai/skills.csv", taxonomy_dir / "skills.csv")
    (taxonomy_dir / "references.csv").write_text("skill_id,reference_material\n", encoding="utf-8")

    return CourseManifest(
        course_id="ai",
        title="test",
        version="1",
        taxonomy_path=taxonomy_dir,
        approved_bank_path=tmp_path / "bank" / "ai-bank-v0.jsonl",
        bkt_model_path=tmp_path / "model.pkl",
        candidate_store_path=tmp_path / "reference_candidates.json",
        review_store_path=tmp_path / "reviews",
        allowed_domains=("example.edu",),
        low_supply_threshold=3,
        target_supply=6,
        default_bkt_model_version="test-v1",
        status="active",
    )


@pytest.fixture
def repository():
    repository = SQLiteReplenishmentJobRepository(":memory:")
    repository.initialize_schema()
    yield repository
    repository.close()


@pytest.fixture
def approved_candidate(manifest) -> ReferenceCandidate:
    candidate = new_candidate(
        SKILL_ID,
        "Heuristics",
        "https://example.edu/pathfinding/heuristics.html",
        "example.edu",
        PASSAGE,
        FIXED_TIME,
        relevance_score=20,
        matched_terms=["heuristic", "frontier"],
    )
    store = CandidateStore(manifest.candidate_store_path)
    store.add([candidate])
    store.replace(approve(candidate, "albert", reviewed_at=FIXED_TIME))
    return candidate


@pytest.fixture
def reviewed_blueprint(tmp_path, monkeypatch, approved_candidate):
    """A scratch, reviewed intent blueprint + grounding brief for SKILL_ID --
    never the real authoring/blueprints/ or grounding_briefs.py data, so this
    never depends on or mutates Albert's actual authored pedagogy."""
    blueprint_dir = tmp_path / "blueprints"
    blueprint_dir.mkdir()
    intent = QuestionIntent(
        intent_id=f"{SKILL_ID}-INT-01",
        skill_id=SKILL_ID,
        assessment_focus="What a heuristic estimates",
        question_archetype="definition recall",
        preferred_reference_ids=[approved_candidate.candidate_id],
        required_concepts=["heuristic", "estimate"],
        prohibited_conflations=["heuristic equals exact cost"],
        difficulty="intermediate",
    )
    blueprint = PilotBlueprint(
        batch_id="test-batch-01",
        prompt_version=question_intents.PILOT_PROMPT_VERSION,
        review_status="blueprint-approved",
        reviewer_id="albert",
        reviewed_at=FIXED_TIME,
        base_seed=1,
        intents=[intent],
    )
    (blueprint_dir / "test-batch-01.json").write_text(
        json.dumps(blueprint.model_dump(mode="json")), encoding="utf-8"
    )
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", blueprint_dir)
    monkeypatch.setattr(
        grounding_briefs,
        "PILOT_GROUNDING_BRIEFS",
        {
            SKILL_ID: CanonicalGroundingBrief(
                skill_id=SKILL_ID,
                version="test-v1",
                statements=["A heuristic estimates the remaining cost to the goal."],
            )
        },
    )
    return blueprint


def _approve_first_pending_question(review_path: Path, output_dir: Path) -> str:
    review = GroundedReviewStore(review_path).load()
    item = review.items[0]
    source = next(
        q
        for q in read_jsonl(output_dir / "pending_questions.jsonl", PendingQuestion)
        if q.question_id == item.original_question_id
    )
    revised = QuizQuestion.model_validate(
        source.question.model_dump()
        | {"explanation": source.question.explanation + " (reviewed)"}
    )
    proposed = propose_revision(
        item,
        source.question,
        revised,
        "albert",
        "reviewer restated wording",
        edited_at=FIXED_TIME,
        provenance=RevisionProvenance.from_source(source),
    )
    GroundedReviewStore(review_path).replace_item(proposed)
    revision_id = proposed.revisions[0].revision_id
    approved = approve_revision(proposed, revision_id, "albert", reviewed_at=FIXED_TIME)
    GroundedReviewStore(review_path).replace_item(approved)
    return item.original_question_id


def test_full_pipeline_walk_reaches_a_promoted_bank(manifest, repository, reviewed_blueprint, approved_candidate):
    repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)

    retrieve_job = repository.claim_next(clock=fixed_clock)
    process_job(
        retrieve_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    after_retrieve = repository.get(retrieve_job.job_id)
    assert after_retrieve.status == "queued"
    assert after_retrieve.job_type == "generate_questions"

    generate_job = repository.claim_next(clock=fixed_clock)
    process_job(
        generate_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    after_generate_stage = repository.get(generate_job.job_id)
    assert after_generate_stage.status == "queued"
    assert after_generate_stage.job_type == "automated_review"

    automated_review_job = repository.claim_next(clock=fixed_clock)
    process_job(
        automated_review_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel,
        reviewer_factory=clean_reviewer_factory,
        clock=fixed_clock,
    )
    after_generate = repository.get(generate_job.job_id)
    assert after_generate.status == "waiting_for_question_review"
    review_path = Path(after_generate.metadata["review_path"])
    output_dir = review_path.parent.parent / "batches" / "test-batch-01__AI-SRC-08"

    assert ready_to_resume(after_generate, manifest) is False
    question_id = _approve_first_pending_question(review_path, output_dir)
    assert ready_to_resume(after_generate, manifest) is True

    repository.mark_queued(generate_job.job_id, job_type="promote_approved_items")
    promote_job = repository.claim_next(clock=fixed_clock)
    process_job(
        promote_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    after_promote = repository.get(promote_job.job_id)
    assert after_promote.status == "completed"

    pointer_path = manifest.approved_bank_path.parent / "ai-active-bank.json"
    pointer = json.loads(pointer_path.read_text())
    assert pointer["version"] == 1
    bank_items = [
        json.loads(line) for line in Path(pointer["path"]).read_text().splitlines()
    ]
    assert len(bank_items) == 1
    assert bank_items[0]["skill_id"] == SKILL_ID
    assert "(reviewed)" in bank_items[0]["question"]["explanation"]

    # Provenance survives end to end: reviewer, timestamp, model, prompt, references.
    review = GroundedReviewStore(review_path).load()
    provenance = approved_item_provenance(review, question_id)
    assert provenance.reviewer_id == "albert"
    assert provenance.model_id == DeterministicFakeModel.model_id
    assert provenance.model_revision == DeterministicFakeModel.model_revision
    assert provenance.prompt_version == reviewed_blueprint.prompt_version
    assert provenance.reference_ids == [approved_candidate.candidate_id]
    assert provenance.changed_fields == ["explanation"]


def test_generation_uses_only_approved_references(manifest, repository, tmp_path, monkeypatch):
    # A pending (not approved) candidate must never reach generation: without
    # an approved reference the retrieve stage stays at the review gate. Isolated
    # from the real authoring/blueprints/ directory (unrelated to this test), which
    # now legitimately carries more than one blueprint covering SKILL_ID.
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", tmp_path / "no-blueprints")
    candidate = new_candidate(
        SKILL_ID, "Heuristics", "https://example.edu/x.html", "example.edu",
        PASSAGE, FIXED_TIME, relevance_score=20,
    )
    CandidateStore(manifest.candidate_store_path).add([candidate])

    repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)
    job = repository.claim_next(clock=fixed_clock)
    process_job(
        job, manifest, job_repository=repository,
        search_provider=FakeSearchProvider([]), fetcher=FakePageFetcher({}),
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    after = repository.get(job.job_id)
    assert after.status == "waiting_for_reference_review"
    assert after.job_type == "retrieve_references"


def test_missing_blueprint_is_a_permanent_failure(
    manifest, repository, approved_candidate, tmp_path, monkeypatch
):
    # An empty blueprint directory: no reviewed intents exist for this skill
    # anywhere, unlike the real repo's authoring/blueprints/.
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", tmp_path / "empty-blueprints")
    repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)
    retrieve_job = repository.claim_next(clock=fixed_clock)
    process_job(
        retrieve_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    generate_job = repository.claim_next(clock=fixed_clock)
    process_job(
        generate_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    after = repository.get(generate_job.job_id)
    assert after.status == "permanent_failure"
    assert after.error_code == "missing_intent_blueprint"
    assert blueprints_covering_skill(SKILL_ID) == []


def _intent(
    skill_id: str, n: int, difficulty, *, preferred_reference_ids: list[str] | None = None
) -> QuestionIntent:
    return QuestionIntent(
        intent_id=f"{skill_id}-INT-{n:02d}",
        skill_id=skill_id,
        assessment_focus="What a heuristic estimates",
        question_archetype="definition recall",
        preferred_reference_ids=preferred_reference_ids or [f"{skill_id}-ref-{n}"],
        required_concepts=["heuristic"],
        prohibited_conflations=["heuristic equals exact cost"],
        difficulty=difficulty,
    )


def _introductory_response(question_text: str) -> str:
    return json.dumps(
        {
            "questions": [
                {
                    "question": question_text,
                    "options": ["Remaining cost to goal", "Total memory used", "Number of nodes", "Branching factor"],
                    "correct_answer": "Remaining cost to goal",
                    "explanation": "A heuristic estimates the cheapest remaining path cost.",
                    "concept": "Heuristics",
                    "difficulty": "introductory",
                }
            ]
        }
    )


def _intermediate_response(question_text: str) -> str:
    return json.dumps(
        {
            "questions": [
                {
                    "question": question_text,
                    "options": ["Remaining cost to goal", "Total memory used", "Number of nodes", "Branching factor"],
                    "correct_answer": "Remaining cost to goal",
                    "explanation": "A heuristic estimates the cheapest remaining path cost.",
                    "concept": "Heuristics",
                    "difficulty": "intermediate",
                }
            ]
        }
    )


@pytest.mark.parametrize("difficulty", ["introductory", "intermediate", "advanced"])
def test_blueprint_generation_difficulty_retains_each_declared_value(difficulty):
    """The blueprint is authoritative: whatever an intent pool declares comes back
    unchanged -- never silently coerced to BatchConfig's own "intermediate" default."""
    intents = [_intent(SKILL_ID, 1, difficulty), _intent(SKILL_ID, 2, difficulty)]
    assert _blueprint_generation_difficulty(SKILL_ID, intents) == difficulty


def test_blueprint_generation_difficulty_rejects_missing_declared_difficulty():
    intents = [_intent(SKILL_ID, 1, "introductory"), _intent(SKILL_ID, 2, None)]
    with pytest.raises(BatchGenerationError, match="consistent, explicit"):
        _blueprint_generation_difficulty(SKILL_ID, intents)


def test_blueprint_generation_difficulty_rejects_inconsistent_declared_difficulty():
    intents = [_intent(SKILL_ID, 1, "introductory"), _intent(SKILL_ID, 2, "advanced")]
    with pytest.raises(BatchGenerationError, match="consistent, explicit"):
        _blueprint_generation_difficulty(SKILL_ID, intents)


def test_blueprint_generation_difficulty_rejects_intent_from_a_different_skill():
    intents = [_intent(SKILL_ID, 1, "introductory"), _intent("AI-FND-01", 1, "introductory")]
    with pytest.raises(BatchGenerationError, match="belongs to skill"):
        _blueprint_generation_difficulty(SKILL_ID, intents)


def test_blueprint_generation_difficulty_rejects_empty_intent_pool():
    with pytest.raises(BatchGenerationError, match="no reviewed question intents"):
        _blueprint_generation_difficulty(SKILL_ID, [])


def test_generation_derives_introductory_difficulty_from_blueprint_end_to_end(
    manifest, repository, tmp_path, monkeypatch, approved_candidate
):
    """Full pipeline proof, not just the pure function: a blueprint that declares
    "introductory" must reach generate_batch as difficulty="introductory", not the
    "intermediate" BatchConfig would silently default to."""
    blueprint_dir = tmp_path / "blueprints"
    blueprint_dir.mkdir()
    intent = QuestionIntent(
        intent_id=f"{SKILL_ID}-INT-01",
        skill_id=SKILL_ID,
        assessment_focus="What a heuristic estimates",
        question_archetype="definition recall",
        preferred_reference_ids=[approved_candidate.candidate_id],
        required_concepts=["heuristic", "estimate"],
        prohibited_conflations=["heuristic equals exact cost"],
        difficulty="introductory",
    )
    blueprint = PilotBlueprint(
        batch_id="test-batch-introductory",
        prompt_version=question_intents.PILOT_PROMPT_VERSION,
        review_status="blueprint-approved",
        reviewer_id="albert",
        reviewed_at=FIXED_TIME,
        base_seed=1,
        intents=[intent],
    )
    (blueprint_dir / "test-batch-introductory.json").write_text(
        json.dumps(blueprint.model_dump(mode="json")), encoding="utf-8"
    )
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", blueprint_dir)
    monkeypatch.setattr(
        grounding_briefs,
        "PILOT_GROUNDING_BRIEFS",
        {
            SKILL_ID: CanonicalGroundingBrief(
                skill_id=SKILL_ID,
                version="test-v1",
                statements=["A heuristic estimates the remaining cost to the goal."],
            )
        },
    )
    introductory_response = json.dumps(
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
                    "difficulty": "introductory",
                }
            ]
        }
    )

    repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)
    retrieve_job = repository.claim_next(clock=fixed_clock)
    process_job(
        retrieve_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    generate_job = repository.claim_next(clock=fixed_clock)
    process_job(
        generate_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=lambda: DeterministicFakeModel([introductory_response]),
        clock=fixed_clock,
    )

    after = repository.get(generate_job.job_id)
    assert after.status == "queued"
    assert after.job_type == "automated_review"  # generation succeeded, not a failure

    output_dir = manifest.review_store_path.parent / "batches" / f"test-batch-introductory__{SKILL_ID}"
    manifest_data = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["difficulty"] == "introductory"
    generated = read_jsonl(output_dir / "pending_questions.jsonl", PendingQuestion)
    assert generated[0].question.difficulty == "introductory"


def test_generation_uses_the_blueprints_explicit_base_seed_when_present(
    manifest, repository, tmp_path, monkeypatch, approved_candidate
):
    """A blueprint's own base_seed (e.g. for reproducing a specific calibration run)
    always wins over the job_id-derived fallback -- proven end to end via the
    persisted batch manifest, not just the seed-selection expression in isolation."""
    blueprint_dir = tmp_path / "blueprints"
    blueprint_dir.mkdir()
    intent = QuestionIntent(
        intent_id=f"{SKILL_ID}-INT-01",
        skill_id=SKILL_ID,
        assessment_focus="What a heuristic estimates",
        question_archetype="definition recall",
        preferred_reference_ids=[approved_candidate.candidate_id],
        required_concepts=["heuristic", "estimate"],
        prohibited_conflations=["heuristic equals exact cost"],
        difficulty="intermediate",
    )
    blueprint = PilotBlueprint(
        batch_id="test-batch-explicit-seed",
        prompt_version=question_intents.PILOT_PROMPT_VERSION,
        review_status="blueprint-approved",
        reviewer_id="albert",
        reviewed_at=FIXED_TIME,
        base_seed=20260811,
        intents=[intent],
    )
    (blueprint_dir / "test-batch-explicit-seed.json").write_text(
        json.dumps(blueprint.model_dump(mode="json")), encoding="utf-8"
    )
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", blueprint_dir)
    monkeypatch.setattr(
        grounding_briefs,
        "PILOT_GROUNDING_BRIEFS",
        {
            SKILL_ID: CanonicalGroundingBrief(
                skill_id=SKILL_ID,
                version="test-v1",
                statements=["A heuristic estimates the remaining cost to the goal."],
            )
        },
    )

    repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)
    retrieve_job = repository.claim_next(clock=fixed_clock)
    process_job(
        retrieve_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    generate_job = repository.claim_next(clock=fixed_clock)
    process_job(
        generate_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )

    after = repository.get(generate_job.job_id)
    assert after.status == "queued"
    assert after.job_type == "automated_review"

    output_dir = manifest.review_store_path.parent / "batches" / f"test-batch-explicit-seed__{SKILL_ID}"
    manifest_data = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest_data["base_seed"] == 20260811


def test_generation_config_error_is_permanent_and_makes_no_model_calls(
    manifest, repository, tmp_path, monkeypatch, approved_candidate
):
    """An intent pool that disagrees on difficulty is a deterministic configuration
    defect: it must fail closed as permanent_failure (never retryable_failure, which
    would just re-fail identically max_attempts times) and must never construct or
    call a model to do so."""
    blueprint_dir = tmp_path / "blueprints"
    blueprint_dir.mkdir()
    intents = [
        QuestionIntent(
            intent_id=f"{SKILL_ID}-INT-01",
            skill_id=SKILL_ID,
            assessment_focus="What a heuristic estimates",
            question_archetype="definition recall",
            preferred_reference_ids=[approved_candidate.candidate_id],
            required_concepts=["heuristic"],
            prohibited_conflations=["heuristic equals exact cost"],
            difficulty="introductory",
        ),
        QuestionIntent(
            intent_id=f"{SKILL_ID}-INT-02",
            skill_id=SKILL_ID,
            assessment_focus="What a heuristic estimates, phrased differently",
            question_archetype="definition recall",
            preferred_reference_ids=[approved_candidate.candidate_id],
            required_concepts=["heuristic"],
            prohibited_conflations=["heuristic equals exact cost"],
            difficulty="advanced",
        ),
    ]
    blueprint = PilotBlueprint(
        batch_id="test-batch-inconsistent",
        prompt_version=question_intents.PILOT_PROMPT_VERSION,
        review_status="blueprint-approved",
        reviewer_id="albert",
        reviewed_at=FIXED_TIME,
        base_seed=1,
        intents=intents,
    )
    (blueprint_dir / "test-batch-inconsistent.json").write_text(
        json.dumps(blueprint.model_dump(mode="json")), encoding="utf-8"
    )
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", blueprint_dir)

    def _model_must_not_be_constructed():
        raise AssertionError("model_factory must not be called on a config error")

    repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=2, clock=fixed_clock)
    retrieve_job = repository.claim_next(clock=fixed_clock)
    process_job(
        retrieve_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    generate_job = repository.claim_next(clock=fixed_clock)
    process_job(
        generate_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=_model_must_not_be_constructed, clock=fixed_clock,
    )

    after = repository.get(generate_job.job_id)
    assert after.status == "permanent_failure"  # not retryable_failure
    assert after.error_code == "generation_config_error"


def _write_bank(manifest, item_ids: list[str], *, skill_id: str = SKILL_ID) -> None:
    """A minimal approved-bank JSONL with exactly these item_ids -- used to simulate
    "these slots are already fulfilled" without running promotion for real."""
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


def _write_blueprint(
    tmp_path, monkeypatch, *, batch_id: str, intents: list[QuestionIntent]
) -> PilotBlueprint:
    blueprint_dir = tmp_path / f"blueprints-{batch_id}"
    blueprint_dir.mkdir()
    blueprint = PilotBlueprint(
        batch_id=batch_id,
        prompt_version=question_intents.PILOT_PROMPT_VERSION,
        review_status="blueprint-approved",
        reviewer_id="albert",
        reviewed_at=FIXED_TIME,
        base_seed=1,
        intents=intents,
    )
    (blueprint_dir / f"{batch_id}.json").write_text(
        json.dumps(blueprint.model_dump(mode="json")), encoding="utf-8"
    )
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", blueprint_dir)
    monkeypatch.setattr(
        grounding_briefs,
        "PILOT_GROUNDING_BRIEFS",
        {
            SKILL_ID: CanonicalGroundingBrief(
                skill_id=SKILL_ID,
                version="test-v1",
                statements=["A heuristic estimates the remaining cost to the goal."],
            )
        },
    )
    return blueprint


def _write_blueprint_into(
    blueprint_dir: Path, *, batch_id: str, intents: list[QuestionIntent]
) -> PilotBlueprint:
    """Add one more blueprint file into an already-existing, already-monkeypatched
    BLUEPRINT_DIRECTORY -- used to make two (or more) blueprints genuinely coexist
    and both cover the same skill, unlike _write_blueprint's one-blueprint-per-fresh-
    directory shape."""
    blueprint = PilotBlueprint(
        batch_id=batch_id,
        prompt_version=question_intents.PILOT_PROMPT_VERSION,
        review_status="blueprint-approved",
        reviewer_id="albert",
        reviewed_at=FIXED_TIME,
        base_seed=1,
        intents=intents,
    )
    (blueprint_dir / f"{batch_id}.json").write_text(
        json.dumps(blueprint.model_dump(mode="json")), encoding="utf-8"
    )
    return blueprint


class _RaisingSearchProvider:
    """Proves zero network calls: fails the test outright if retrieval ever calls it."""

    def search(self, schedule, diagnostics, budget):
        raise AssertionError("search_provider must not be called when demand is already satisfied")
        yield  # pragma: no cover -- generator shape only


def _fake_job(*, skill_id: str = SKILL_ID, metadata: dict | None = None) -> ReplenishmentJob:
    """A standalone job row for exercising resolve_job_blueprint as a pure function,
    without the ceremony of a real job repository."""
    return ReplenishmentJob(
        job_id="test-job-01",
        course_id="ai",
        skill_id=skill_id,
        job_type="generate_questions",
        status="running",
        requested_count=1,
        attempts=1,
        created_at=FIXED_TIME,
        metadata=metadata or {},
    )


# --- Explicit batch_id resolution: coexistence, mtime independence, missing/invalid ids ---
#
# Real motivating case: authoring/blueprints/grounded-calibration-v1a-introductory.json and
# grounded-calibration-v1b-intermediate.json both cover AI-SRC-01 and AI-SRC-08 alongside the
# original grounded-pilot-20260811-v3.json blueprint. The old find_blueprint_for_skill picked
# "whichever blueprint file was modified most recently" -- nondeterministic across git
# checkouts/edits, and silently wrong the moment two blueprints legitimately coexist for the
# same skill. resolve_job_blueprint replaces it: a job's batch_id, once known, is resolved by
# id alone (never scanned by mtime); with no batch_id, resolution only succeeds when exactly
# one blueprint covers the skill, and fails closed (AmbiguousBlueprintError) otherwise.


def test_blueprints_covering_skill_finds_every_coexisting_blueprint_independent_of_mtime(
    tmp_path, monkeypatch
):
    blueprint_dir = tmp_path / "blueprints"
    blueprint_dir.mkdir()
    _write_blueprint_into(
        blueprint_dir, batch_id="calib-a", intents=[_intent(SKILL_ID, 1, "introductory")]
    )
    _write_blueprint_into(
        blueprint_dir, batch_id="calib-b", intents=[_intent(SKILL_ID, 1, "intermediate")]
    )
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", blueprint_dir)

    before = {bp.batch_id for bp in blueprints_covering_skill(SKILL_ID)}
    assert before == {"calib-a", "calib-b"}

    # calib-b was written after calib-a, so it currently has the newer mtime -- the
    # mtime-based bug this replaced would have silently picked calib-b alone. Flip
    # which file is newest and confirm the result is byte-for-byte identical.
    now = datetime.now().timestamp()
    os.utime(blueprint_dir / "calib-a.json", (now + 100, now + 100))
    os.utime(blueprint_dir / "calib-b.json", (now, now))
    after = {bp.batch_id for bp in blueprints_covering_skill(SKILL_ID)}
    assert after == before


def test_resolve_job_blueprint_fails_closed_when_coexisting_blueprints_have_no_explicit_batch_id(
    tmp_path, monkeypatch
):
    blueprint_dir = tmp_path / "blueprints"
    blueprint_dir.mkdir()
    _write_blueprint_into(
        blueprint_dir, batch_id="calib-a", intents=[_intent(SKILL_ID, 1, "introductory")]
    )
    _write_blueprint_into(
        blueprint_dir, batch_id="calib-b", intents=[_intent(SKILL_ID, 1, "intermediate")]
    )
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", blueprint_dir)

    with pytest.raises(AmbiguousBlueprintError, match="2 blueprints cover"):
        resolve_job_blueprint(_fake_job())


def test_resolve_job_blueprint_uses_explicit_batch_id_to_disambiguate_coexisting_blueprints(
    tmp_path, monkeypatch
):
    blueprint_dir = tmp_path / "blueprints"
    blueprint_dir.mkdir()
    _write_blueprint_into(
        blueprint_dir, batch_id="calib-a", intents=[_intent(SKILL_ID, 1, "introductory")]
    )
    _write_blueprint_into(
        blueprint_dir, batch_id="calib-b", intents=[_intent(SKILL_ID, 1, "intermediate")]
    )
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", blueprint_dir)

    resolved, batch_id = resolve_job_blueprint(_fake_job(metadata={"batch_id": "calib-b"}))
    assert batch_id == "calib-b"
    assert resolved.batch_id == "calib-b"
    assert resolved.intents[0].difficulty == "intermediate"


def test_resolve_job_blueprint_returns_none_when_no_blueprint_covers_the_skill(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", tmp_path / "empty-blueprints")
    blueprint, batch_id = resolve_job_blueprint(_fake_job())
    assert blueprint is None
    assert batch_id is None


def test_resolve_job_blueprint_rejects_unknown_batch_id(tmp_path, monkeypatch):
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", tmp_path / "empty-blueprints")
    with pytest.raises(ValueError, match="no question-intent blueprint"):
        resolve_job_blueprint(_fake_job(metadata={"batch_id": "does-not-exist"}))


def test_resolve_job_blueprint_rejects_batch_id_that_does_not_cover_this_skill(
    tmp_path, monkeypatch
):
    blueprint = _write_blueprint(
        tmp_path, monkeypatch,
        batch_id="other-skill-batch",
        intents=[_intent("AI-FND-01", 1, "introductory")],
    )
    with pytest.raises(ValueError, match="does not cover skill"):
        resolve_job_blueprint(_fake_job(metadata={"batch_id": blueprint.batch_id}))


def test_ambiguous_blueprint_selection_fails_closed_before_any_network_call(
    manifest, repository, tmp_path, monkeypatch
):
    blueprint_dir = tmp_path / "blueprints"
    blueprint_dir.mkdir()
    _write_blueprint_into(
        blueprint_dir, batch_id="calib-a", intents=[_intent(SKILL_ID, 1, "introductory")]
    )
    _write_blueprint_into(
        blueprint_dir, batch_id="calib-b", intents=[_intent(SKILL_ID, 1, "intermediate")]
    )
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", blueprint_dir)

    repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)
    job = repository.claim_next(clock=fixed_clock)
    process_job(
        job, manifest, job_repository=repository,
        search_provider=_RaisingSearchProvider(), fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    after = repository.get(job.job_id)
    assert after.status == "permanent_failure"
    assert after.error_code == "ambiguous_blueprint_selection"


def test_ambiguous_blueprint_selection_is_independent_of_blueprint_file_modification_order(
    manifest, repository, tmp_path, monkeypatch
):
    """Same setup as the test above, but calib-a is written (and thus modified) last
    -- newer than calib-b. The old mtime-based find_blueprint_for_skill would have
    silently resolved calib-a here and calib-b in the test above; the fix must reach
    the identical ambiguous failure regardless of write order."""
    blueprint_dir = tmp_path / "blueprints"
    blueprint_dir.mkdir()
    _write_blueprint_into(
        blueprint_dir, batch_id="calib-b", intents=[_intent(SKILL_ID, 1, "intermediate")]
    )
    _write_blueprint_into(
        blueprint_dir, batch_id="calib-a", intents=[_intent(SKILL_ID, 1, "introductory")]
    )
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", blueprint_dir)

    repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)
    job = repository.claim_next(clock=fixed_clock)
    process_job(
        job, manifest, job_repository=repository,
        search_provider=_RaisingSearchProvider(), fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    after = repository.get(job.job_id)
    assert after.status == "permanent_failure"
    assert after.error_code == "ambiguous_blueprint_selection"
    assert "calib-a" in after.error_message and "calib-b" in after.error_message


def test_invalid_explicit_batch_id_is_a_permanent_failure_before_any_network_call(
    manifest, repository, tmp_path, monkeypatch
):
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", tmp_path / "empty-blueprints")
    repository.enqueue(
        course_id="ai", skill_id=SKILL_ID, requested_count=1,
        metadata={"batch_id": "does-not-exist"}, clock=fixed_clock,
    )
    job = repository.claim_next(clock=fixed_clock)
    process_job(
        job, manifest, job_repository=repository,
        search_provider=_RaisingSearchProvider(), fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    after = repository.get(job.job_id)
    assert after.status == "permanent_failure"
    assert after.error_code == "invalid_batch_id"


def test_explicit_batch_id_selects_the_correct_blueprint_when_two_coexist(
    manifest, repository, tmp_path, monkeypatch, approved_candidate
):
    """Two blueprints coexist and both cover SKILL_ID (mirrors the real
    calibration-v1a/v1b split, which both cover AI-SRC-01 and AI-SRC-08) -- a job
    enqueued with an explicit batch_id must generate against exactly that
    blueprint's intents, never the other coexisting one, and must persist the same
    batch_id into its own job metadata so later stages never re-resolve it."""
    blueprint_dir = tmp_path / "blueprints"
    blueprint_dir.mkdir()
    _write_blueprint_into(
        blueprint_dir, batch_id="calib-a",
        intents=[
            _intent(SKILL_ID, 1, "introductory", preferred_reference_ids=[approved_candidate.candidate_id])
        ],
    )
    _write_blueprint_into(
        blueprint_dir, batch_id="calib-b",
        intents=[
            _intent(SKILL_ID, 1, "intermediate", preferred_reference_ids=[approved_candidate.candidate_id])
        ],
    )
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", blueprint_dir)
    monkeypatch.setattr(
        grounding_briefs,
        "PILOT_GROUNDING_BRIEFS",
        {
            SKILL_ID: CanonicalGroundingBrief(
                skill_id=SKILL_ID,
                version="test-v1",
                statements=["A heuristic estimates the remaining cost to the goal."],
            )
        },
    )

    repository.enqueue(
        course_id="ai", skill_id=SKILL_ID, requested_count=1,
        metadata={"batch_id": "calib-b"}, clock=fixed_clock,
    )
    retrieve_job = repository.claim_next(clock=fixed_clock)
    process_job(
        retrieve_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    after_retrieve = repository.get(retrieve_job.job_id)
    assert after_retrieve.status == "queued"
    assert after_retrieve.job_type == "generate_questions"
    assert after_retrieve.metadata["batch_id"] == "calib-b"  # pinned, not re-derived

    generate_job = repository.claim_next(clock=fixed_clock)
    process_job(
        generate_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        # DeterministicFakeModel's default canned response already declares
        # difficulty="intermediate", matching calib-b's intent exactly.
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    after_generate = repository.get(generate_job.job_id)
    assert after_generate.status == "queued"
    assert after_generate.job_type == "automated_review"
    assert after_generate.metadata["batch_id"] == "calib-b"

    output_dir = manifest.review_store_path.parent / "batches" / f"calib-b__{SKILL_ID}"
    assert output_dir.is_dir()
    other_output_dir = manifest.review_store_path.parent / "batches" / f"calib-a__{SKILL_ID}"
    assert not other_output_dir.exists()  # the other coexisting blueprint was never touched


def test_fully_satisfied_blueprint_stops_before_any_retrieval_call(
    manifest, repository, tmp_path, monkeypatch
):
    """The exact shape of the AI-SRC-08 incident: a blueprint whose only intent slot
    is already an approved bank item must never even attempt a Brave Search call."""
    blueprint = _write_blueprint(
        tmp_path, monkeypatch,
        batch_id="test-batch-satisfied",
        intents=[_intent(SKILL_ID, 1, "introductory")],
    )
    _write_bank(manifest, [question_id(blueprint.batch_id, SKILL_ID, 0)])

    repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)
    job = repository.claim_next(clock=fixed_clock)
    process_job(
        job, manifest, job_repository=repository,
        search_provider=_RaisingSearchProvider(), fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )

    after = repository.get(job.job_id)
    assert after.status == "permanent_failure"
    assert after.error_code == "demand_already_satisfied"
    assert after.job_type == "replenish_skill"  # never advanced to generate_questions


def test_partially_deficient_blueprint_generates_only_the_non_contiguous_deficit(
    manifest, repository, tmp_path, monkeypatch, approved_candidate
):
    """3 intents, only the middle slot (index 0) already satisfied -- generation must
    skip index 0 and produce exactly indices 1 and 2, proving the fix works even when
    the deficient slots are not a leading contiguous run (generate_batch always starts
    iterating at index 0, so a naive count-based cap alone would get this wrong)."""
    intents = [
        _intent(SKILL_ID, n, "introductory", preferred_reference_ids=[approved_candidate.candidate_id])
        for n in (1, 2, 3)
    ]
    blueprint = _write_blueprint(
        tmp_path, monkeypatch, batch_id="test-batch-partial", intents=intents
    )
    _write_bank(manifest, [question_id(blueprint.batch_id, SKILL_ID, 0)])

    repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)
    retrieve_job = repository.claim_next(clock=fixed_clock)
    process_job(
        retrieve_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    generate_job = repository.claim_next(clock=fixed_clock)
    process_job(
        generate_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=lambda: DeterministicFakeModel([
            _introductory_response("What does the heuristic estimate for slot two?"),
            _introductory_response("What does the heuristic estimate for slot three?"),
        ]),
        clock=fixed_clock,
    )

    after = repository.get(generate_job.job_id)
    assert after.status == "queued"
    assert after.job_type == "automated_review"  # generation succeeded, not a config failure

    output_dir = manifest.review_store_path.parent / "batches" / f"test-batch-partial__{SKILL_ID}"
    generated = read_jsonl(output_dir / "pending_questions.jsonl", PendingQuestion)
    assert {item.question_index for item in generated} == {1, 2}
    assert {item.intent_id for item in generated} == {f"{SKILL_ID}-INT-02", f"{SKILL_ID}-INT-03"}


def test_approved_item_added_between_retrieval_and_generation_stops_safely(
    manifest, repository, tmp_path, monkeypatch, approved_candidate
):
    """Demand is rechecked fresh at the top of generate_questions, not only once at
    enqueue/retrieval time -- if another job or a human fills the only deficient slot
    in between, this job must stop with zero model calls, not regenerate a duplicate."""
    blueprint = _write_blueprint(
        tmp_path, monkeypatch,
        batch_id="test-batch-race",
        intents=[
            _intent(SKILL_ID, 1, "introductory", preferred_reference_ids=[approved_candidate.candidate_id])
        ],
    )
    # No bank file yet at enqueue time -- genuinely deficient.
    repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)
    retrieve_job = repository.claim_next(clock=fixed_clock)
    process_job(
        retrieve_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    after_retrieve = repository.get(retrieve_job.job_id)
    assert after_retrieve.job_type == "generate_questions"  # deficient, proceeded normally

    # Someone else fills the only slot before this job's generate_questions tick runs.
    _write_bank(manifest, [question_id(blueprint.batch_id, SKILL_ID, 0)])

    def _model_must_not_be_constructed():
        raise AssertionError("model_factory must not be called once demand is satisfied")

    generate_job = repository.claim_next(clock=fixed_clock)
    process_job(
        generate_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=_model_must_not_be_constructed, clock=fixed_clock,
    )

    after = repository.get(generate_job.job_id)
    assert after.status == "permanent_failure"
    assert after.error_code == "demand_already_satisfied"


def test_rerun_after_the_only_slot_is_filled_makes_zero_network_calls(
    manifest, repository, tmp_path, monkeypatch, approved_candidate
):
    """A second, independent job for the same (course, skill) -- e.g. from a rerun or
    a fresh scan after this skill was already fully replenished -- must not recreate
    demand another job already filled."""
    blueprint = _write_blueprint(
        tmp_path, monkeypatch,
        batch_id="test-batch-rerun",
        intents=[_intent(SKILL_ID, 1, "introductory")],
    )
    _write_bank(manifest, [question_id(blueprint.batch_id, SKILL_ID, 0)])

    # A prior job already completed (status irrelevant to _active_job's ACTIVE_STATUSES
    # scoping -- this just needs to not collide with the dedup index).
    repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)
    first = repository.claim_next(clock=fixed_clock)
    repository.mark_completed(first.job_id, clock=fixed_clock)

    # A fresh rerun enqueues a brand new job, same as cli.py scan() would.
    repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)
    second = repository.claim_next(clock=fixed_clock)
    process_job(
        second, manifest, job_repository=repository,
        search_provider=_RaisingSearchProvider(), fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )

    after = repository.get(second.job_id)
    assert after.status == "permanent_failure"
    assert after.error_code == "demand_already_satisfied"


def _semantic_reject_review_result() -> SemanticReviewResult:
    """A genuine semantic-reviewer critical rejection (disagrees with the declared
    answer) -- distinct from a deterministic-only rejection, for proving
    rejected_by_automated_review vs. rejected_deterministically route differently."""
    clean = _clean_review_result("Remaining cost to goal")
    return clean.model_copy(
        update={
            "answer_assessment": clean.answer_assessment.model_copy(
                update={"matches_declared_answer": False, "selected_option_text": "Total memory used"}
            )
        }
    )


def test_deterministic_only_reject_never_calls_the_semantic_reviewer(
    manifest, repository, reviewed_blueprint, approved_candidate
):
    """Reproduces the live AI-SRC-08 incident directly: a candidate whose deterministic
    question_id already exists in the approved bank must be rejected by the
    no_duplicate_item_id check alone, landing on rejected_deterministically -- never
    rejected_by_automated_review, and never calling the semantic reviewer at all."""
    job = _advance_to_automated_review(manifest, repository)
    review_job = repository.claim_next(clock=fixed_clock)
    review_path = Path(review_job.metadata["review_path"])
    review = GroundedReviewStore(review_path).load()
    collided_id = review.items[0].original_question_id
    _write_bank(manifest, [collided_id])

    reviewer = FakeContentReviewer({})  # must never be consulted
    process_job(
        review_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel,
        reviewer_factory=lambda: reviewer,
        review_config=ReviewPolicyConfig(shadow_mode=True),
        clock=fixed_clock,
    )

    after = repository.get(job.job_id)
    assert after.status == "rejected_deterministically"
    assert reviewer.calls == []


def test_semantic_reject_uses_rejected_by_automated_review(
    manifest, repository, reviewed_blueprint, approved_candidate
):
    job = _advance_to_automated_review(manifest, repository)
    review_job = repository.claim_next(clock=fixed_clock)
    reviewer = FakeContentReviewer(
        {"What does a heuristic estimate?": _semantic_reject_review_result()}
    )
    process_job(
        review_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel,
        reviewer_factory=lambda: reviewer,
        review_config=ReviewPolicyConfig(shadow_mode=True),
        clock=fixed_clock,
    )

    after = repository.get(job.job_id)
    assert after.status == "rejected_by_automated_review"
    assert reviewer.calls == ["What does a heuristic estimate?"]


def test_a_completed_batch_stays_immutable_even_when_more_demand_appears_later(
    manifest, repository, tmp_path, monkeypatch, approved_candidate
):
    """generate_batch refuses to resume a batch whose manifest already says "complete"
    (authoring/grounded_batch.py's own "a completed generated batch is immutable"
    guard, proven separately in test_grounded_review.py). skip_question_indices does
    not, and must not, create a way around that: once this blueprint's batch has
    completed for whatever set of slots was deficient at the time, a slot that becomes
    deficient afterward (e.g. an earlier approval gets reverted) can never be filled by
    resuming the same batch_id/output_dir -- it needs a genuinely new generation round,
    same as it would without this feature."""
    intents = [
        _intent(SKILL_ID, n, "introductory", preferred_reference_ids=[approved_candidate.candidate_id])
        for n in (1, 2)
    ]
    blueprint = _write_blueprint(
        tmp_path, monkeypatch, batch_id="test-batch-immutable", intents=intents
    )
    # Slot 1 (index 1) starts satisfied -- only slot 0 is deficient on the first tick.
    _write_bank(manifest, [question_id(blueprint.batch_id, SKILL_ID, 1)])

    repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)
    retrieve_job = repository.claim_next(clock=fixed_clock)
    process_job(
        retrieve_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    generate_job = repository.claim_next(clock=fixed_clock)
    process_job(
        generate_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=lambda: DeterministicFakeModel(
            [_introductory_response("What does the heuristic estimate for slot one?")]
        ),
        clock=fixed_clock,
    )
    after_first_tick = repository.get(generate_job.job_id)
    assert after_first_tick.job_type == "automated_review"  # batch reached "complete"

    # Slot 1's approval is reverted -- it is now deficient too, but the batch this
    # blueprint's output_dir belongs to already completed.
    active_bank_path(manifest).unlink()
    repository.mark_queued(generate_job.job_id, job_type="generate_questions")
    second_generate_job = repository.claim_next(clock=fixed_clock)
    spy_model = DeterministicFakeModel()
    process_job(
        second_generate_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=lambda: spy_model, clock=fixed_clock,
    )

    after = repository.get(generate_job.job_id)
    assert after.status == "permanent_failure"
    assert after.error_code == "generation_config_error"
    assert "immutable" in after.error_message
    assert spy_model.calls == []  # generate_batch raises before ever calling generate()


def test_model_unavailable_moves_to_waiting_for_model_and_never_fails(
    manifest, repository, reviewed_blueprint, approved_candidate
):
    repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)
    retrieve_job = repository.claim_next(clock=fixed_clock)
    process_job(
        retrieve_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    generate_job = repository.claim_next(clock=fixed_clock)
    process_job(
        generate_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=ModelDownFakeModel, clock=fixed_clock,
    )
    after = repository.get(generate_job.job_id)
    assert after.status == "waiting_for_model"
    assert after.job_type == "generate_questions"
    assert after.error_code is None  # waiting, not failed
    assert ready_to_resume(after, manifest) is True  # always worth another try


def test_pending_questions_are_excluded_from_promotion(manifest, repository, reviewed_blueprint, approved_candidate):
    repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)
    retrieve_job = repository.claim_next(clock=fixed_clock)
    process_job(
        retrieve_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    generate_job = repository.claim_next(clock=fixed_clock)
    process_job(
        generate_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    after_generate = repository.get(generate_job.job_id)
    review_path = Path(after_generate.metadata["review_path"])

    # No approval happens: the question stays pending.
    assert ready_to_resume(after_generate, manifest) is False
    repository.mark_waiting(
        generate_job.job_id, "waiting_for_question_review",
        job_type="generate_questions", metadata=after_generate.metadata,
    )
    repository.mark_queued(generate_job.job_id, job_type="promote_approved_items")
    promote_job = repository.claim_next(clock=fixed_clock)
    process_job(
        promote_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    after_promote = repository.get(promote_job.job_id)
    assert after_promote.status == "permanent_failure"
    assert after_promote.error_code == "no_approved_items"
    assert not manifest.approved_bank_path.is_file()


def test_process_one_drives_retrieval_through_the_real_search_path(
    manifest, repository, tmp_path, monkeypatch
):
    # Isolated from the real authoring/blueprints/ directory (unrelated to this
    # test), which now legitimately carries more than one blueprint covering
    # SKILL_ID.
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", tmp_path / "no-blueprints")
    url = "https://example.edu/pathfinding/heuristics.html"
    provider = FakeSearchProvider([SearchResult(title="Heuristics", url=url, snippet="")])
    fetcher = FakePageFetcher({url: PASSAGE})
    repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)

    job = process_one(
        repository, {"ai": manifest},
        search_provider_factory=lambda m: provider,
        fetcher_factory=lambda m: fetcher,
        clock=fixed_clock,
    )
    after = repository.get(job.job_id)
    assert after.status == "waiting_for_reference_review"
    candidates = CandidateStore(manifest.candidate_store_path).load()
    assert len(candidates) == 1
    assert candidates[0].skill_id == SKILL_ID


def _flagged_review_result(*, unsupported_claims: list[str]) -> SemanticReviewResult:
    flagged = _clean_review_result("Remaining cost to goal")
    return flagged.model_copy(
        update={
            "grounding_assessment": flagged.grounding_assessment.model_copy(
                update={"unsupported_claims": unsupported_claims}
            )
        }
    )


def _advance_to_automated_review(manifest, repository):
    """Drive a fresh job through retrieve_references and generate_questions, landing
    it queued at job_type="automated_review", ready for the caller's own reviewer."""
    repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)
    retrieve_job = repository.claim_next(clock=fixed_clock)
    process_job(
        retrieve_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    generate_job = repository.claim_next(clock=fixed_clock)
    process_job(
        generate_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    return generate_job


def test_reviewer_outage_at_automated_review_moves_to_waiting_for_model(
    manifest, repository, reviewed_blueprint, approved_candidate
):
    job = _advance_to_automated_review(manifest, repository)
    review_job = repository.claim_next(clock=fixed_clock)
    process_job(
        review_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel,
        reviewer_factory=lambda: FakeContentReviewer(
            {"What does a heuristic estimate?": ReviewerUnavailableError("endpoint down")}
        ),
        clock=fixed_clock,
    )
    after = repository.get(job.job_id)
    assert after.status == "waiting_for_model"
    assert after.job_type == "automated_review"
    assert after.error_code is None  # waiting, not failed
    assert ready_to_resume(after, manifest) is True


def test_equivalence_gate_initialization_failure_escalates_and_worker_survives_to_later_jobs(
    manifest, repository, reviewed_blueprint, approved_candidate, monkeypatch
):
    """An unavailable/broken NLI model must never crash the worker process -- the
    affected job escalates to waiting_for_full_human_review (not a raised exception,
    not a permanent failure), and the worker process is still able to claim and
    process a later, independent job afterward -- proving no uncaught exception
    propagated out of process_job()/review_candidate() (see
    authoring/review/equivalence_gate.py's bounded warm-up)."""

    class UnavailableScorer:
        def warm_up(self):
            raise OSError("simulated: nli model unreachable")

        def score(self, premise, hypothesis):
            raise AssertionError("must never be called -- warm-up already failed")

    monkeypatch.setattr(
        "authoring.review.equivalence_gate.get_default_scorer", lambda: UnavailableScorer()
    )

    job = _advance_to_automated_review(manifest, repository)
    review_job = repository.claim_next(clock=fixed_clock)
    process_job(
        review_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel,
        reviewer_factory=clean_reviewer_factory,
        clock=fixed_clock,
    )
    after = repository.get(job.job_id)
    assert after.status == "waiting_for_full_human_review"
    assert after.job_type == "automated_review"
    assert after.error_code is None  # escalated, not failed -- see worker.py's
    # any_require_full_human_review branch

    # The worker process itself survived: cli.py's poll loop (`while True:
    # run_once() ...`) calls exactly this claim_next()/process_job() sequence every
    # tick, with no surrounding try/except of its own -- before this fix, an
    # uncaught OSError from NLI warm-up would have propagated out of process_job()
    # here and killed that loop outright. Calling it again, cleanly, is the direct
    # regression check for that failure mode: this skill's only job is now legitimately
    # active (waiting_for_full_human_review, per ACTIVE_STATUSES) so nothing new is
    # claimable, but the call completing without raising proves the dispatch loop --
    # and thus the worker process -- is still alive and able to serve the next tick,
    # whatever job that turns out to be.
    assert repository.claim_next(clock=fixed_clock) is None


def test_model_outage_during_automated_revision_moves_to_waiting_for_model(
    manifest, repository, reviewed_blueprint, approved_candidate
):
    job = _advance_to_automated_review(manifest, repository)
    review_job = repository.claim_next(clock=fixed_clock)
    process_job(
        review_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel,
        reviewer_factory=lambda: FakeContentReviewer(
            {
                "What does a heuristic estimate?": _flagged_review_result(
                    unsupported_claims=["explanation asserts a fact no reference states"]
                )
            }
        ),
        clock=fixed_clock,
    )
    after_review = repository.get(job.job_id)
    assert after_review.status == "queued"
    assert after_review.job_type == "automated_revision"

    revision_job = repository.claim_next(clock=fixed_clock)
    process_job(
        revision_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=ModelDownFakeModel, clock=fixed_clock,
    )
    after_revision = repository.get(job.job_id)
    assert after_revision.status == "waiting_for_model"
    assert after_revision.job_type == "automated_revision"
    assert after_revision.error_code is None


def test_shadow_mode_never_triggers_automated_revision(
    manifest, repository, reviewed_blueprint, approved_candidate
):
    """A recommendation that would normally queue automated_revision (propose_revision,
    from an unsupported-claims finding) must instead land straight on
    waiting_for_question_review when shadow_mode is on, with no model-authored rewrite
    ever attempted -- see [[automated_review_layer]]'s shadow-mode definition."""
    job = _advance_to_automated_review(manifest, repository)
    review_job = repository.claim_next(clock=fixed_clock)
    process_job(
        review_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel,
        reviewer_factory=lambda: FakeContentReviewer(
            {
                "What does a heuristic estimate?": _flagged_review_result(
                    unsupported_claims=["explanation asserts a fact no reference states"]
                )
            }
        ),
        review_config=ReviewPolicyConfig(shadow_mode=True),
        clock=fixed_clock,
    )
    after_review = repository.get(job.job_id)
    assert after_review.status == "waiting_for_question_review"
    assert after_review.job_type == "automated_review"

    review = GroundedReviewStore(Path(after_review.metadata["review_path"])).load()
    item = review.items[0]
    assert item.recommendation == "propose_revision"
    assert item.revisions == []  # no automated rewrite was ever attempted
    assert item.final_review_status == "pending"


def test_bounded_automatic_revision_settles_for_human_review_without_looping_forever(
    manifest, repository, reviewed_blueprint, approved_candidate
):
    """DeterministicFakeModel always regenerates the identical question, so every
    automated revision attempt is a no-op edit that propose_automated_revision
    rejects -- this must still terminate within max_automatic_revisions attempts,
    never auto-approve anything, and never spin between the two stages forever."""
    job = _advance_to_automated_review(manifest, repository)
    reviewer_factory = lambda: FakeContentReviewer(
        {
            "What does a heuristic estimate?": _flagged_review_result(
                unsupported_claims=["explanation asserts a fact no reference states"]
            )
        }
    )
    review_config = ReviewPolicyConfig(max_automatic_revisions=1)

    for _ in range(6):  # generous bound: real settlement takes 3 ticks
        current = repository.get(job.job_id)
        if current.status != "queued":
            break
        claimed = repository.claim_next(clock=fixed_clock)
        process_job(
            claimed, manifest, job_repository=repository,
            search_provider=None, fetcher=None,
            model_factory=DeterministicFakeModel,
            reviewer_factory=reviewer_factory,
            review_config=review_config,
            clock=fixed_clock,
        )

    after = repository.get(job.job_id)
    assert after.status == "waiting_for_question_review"
    assert after.metadata["automated_revision_attempts"] == {
        "AI-SRC-08-8fc980b05192e581": 1
    }
    review = GroundedReviewStore(Path(after.metadata["review_path"])).load()
    item = review.items[0]
    assert item.final_review_status == "pending"
    assert item.revisions == []  # every attempted revision was a no-op, none recorded


def test_human_decision_mid_batch_is_never_overwritten_by_automated_revision(
    manifest, repository, reviewed_blueprint, approved_candidate
):
    job = _advance_to_automated_review(manifest, repository)
    review_job = repository.claim_next(clock=fixed_clock)
    process_job(
        review_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel,
        reviewer_factory=lambda: FakeContentReviewer(
            {
                "What does a heuristic estimate?": _flagged_review_result(
                    unsupported_claims=["explanation asserts a fact no reference states"]
                )
            }
        ),
        clock=fixed_clock,
    )
    after_review = repository.get(job.job_id)
    assert after_review.job_type == "automated_revision"
    review_path = Path(after_review.metadata["review_path"])

    # A human rejects the item directly, out from under the worker, before the
    # automated_revision stage ever runs.
    review = GroundedReviewStore(review_path).load()
    rejected = reject_item(review.items[0], "albert", "not a good question", reviewed_at=FIXED_TIME)
    GroundedReviewStore(review_path).replace_item(rejected)

    revision_job = repository.claim_next(clock=fixed_clock)
    process_job(
        revision_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )

    final_review = GroundedReviewStore(review_path).load()
    item = final_review.items[0]
    assert item.final_review_status == "rejected"
    assert item.reviewed_by == "albert"
    assert item.revisions == []  # automated revision never touched a human-decided item


def _minimal_automated_report(content_hash: str, intent_id: str) -> AutomatedReviewReport:
    """A report with just enough substance to satisfy the promotion gate's
    latest_for_hash() lookup -- the gate only checks a report exists for the
    reviewed content, not its verdict."""
    return AutomatedReviewReport(
        review_id=f"report-{content_hash[:12]}",
        candidate_id="candidate",
        skill_id=SKILL_ID,
        intent_id=intent_id,
        review_policy_version="review-v1",
        reviewer_model_id="fake-reviewer",
        reviewer_model_revision="fake-rev-1",
        reviewer_prompt_version="review-v1",
        reviewer_prompt_template_hash="d" * 64,
        rendered_review_request_hash="d" * 64,
        reviewed_content_hash=content_hash,
        created_at=FIXED_TIME,
        deterministic_checks=DeterministicChecks(checks=[]),
        risk_score=0.1,
        risk_level="low",
        recommendation="recommend_human_approval",
    )


def test_promotion_exports_only_the_two_approved_items_from_a_mixed_batch(
    manifest, repository, approved_candidate, tmp_path, monkeypatch
):
    """End-to-end promotion: a review store carrying one approve-as-written item, one
    approved revision, one rejected item, and one still-pending item -- promotion
    must succeed and the active bank must contain exactly the two approved items,
    exercising the real _handle_promote_approved_items -> export_approved_bank_items
    path (the promotion-blocking StopIteration this fixes)."""
    blueprint_dir = tmp_path / "blueprints"
    blueprint_dir.mkdir()
    intents = [
        _intent(SKILL_ID, n, "intermediate", preferred_reference_ids=[approved_candidate.candidate_id])
        for n in range(1, 5)
    ]
    blueprint = PilotBlueprint(
        batch_id="mixed-batch-01",
        prompt_version=question_intents.PILOT_PROMPT_VERSION,
        review_status="blueprint-approved",
        reviewer_id="albert",
        reviewed_at=FIXED_TIME,
        base_seed=1,
        intents=intents,
    )
    (blueprint_dir / "mixed-batch-01.json").write_text(
        json.dumps(blueprint.model_dump(mode="json")), encoding="utf-8"
    )
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", blueprint_dir)
    monkeypatch.setattr(
        grounding_briefs,
        "PILOT_GROUNDING_BRIEFS",
        {
            SKILL_ID: CanonicalGroundingBrief(
                skill_id=SKILL_ID,
                version="test-v1",
                statements=["A heuristic estimates the remaining cost to the goal."],
            )
        },
    )

    model = DeterministicFakeModel(
        responses=[
            _intermediate_response("Which as-written question stands unedited?"),
            _intermediate_response("Which revised question needs an edit?"),
            _intermediate_response("Which rejected question fails review?"),
            _intermediate_response("Which pending question awaits a decision?"),
        ]
    )
    import_candidates(
        manifest.candidate_store_path, manifest.skills_path(), manifest.references_path(), manifest.provenance_path()
    )
    output_dir = manifest.review_store_path.parent / "batches" / "mixed-batch-01__AI-SRC-08"
    result = generate_batch(
        BatchConfig(
            batch_id="mixed-batch-01",
            skill_ids=[SKILL_ID],
            questions_per_skill=len(intents),
            base_seed=1,
            model_id=model.model_id,
            prompt_version=question_intents.PILOT_PROMPT_VERSION,
            difficulty="intermediate",
        ),
        model,
        output_dir,
        skills_path=manifest.skills_path(),
        references_path=manifest.references_path(),
        provenance_path=manifest.provenance_path(),
        clock=fixed_clock,
        git_commit="deadbeef",
    )
    assert result.status == "complete"

    review = build_pending_review(output_dir)
    source_by_id = {q.question_id: q for q in load_source_questions(output_dir)}
    as_written_item, revised_item, rejected_item, pending_item = review.items

    # 1) approve as written -- the promotion gate requires an automated review
    # report on file for its exact source content hash.
    report_store = AutomatedReviewReportStore(
        review_report_path(manifest.review_store_path, "mixed-batch-01", SKILL_ID)
    )
    report_store.append(
        _minimal_automated_report(
            question_content_hash(source_by_id[as_written_item.original_question_id].question),
            as_written_item.intent_id,
        )
    )
    as_written_item = as_written_item.model_copy(update={"recommendation": "approve_as_written"})
    as_written_item = approve_as_written(as_written_item, "albert", reviewed_at=FIXED_TIME)

    # 2) approve a proposed revision.
    revised_source = source_by_id[revised_item.original_question_id]
    revised_edit = QuizQuestion.model_validate(
        revised_source.question.model_dump() | {"explanation": revised_source.question.explanation + " (edited)"}
    )
    proposed = propose_revision(
        revised_item,
        revised_source.question,
        revised_edit,
        "albert",
        "reviewer restated wording",
        edited_at=FIXED_TIME,
        provenance=RevisionProvenance.from_source(revised_source),
    )
    revised_item = approve_revision(proposed, proposed.revisions[0].revision_id, "albert", reviewed_at=FIXED_TIME)

    # 3) explicitly reject.
    rejected_item = reject_item(rejected_item, "albert", "Not accurate.", reviewed_at=FIXED_TIME)

    # 4) pending_item is left untouched.

    review_path = manifest.review_store_path / "mixed-batch-01__AI-SRC-08.json"
    store = GroundedReviewStore(review_path)
    store.save(review)
    store.replace_item(as_written_item)
    store.replace_item(revised_item)
    store.replace_item(rejected_item)

    repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=1, clock=fixed_clock)
    job = repository.claim_next(clock=fixed_clock)
    repository.mark_queued(
        job.job_id,
        job_type="promote_approved_items",
        metadata={"batch_id": "mixed-batch-01", "review_path": str(review_path)},
    )
    promote_job = repository.claim_next(clock=fixed_clock)
    process_job(
        promote_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=DeterministicFakeModel, clock=fixed_clock,
    )
    after_promote = repository.get(promote_job.job_id)
    assert after_promote.status == "completed"

    pointer_path = manifest.approved_bank_path.parent / "ai-active-bank.json"
    pointer = json.loads(pointer_path.read_text())
    bank_items = [json.loads(line) for line in Path(pointer["path"]).read_text().splitlines()]

    assert len(bank_items) == 2
    exported_ids = {item["item_id"] for item in bank_items}
    approved_revision_id = next(
        revision.revision_id for revision in revised_item.revisions if revision.final_review_status == "approved"
    )
    assert exported_ids == {as_written_item.original_question_id, approved_revision_id}
    as_written_export = next(item for item in bank_items if item["item_id"] == as_written_item.original_question_id)
    assert as_written_export["question"] == source_by_id[as_written_item.original_question_id].question.model_dump(
        exclude={"intent_id"}, mode="json"
    )
