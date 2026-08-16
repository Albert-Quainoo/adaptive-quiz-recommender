"""Deterministic, fully local demonstration of the replenishment pipeline.

Everything here runs under a temporary directory with fake search, fetch and
model providers -- no network call, no live Llama call, and no write to any
real file under manifests/, outputs/, or taxonomy/data/. Run it twice; the
second run enqueues zero new jobs.

    python -m scripts.demo_replenishment_pipeline
"""

import json
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import authoring.grounding_briefs as grounding_briefs
import authoring.question_intents as question_intents
from api.schemas import QuizQuestion
from authoring.grounded_batch import IntentQuestion, PendingQuestion, read_jsonl
from authoring.grounded_review import (
    GroundedReviewStore,
    RevisionProvenance,
    approve_revision,
    propose_revision,
    question_content_hash,
)
from authoring.grounding_briefs import PILOT_GROUNDING_BRIEFS, CanonicalGroundingBrief
from authoring.question_intents import PilotBlueprint, QuestionIntent
from authoring.replenishment.inventory import compute_course_inventory
from authoring.replenishment.jobs import SQLiteReplenishmentJobRepository
from authoring.replenishment.manifest import CourseManifest
from authoring.replenishment.policy import ReplenishmentPolicyConfig, decide_replenishment
from authoring.replenishment.worker import ready_to_resume, process_job
from authoring.retrieval.models import SearchResult
from authoring.retrieval.search import FetchedPage
from authoring.retrieval.store import CandidateStore
from authoring.review.backpressure import apply_review_backlog_caps
from authoring.review.config import ReviewPolicyConfig
from authoring.review.models import (
    AnswerAssessment,
    DifficultyAssessment,
    DuplicateAssessment,
    GroundingAssessment,
    ObjectiveAssessment,
    SemanticReviewResult,
)
from authoring.review.reports import AutomatedReviewReportStore
from authoring.review.reviewer import FakeContentReviewer
from authoring.review.service import review_candidate
from recommendation.policy import RecommendationPolicyConfig
from recommendation.schemas import RecommendationRequest
from recommendation.service import RecommendationService
from recommendation.sqlite_repository import SQLiteRecommendationRepository
from taxonomy.loader import load_skills
from taxonomy.schemas import ReferenceProvenance, SkillDefinition

REPO_ROOT = Path(__file__).resolve().parents[1]
SKILL_ID = "AI-SRC-08"
FIXED_TIME = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)

PASSAGE = (
    "A heuristic function estimates the cost of the cheapest path from a given "
    "state to a goal state. It lets an informed search order the frontier by "
    "how promising a state looks rather than by how far it already is, which "
    "is what separates informed search from uninformed search."
)


def step(label: str) -> None:
    print(f"\n=== {label} ===")


class FakeSearchProvider:
    def __init__(self, results):
        self.results = results

    def search(self, schedule, diagnostics, budget):
        for search_step in schedule:
            if not budget.may_request():
                return
            budget.spend_request()
            diagnostics.record_query(search_step.domain)
            for result in self.results:
                yield search_step, result


class FakePageFetcher:
    def __init__(self, pages):
        self.pages = pages

    def fetch(self, url):
        return FetchedPage(url=url, text=self.pages[url])


class DeterministicFakeModel:
    model_id = "demo-fake-model"
    model_revision = "demo-fake-revision-1"

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


def clean_semantic_result(correct_answer: str) -> SemanticReviewResult:
    """A reviewer pass that finds nothing wrong -- used wherever this demo only
    needs the pipeline to reach human review, not automated review's own opinion."""
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
        reviewer_model_id="demo-fake-reviewer",
        reviewer_model_revision="demo-fake-rev-1",
        reviewer_prompt_version="review-v1",
        reviewer_prompt_template_hash="d" * 64,
        rendered_review_request_hash="d" * 64,
    )


def build_demo_manifest(tmp_path: Path) -> CourseManifest:
    taxonomy_dir = tmp_path / "taxonomy"
    taxonomy_dir.mkdir()
    shutil.copy(REPO_ROOT / "taxonomy/data/ai/skills.csv", taxonomy_dir / "skills.csv")
    (taxonomy_dir / "references.csv").write_text("skill_id,reference_material\n", encoding="utf-8")

    # Lowers the threshold for this one demo skill only -- this manifest
    # lives entirely in the temp directory; authoring/replenishment/manifests
    # /ai.json on disk is never read or written by this script.
    return CourseManifest(
        course_id="ai",
        title="Introduction to Artificial Intelligence (demo copy)",
        version="demo",
        taxonomy_path=taxonomy_dir,
        approved_bank_path=tmp_path / "bank" / "ai-bank-v0.jsonl",
        candidate_store_path=tmp_path / "reference_candidates.json",
        review_store_path=tmp_path / "reviews",
        allowed_domains=("example.edu",),
        low_supply_threshold=1,
        target_supply=1,
        default_bkt_model_version="demo-v1",
        status="active",
    )


def build_blueprint(tmp_path: Path, candidate_id: str) -> PilotBlueprint:
    blueprint_dir = tmp_path / "blueprints"
    blueprint_dir.mkdir()
    intent = QuestionIntent(
        intent_id=f"{SKILL_ID}-INT-01",
        skill_id=SKILL_ID,
        assessment_focus="What a heuristic estimates",
        question_archetype="definition recall",
        preferred_reference_ids=[candidate_id],
        required_concepts=["heuristic", "estimate"],
        prohibited_conflations=["heuristic equals exact cost"],
        difficulty="intermediate",
    )
    blueprint = PilotBlueprint(
        batch_id="demo-batch-01",
        prompt_version=question_intents.PILOT_PROMPT_VERSION,
        review_status="blueprint-approved",
        reviewer_id="demo",
        reviewed_at=FIXED_TIME,
        base_seed=1,
        intents=[intent],
    )
    (blueprint_dir / "demo-batch-01.json").write_text(
        json.dumps(blueprint.model_dump(mode="json")), encoding="utf-8"
    )
    question_intents.BLUEPRINT_DIRECTORY = blueprint_dir
    grounding_briefs.PILOT_GROUNDING_BRIEFS = {
        SKILL_ID: CanonicalGroundingBrief(
            skill_id=SKILL_ID,
            version="demo-v1",
            statements=["A heuristic estimates the remaining cost to the goal."],
        )
    }
    return blueprint


def approve_the_generated_question(review_path: Path, output_dir: Path) -> None:
    review = GroundedReviewStore(review_path).load()
    item = review.items[0]
    source = next(
        question
        for question in read_jsonl(output_dir / "pending_questions.jsonl", PendingQuestion)
        if question.question_id == item.original_question_id
    )
    revised = QuizQuestion.model_validate(
        source.question.model_dump()
        | {"explanation": source.question.explanation + " (reviewed by demo)"}
    )
    proposed = propose_revision(
        item, source.question, revised, "demo-reviewer", "approved as generated",
        edited_at=FIXED_TIME, provenance=RevisionProvenance.from_source(source),
    )
    GroundedReviewStore(review_path).replace_item(proposed)
    approved = approve_revision(
        proposed, proposed.revisions[0].revision_id, "demo-reviewer", reviewed_at=FIXED_TIME
    )
    GroundedReviewStore(review_path).replace_item(approved)


def prove_learner_app_still_works() -> None:
    """The real, untouched 38-item bank and BKT model -- served the whole time."""
    bank_path = REPO_ROOT / "outputs/approved_banks/pilot-approved-bank-38-v1.jsonl"
    skills_path = REPO_ROOT / "taxonomy/data/ai/skills.csv"
    references_path = REPO_ROOT / "taxonomy/data/ai/references.csv"
    from app.bootstrap import load_approved_bank

    catalogue = load_skills(skills_path, references_path)
    items = load_approved_bank(bank_path)

    with tempfile.TemporaryDirectory() as tmp:
        database_path = Path(tmp) / "learner-demo.sqlite3"
        repository = SQLiteRecommendationRepository(
            database_path, course_id="intro-ai", skills=catalogue.skills, items=items
        )
        repository.initialize_schema()
        policy = RecommendationPolicyConfig(policy_version="demo-policy-v1")
        service = RecommendationService(
            repository, course_id="intro-ai", model_version="demo-model-v1", config=policy
        )
        available_skill_ids = sorted({item.skill_id for item in items})
        result = service.recommend(
            RecommendationRequest(
                learner_id="demo-learner", available_skill_ids=available_skill_ids
            )
        )
        print(
            f"Learner app served {result.skill_id}/{result.item_id} "
            f"(reason={result.reason}) -- from the real, untouched 38-item bank."
        )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="replenishment-demo-") as raw_tmp:
        tmp_path = Path(raw_tmp)

        step("1. Learner app usable before any replenishment work")
        prove_learner_app_still_works()

        step("2. Build a demo manifest with a lowered threshold for one skill")
        manifest = build_demo_manifest(tmp_path)
        print(f"{SKILL_ID}: low_supply_threshold={manifest.low_supply_threshold}")

        repository = SQLiteReplenishmentJobRepository(tmp_path / "jobs.sqlite3")
        repository.initialize_schema()
        inventory = compute_course_inventory(manifest, repository)
        assert inventory[SKILL_ID].readiness == "taxonomy_only"

        config = ReplenishmentPolicyConfig(
            low_supply_threshold=manifest.low_supply_threshold,
            target_supply=manifest.target_supply,
        )
        decisions = [
            decision
            for decision in decide_replenishment({SKILL_ID: inventory[SKILL_ID]}, config)
            if decision.should_enqueue
        ]
        assert len(decisions) == 1, "expected exactly one replenishment decision"
        job = repository.enqueue(
            course_id="ai", skill_id=SKILL_ID, requested_count=decisions[0].requested_count
        )
        print(f"Enqueued exactly one job: {job.job_id} ({job.job_type}/{job.status})")

        step("3. Worker processes retrieval with fake search + fetch providers")
        url = "https://example.edu/pathfinding/heuristics.html"
        provider = FakeSearchProvider([SearchResult(title="Heuristics", url=url, snippet="")])
        fetcher = FakePageFetcher({url: PASSAGE})
        retrieve_job = repository.claim_next()
        process_job(
            retrieve_job, manifest, job_repository=repository,
            search_provider=provider, fetcher=fetcher,
            model_factory=DeterministicFakeModel, clock=lambda: FIXED_TIME,
        )
        after_retrieve = repository.get(retrieve_job.job_id)
        print(f"Job status: {after_retrieve.status}")

        step("4. Pauses at reference review")
        candidates = CandidateStore(manifest.candidate_store_path).load()
        pending = [c for c in candidates if c.review_status == "pending"]
        assert after_retrieve.status == "waiting_for_reference_review"
        assert pending, "expected at least one pending reference candidate"
        print(f"{len(pending)} pending reference candidate(s), e.g. {pending[0].candidate_id}")

        step("5. Resumes after a human approves the reference")
        from authoring.retrieval.models import approve

        store = CandidateStore(manifest.candidate_store_path)
        store.replace(approve(pending[0], "demo-reviewer", reviewed_at=FIXED_TIME))
        blueprint = build_blueprint(tmp_path, pending[0].candidate_id)
        assert ready_to_resume(after_retrieve, manifest) is True
        repository.mark_queued(retrieve_job.job_id, job_type="retrieve_references")
        retrieve_again = repository.claim_next()
        process_job(
            retrieve_again, manifest, job_repository=repository,
            search_provider=provider, fetcher=fetcher,
            model_factory=DeterministicFakeModel, clock=lambda: FIXED_TIME,
        )
        generate_job = repository.claim_next()
        process_job(
            generate_job, manifest, job_repository=repository,
            search_provider=provider, fetcher=fetcher,
            model_factory=DeterministicFakeModel, clock=lambda: FIXED_TIME,
        )
        after_generate_stage = repository.get(generate_job.job_id)
        print(f"Job status: {after_generate_stage.status}/{after_generate_stage.job_type}")

        step("5b. Automated review runs before any human sees this batch")
        review_job = repository.claim_next()
        process_job(
            review_job, manifest, job_repository=repository,
            search_provider=provider, fetcher=fetcher,
            model_factory=DeterministicFakeModel,
            reviewer_factory=lambda: FakeContentReviewer(
                {"What does a heuristic estimate?": clean_semantic_result("Remaining cost to goal")}
            ),
            clock=lambda: FIXED_TIME,
        )
        after_generate = repository.get(generate_job.job_id)
        print(f"Job status: {after_generate.status}")
        review = GroundedReviewStore(Path(after_generate.metadata["review_path"])).load()
        print(
            f"Automated review recommendation for {review.items[0].original_question_id}: "
            f"{review.items[0].recommendation!r} -- still pending human approval "
            f"(final_review_status={review.items[0].final_review_status!r})"
        )

        step("6. Pauses at question review")
        assert after_generate.status == "waiting_for_question_review"
        review_path = Path(after_generate.metadata["review_path"])
        review = GroundedReviewStore(review_path).load()
        print(f"{len(review.items)} generated question(s) pending human review")

        step("7. Promotes approved content into a temporary versioned bank")
        output_dir = review_path.parent.parent / "batches" / "demo-batch-01__AI-SRC-08"
        approve_the_generated_question(review_path, output_dir)
        assert ready_to_resume(after_generate, manifest) is True
        repository.mark_queued(generate_job.job_id, job_type="promote_approved_items")
        promote_job = repository.claim_next()
        process_job(
            promote_job, manifest, job_repository=repository,
            search_provider=provider, fetcher=fetcher,
            model_factory=DeterministicFakeModel, clock=lambda: FIXED_TIME,
        )
        after_promote = repository.get(promote_job.job_id)
        pointer_path = manifest.approved_bank_path.parent / "ai-active-bank.json"
        pointer = json.loads(pointer_path.read_text())
        print(
            f"Job status: {after_promote.status}. New versioned bank: "
            f"{pointer['path']} (version {pointer['version']})"
        )

        step("8. Learner app remained usable the entire time")
        prove_learner_app_still_works()

        step("9. The inaccurate heuristic candidate is blocked, never silently promoted")
        demo_skill = SkillDefinition(
            skill_id=SKILL_ID, topic="Search", subtopic="Informed search", name="Heuristics",
            learning_objective="Explain how a heuristic estimates the remaining cost to the goal.",
            cognitive_process="understand", generation_strategy="generated",
            reference_material=["A heuristic estimates the remaining cost to the goal."],
        )
        demo_intent = QuestionIntent(
            intent_id=f"{SKILL_ID}-INT-01", skill_id=SKILL_ID,
            assessment_focus="Distinguish h(n) from g(n)", question_archetype="g-versus-h",
            preferred_reference_ids=["AI-SRC-08-REF-01"],
            required_concepts=["g(n)", "h(n)"], prohibited_conflations=["g and h are interchangeable"],
        )
        demo_reference = ReferenceProvenance(
            reference_id="AI-SRC-08-REF-01", skill_id=SKILL_ID,
            reference_material=PILOT_GROUNDING_BRIEFS[SKILL_ID].statements[0],
            title="Informed search", source_url="https://example.edu/informed-search.html",
            source_domain="example.edu", content_hash="a" * 64,
            retrieved_at=FIXED_TIME, reviewer_id="demo-reviewer", reviewed_at=FIXED_TIME,
        )
        inaccurate_question = IntentQuestion(
            intent_id=demo_intent.intent_id,
            question="What is the heuristic value h(n) if we have travelled 3 units and 5 remain?",
            options=[
                "The cost of the cheapest path from the initial state to the goal state",
                "The accumulated cost from the start, which is 3 units",
                "The total cost from start to goal, which is 8 units",
                "The sum of accumulated and estimated remaining cost",
            ],
            correct_answer="The cost of the cheapest path from the initial state to the goal state",
            explanation="h(n) is the cost of the cheapest path from the initial state to the goal state.",
            concept="heuristic value h(n)", difficulty="introductory",
        )
        inaccurate_candidate = PendingQuestion(
            batch_id="demo-regression", question_id="demo-regression-original", skill_id=SKILL_ID,
            question_index=0, intent_id=demo_intent.intent_id, seed=1,
            reference_ids=[demo_reference.reference_id], prompt_version="v3.3", prompt_hash="b" * 64,
            model_id="demo-fake-model", model_revision="demo-fake-revision-1", generation_parameters={},
            generated_at=FIXED_TIME, git_commit="0" * 40, raw_response="{}", question=inaccurate_question,
        )
        conflated_result = clean_semantic_result(
            "The remaining cost from the current state n to a goal state"
        ).model_copy(
            update={
                "grounding_assessment": GroundingAssessment(
                    grounded=False, independently_supported_answer=False,
                    contradictions=["declared answer describes path cost, not h(n)'s remaining-cost estimate"],
                    grounding_confidence=0.2,
                ),
                "answer_assessment": AnswerAssessment(
                    selected_option_text="The remaining cost from the current state n to a goal state",
                    matches_declared_answer=False, multiple_defensible_answers=False,
                    obviously_signalled_answer=False, answer_confidence=0.9,
                ),
            }
        )
        regression_report = review_candidate(
            inaccurate_candidate, demo_skill, demo_intent, [demo_reference],
            reviewer=FakeContentReviewer({inaccurate_question.question: conflated_result}),
            config=ReviewPolicyConfig(), report_store=AutomatedReviewReportStore(tmp_path / "demo-reports.json"),
            clock=lambda: FIXED_TIME,
        )
        assert regression_report.recommendation != "recommend_human_approval"
        print(
            f"Original inaccurate candidate: risk={regression_report.risk_level}, "
            f"recommendation={regression_report.recommendation!r} -- never eligible for promotion."
        )

        step("10. An automatic revision is proposed and receives a fresh, from-scratch review")
        from authoring.review.revision import generate_revision_candidate, propose_automated_revision
        from authoring.grounded_review import CurationItem

        corrected_question = inaccurate_question.model_copy(
            update={
                "options": [
                    "The remaining cost from the current state n to a goal state",
                    *inaccurate_question.options[1:],
                ],
                "correct_answer": "The remaining cost from the current state n to a goal state",
                "explanation": (
                    "h(n) estimates the remaining cost from the current state n to a goal "
                    "state; the 3 units already travelled are g(n), not h(n)."
                ),
            }
        )

        class _RevisionFakeModel:
            model_id = "demo-fake-model"
            model_revision = "demo-fake-revision-1"

            def generate(self, messages, seed, generation_parameters):
                return json.dumps({"questions": [corrected_question.model_dump()]})

        revised_question = generate_revision_candidate(
            inaccurate_candidate, regression_report, demo_skill, demo_intent, [demo_reference],
            model=_RevisionFakeModel(), seed=1,
        )
        pending_item = CurationItem(
            original_question_id=inaccurate_candidate.question_id, skill_id=SKILL_ID,
            intent_id=demo_intent.intent_id, recommendation="propose_revision",
            recommendation_reason="pending automated review",
        )
        revised_item = propose_automated_revision(
            pending_item, inaccurate_question, revised_question, regression_report,
            RevisionProvenance.from_source(inaccurate_candidate), clock=lambda: FIXED_TIME,
        )
        revised_candidate = inaccurate_candidate.model_copy(update={"question": revised_question})
        fresh_report = review_candidate(
            revised_candidate, demo_skill, demo_intent, [demo_reference],
            reviewer=FakeContentReviewer(
                {corrected_question.question: clean_semantic_result(corrected_question.correct_answer)}
            ),
            config=ReviewPolicyConfig(),
            report_store=AutomatedReviewReportStore(tmp_path / "demo-reports.json"),
            clock=lambda: FIXED_TIME,
        )
        assert fresh_report.reviewed_content_hash != regression_report.reviewed_content_hash
        assert fresh_report.recommendation == "recommend_human_approval"
        assert revised_item.revisions[0].final_review_status == "pending"  # never auto-approved
        print(
            f"Revision {revised_item.revisions[0].revision_id} reviewed fresh "
            f"(content hash changed): risk={fresh_report.risk_level}, "
            f"recommendation={fresh_report.recommendation!r}, still pending human approval."
        )

        step("11. Backlog cap pauses further generation, learner quizzes stay available")
        capped = apply_review_backlog_caps(
            decide_replenishment({SKILL_ID: inventory[SKILL_ID]}, config),
            pending_per_skill={SKILL_ID: 10},
            total_pending_backlog=0,
            config=ReviewPolicyConfig(max_pending_per_skill=4),
        )
        assert all(not decision.should_enqueue for decision in capped)
        print(f"{SKILL_ID}: 10 pending questions >= cap of 4 -- generation paused, reason:")
        print(f"  {capped[0].reason}")
        prove_learner_app_still_works()

        step("Idempotency check")
        rerun_inventory = compute_course_inventory(manifest, repository)
        rerun_decisions = [
            decision
            for decision in decide_replenishment({SKILL_ID: rerun_inventory[SKILL_ID]}, config)
            if decision.should_enqueue
        ]
        print(f"Re-scanning after completion enqueues {len(rerun_decisions)} new job(s).")

        print("\nDemo complete: temporary directory cleaned up on exit.")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
