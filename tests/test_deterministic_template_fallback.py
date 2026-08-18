"""Regression coverage for the deterministic template fallback (LA-DET-01-INT-05).

Three layers, cheapest first:

1. Pure math/registry tests for authoring/deterministic_templates.py's determinant
   template -- no model, no pipeline.
2. generate_batch-level tests proving the fallback mechanism itself: it only
   substitutes after the existing per-question retry cap is exhausted, never for
   an infrastructure/configuration failure, and a broken template still has to
   pass the same deterministic validation a live attempt would. These use a
   synthetic template injected for a real, unrelated intent (AI-SRC-08-INT-01)
   so the mechanism is proven independently of the determinant template's own
   correctness (covered separately in layer 1) or the real LA taxonomy.
3. One generate_batch test against the REAL LA-DET-01 blueprint/taxonomy and the
   real DETERMINISTIC_TEMPLATES registry, with a fake model reproducing the exact
   JSONDecodeError the live pilot rerun hit 3/3 times -- proving the real route,
   not just a synthetic fixture.
4. One worker-level test (process_job) proving a template-authored candidate
   still goes through automated review and stops at the human-review boundary --
   it cannot bypass promotion just because no model produced it.
"""

import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from api.bank import BankItem
from api.schemas import QuizQuestion
from authoring.deterministic_templates import (
    DETERMINISTIC_TEMPLATES,
    determinant,
    generate_determinant_question,
)
from authoring.grounded_batch import (
    TEMPLATE_MODEL_ID,
    TEMPLATE_MODEL_REVISION,
    BatchConfig,
    generate_batch,
)
from authoring.question_intents import intents_by_skill, load_blueprint_for_batch
from taxonomy.loader import course_paths, course_provenance_path

from tests.test_grounded_batch import DeterministicFakeModel, GIT_COMMIT, config as ai_config
from tests.test_grounded_batch import SKILLS_PATH as AI_SKILLS_PATH
from tests.test_grounded_batch import REFERENCES_PATH as AI_REFERENCES_PATH
from tests.test_grounded_batch import PROVENANCE_PATH as AI_PROVENANCE_PATH

FIXED_TIME = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)
LA_SKILLS_PATH, LA_REFERENCES_PATH = course_paths("linear-algebra")
LA_PROVENANCE_PATH = course_provenance_path("linear-algebra")
LA_BLUEPRINT_ID = "grounded-linear-algebra-v1"
LA_SKILL_ID = "LA-DET-01"
LA_INTENT_ID = "LA-DET-01-INT-05"

# The exact string the live Modal endpoint produced 3/3 times against
# LA-DET-01-INT-05 in the disposable pilot rerun (LaTeX-style matrix notation
# breaking JSON escaping) -- reproduced here so the fallback trigger condition
# is proven against the real recorded failure mode, not an invented one.
REAL_JSON_DECODE_FAILURE_TEXT = r'{"questions": [{"question": "det \[ \begin{bmatrix} 1 & 2 \end{bmatrix} \]"}]}'


def _leibniz_det3(matrix: list[list[int]]) -> int:
    """An independent determinant implementation (permutation expansion, not
    cofactor expansion) used only to cross-check determinant()'s correctness."""
    total = 0
    for perm in itertools.permutations(range(3)):
        inversions = sum(1 for i in range(3) for j in range(i + 1, 3) if perm[i] > perm[j])
        sign = -1 if inversions % 2 else 1
        product = 1
        for row, col in enumerate(perm):
            product *= matrix[row][col]
        total += sign * product
    return total


def _la_intent():
    blueprint = load_blueprint_for_batch(LA_BLUEPRINT_ID)
    pool = intents_by_skill(blueprint)[LA_SKILL_ID]
    return next(intent for intent in pool if intent.intent_id == LA_INTENT_ID)


def _la_skill():
    from taxonomy.loader import load_skills

    catalogue = load_skills(LA_SKILLS_PATH, LA_REFERENCES_PATH)
    return next(skill for skill in catalogue.skills if skill.skill_id == LA_SKILL_ID)


# --- Layer 1: pure math/registry ---


def test_determinant_matches_an_independent_permutation_formula():
    for seed in range(50):
        question = generate_determinant_question(_la_intent(), _la_skill(), [], seed, "intermediate")
        matrix = json.loads(
            question.question.split("A = ", 1)[1].split(" by cofactor", 1)[0]
        )
        assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
        assert determinant(matrix) == _leibniz_det3(matrix)
        assert int(question.correct_answer) == determinant(matrix)


def test_determinant_question_has_exactly_one_correct_option_and_distinct_options():
    for seed in range(50):
        question = generate_determinant_question(_la_intent(), _la_skill(), [], seed, "intermediate")
        assert len(question.options) == 4
        assert len(set(question.options)) == 4
        assert sum(option == question.correct_answer for option in question.options) == 1
        assert question.correct_answer in question.options


def test_determinant_matrices_are_always_square():
    for seed in range(20):
        question = generate_determinant_question(_la_intent(), _la_skill(), [], seed, "intermediate")
        matrix = json.loads(question.question.split("A = ", 1)[1].split(" by cofactor", 1)[0])
        assert len({len(row) for row in matrix} | {len(matrix)}) == 1


def test_registry_only_covers_the_one_reference_archetype():
    assert set(DETERMINISTIC_TEMPLATES) == {LA_INTENT_ID}


# --- Layer 2: fallback mechanism, isolated from the determinant template itself ---


def _template_registered_for_ai_src_08_int_01(question: str, options: list[str], correct_answer: str):
    def template(intent, skill, references, seed, difficulty):
        from authoring.grounded_batch import IntentQuestion

        return IntentQuestion(
            question=question,
            options=options,
            correct_answer=correct_answer,
            explanation="Deterministically constructed for this test.",
            concept="test concept",
            difficulty=difficulty,
            intent_id=intent.intent_id,
        )

    return {"AI-SRC-08-INT-01": template}


def test_fallback_activates_only_after_the_existing_retry_cap_is_exhausted(tmp_path):
    model = DeterministicFakeModel(["bad json", "still bad json"])
    templates = _template_registered_for_ai_src_08_int_01(
        "Which statement is deterministically constructed?",
        ["Correct template answer", "Wrong A", "Wrong B", "Wrong C"],
        "Correct template answer",
    )
    result = generate_batch(
        ai_config(skill_ids=["AI-SRC-08"], questions_per_skill=1, max_attempts_per_question=2),
        model,
        tmp_path / "batch",
        skills_path=AI_SKILLS_PATH,
        references_path=AI_REFERENCES_PATH,
        provenance_path=AI_PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
        templates=templates,
    )

    assert result.status == "complete"
    assert len(model.calls) == 2, "the template must not run before the retry cap is exhausted"
    assert [attempt.validation_status for attempt in result.attempts] == ["invalid", "invalid", "accepted"]
    assert result.attempts[-1].generation_method == "deterministic_template"
    assert result.questions[0].generation_method == "deterministic_template"
    assert result.questions[0].model_id == TEMPLATE_MODEL_ID
    assert result.questions[0].model_revision == TEMPLATE_MODEL_REVISION
    assert result.questions[0].question.correct_answer == "Correct template answer"


def test_fallback_does_not_activate_for_infrastructure_failures(tmp_path):
    class ModelUnavailableError(RuntimeError):
        """Named to match the exact class authoring/replenishment/worker.py raises --
        generate_batch's exception handler identifies an infrastructure failure by
        exception class name (type(error).__name__), the same convention worker.py's
        own model_unavailable check already relies on."""

    class ModelDown:
        model_id = DeterministicFakeModel.model_id
        model_revision = DeterministicFakeModel.model_revision

        def generate(self, messages, seed, generation_parameters):
            raise ModelUnavailableError("no GPU available in this test")

        calls: list = []

    templates = _template_registered_for_ai_src_08_int_01(
        "Should never be used?", ["A", "B", "C", "D"], "A"
    )
    result = generate_batch(
        ai_config(skill_ids=["AI-SRC-08"], questions_per_skill=1, max_attempts_per_question=2),
        ModelDown(),
        tmp_path / "batch",
        skills_path=AI_SKILLS_PATH,
        references_path=AI_REFERENCES_PATH,
        provenance_path=AI_PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
        templates=templates,
    )

    assert result.status == "incomplete"
    assert all(attempt.generation_method == "model" for attempt in result.attempts)
    assert not result.questions


def test_invalid_template_output_still_fails_normal_validation(tmp_path):
    templates = _template_registered_for_ai_src_08_int_01(
        "Broken template output with duplicate options?",
        ["same", "same", "b", "c"],
        "same",
    )
    result = generate_batch(
        ai_config(skill_ids=["AI-SRC-08"], questions_per_skill=1, max_attempts_per_question=1),
        DeterministicFakeModel(["bad json"]),
        tmp_path / "batch",
        skills_path=AI_SKILLS_PATH,
        references_path=AI_REFERENCES_PATH,
        provenance_path=AI_PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
        templates=templates,
    )

    assert result.status == "incomplete"
    assert not result.questions
    template_attempt = result.attempts[-1]
    assert template_attempt.generation_method == "deterministic_template"
    assert template_attempt.validation_status == "invalid"
    assert "duplicate" in template_attempt.validation_error.lower()


def test_non_target_intents_are_unaffected_by_a_populated_registry(tmp_path):
    """Passing the real, non-empty DETERMINISTIC_TEMPLATES registry into a batch
    for a skill/intent it does not cover must behave identically to templates=None."""
    with_templates = generate_batch(
        ai_config(skill_ids=["AI-SRC-08"], questions_per_skill=1),
        DeterministicFakeModel(),
        tmp_path / "with",
        skills_path=AI_SKILLS_PATH,
        references_path=AI_REFERENCES_PATH,
        provenance_path=AI_PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
        templates=DETERMINISTIC_TEMPLATES,
    )
    without_templates = generate_batch(
        ai_config(skill_ids=["AI-SRC-08"], questions_per_skill=1),
        DeterministicFakeModel(),
        tmp_path / "without",
        skills_path=AI_SKILLS_PATH,
        references_path=AI_REFERENCES_PATH,
        provenance_path=AI_PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
    )

    assert with_templates.status == without_templates.status == "complete"
    assert with_templates.questions[0].generation_method == "model"
    assert (
        with_templates.questions[0].model_dump(mode="json", exclude={"generated_at"})
        == without_templates.questions[0].model_dump(mode="json", exclude={"generated_at"})
    )


# --- Layer 3: the real LA-DET-01-INT-05 route ---


def test_real_la_det_01_int_05_route_falls_back_after_three_real_construction_failures(tmp_path):
    """Reproduces the pilot's exact recorded failure (JSONDecodeError from
    LaTeX-style matrix notation breaking JSON escaping, 3/3 live attempts) against
    the real blueprint and real taxonomy, and proves the real registry entry
    (not a synthetic one) resolves and produces a valid, reviewable candidate."""
    model = DeterministicFakeModel(
        [REAL_JSON_DECODE_FAILURE_TEXT, REAL_JSON_DECODE_FAILURE_TEXT, REAL_JSON_DECODE_FAILURE_TEXT]
    )
    blueprint = load_blueprint_for_batch(LA_BLUEPRINT_ID)
    pool = intents_by_skill(blueprint)[LA_SKILL_ID]
    target_index = next(index for index, intent in enumerate(pool) if intent.intent_id == LA_INTENT_ID)

    result = generate_batch(
        BatchConfig(
            batch_id=LA_BLUEPRINT_ID,
            skill_ids=[LA_SKILL_ID],
            questions_per_skill=len(pool),
            base_seed=20260818,
            model_id=model.model_id,
            prompt_version=blueprint.prompt_version,
            difficulty="mixed",
        ),
        model,
        tmp_path / "la-batch",
        skills_path=LA_SKILLS_PATH,
        references_path=LA_REFERENCES_PATH,
        provenance_path=LA_PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
        skip_question_indices=frozenset(range(len(pool))) - {target_index},
        templates=DETERMINISTIC_TEMPLATES,
    )

    assert result.status == "complete"
    assert len(model.calls) == 3, "must exhaust exactly the existing retry cap before falling back"
    target_question = next(q for q in result.questions if q.intent_id == LA_INTENT_ID)
    assert target_question.generation_method == "deterministic_template"
    assert target_question.model_id == TEMPLATE_MODEL_ID
    matrix = json.loads(
        target_question.question.question.split("A = ", 1)[1].split(" by cofactor", 1)[0]
    )
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
    assert determinant(matrix) == int(target_question.question.correct_answer)


# --- Layer 4: worker-level human-review boundary ---


@pytest.fixture
def _la_worker_fixtures(tmp_path, monkeypatch):
    import authoring.grounding_briefs as grounding_briefs
    import authoring.question_intents as question_intents
    from authoring.question_intents import PilotBlueprint, QuestionIntent
    from authoring.replenishment.manifest import CourseManifest
    from authoring.grounding_briefs import CanonicalGroundingBrief
    from authoring.retrieval.models import new_candidate, approve
    from authoring.retrieval.store import CandidateStore

    taxonomy_dir = tmp_path / "taxonomy"
    taxonomy_dir.mkdir()
    (taxonomy_dir / "skills.csv").write_text(
        "skill_id,topic,subtopic,name,learning_objective,cognitive_process,generation_strategy,template_id,prerequisite_skill_ids\n"
        "LA-DET-01,Determinants,Cofactor expansion,Determinants,"
        "Compute a matrix determinant by cofactor expansion,apply,generated,,\n",
        encoding="utf-8",
    )
    (taxonomy_dir / "references.csv").write_text("skill_id,reference_material\n", encoding="utf-8")

    manifest = CourseManifest(
        course_id="linear-algebra-test",
        title="test",
        version="1",
        taxonomy_path=taxonomy_dir,
        approved_bank_path=tmp_path / "bank" / "la-bank-v0.jsonl",
        bkt_model_path=tmp_path / "model.pkl",
        candidate_store_path=tmp_path / "reference_candidates.json",
        review_store_path=tmp_path / "reviews",
        allowed_domains=("example.edu",),
        low_supply_threshold=1,
        target_supply=1,
        default_bkt_model_version="test-v1",
        status="active",
    )

    candidate = new_candidate(
        LA_SKILL_ID,
        "Determinants",
        "https://example.edu/linear-algebra/determinants.html",
        "example.edu",
        "The determinant of a square matrix is computed by cofactor expansion.",
        FIXED_TIME,
        relevance_score=10,
        matched_terms=["determinant"],
    )
    store = CandidateStore(manifest.candidate_store_path)
    store.add([candidate])
    store.replace(approve(candidate, "albert", reviewed_at=FIXED_TIME))

    blueprint_dir = tmp_path / "blueprints"
    blueprint_dir.mkdir()
    intent = QuestionIntent(
        intent_id=LA_INTENT_ID,
        skill_id=LA_SKILL_ID,
        assessment_focus="Apply cofactor expansion to compute a determinant.",
        question_archetype="cofactor-expansion computation",
        preferred_reference_ids=[candidate.candidate_id],
        required_concepts=["cofactor expansion", "determinant computation"],
        prohibited_conflations=["expanding a different row changes the value"],
        difficulty="intermediate",
    )
    blueprint = PilotBlueprint(
        batch_id="test-la-batch-01",
        prompt_version=question_intents.PILOT_PROMPT_VERSION,
        review_status="blueprint-approved",
        reviewer_id="albert",
        reviewed_at=FIXED_TIME,
        base_seed=1,
        intents=[intent],
    )
    (blueprint_dir / "test-la-batch-01.json").write_text(
        json.dumps(blueprint.model_dump(mode="json")), encoding="utf-8"
    )
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", blueprint_dir)
    monkeypatch.setattr(
        grounding_briefs,
        "PILOT_GROUNDING_BRIEFS",
        {
            LA_SKILL_ID: CanonicalGroundingBrief(
                skill_id=LA_SKILL_ID,
                version="test-v1",
                statements=["The determinant of a square matrix is computed by cofactor expansion."],
            )
        },
    )
    return manifest


def test_worker_routes_a_construction_failure_through_the_real_registry_and_stops_at_human_review(
    _la_worker_fixtures,
):
    from authoring.replenishment.jobs import SQLiteReplenishmentJobRepository
    from authoring.replenishment.worker import process_job
    from authoring.review.reviewer import FakeContentReviewer
    from authoring.review.models import (
        AnswerAssessment, DifficultyAssessment, DuplicateAssessment,
        GroundingAssessment, ObjectiveAssessment, SemanticReviewResult,
    )

    manifest = _la_worker_fixtures
    repository = SQLiteReplenishmentJobRepository(":memory:")
    repository.initialize_schema()

    model = DeterministicFakeModel(
        [REAL_JSON_DECODE_FAILURE_TEXT, REAL_JSON_DECODE_FAILURE_TEXT, REAL_JSON_DECODE_FAILURE_TEXT]
    )

    def fixed_clock():
        return FIXED_TIME

    repository.enqueue(
        course_id=manifest.course_id, skill_id=LA_SKILL_ID, requested_count=1,
        clock=fixed_clock, job_type="generate_questions",
    )
    generate_job = repository.claim_next(clock=fixed_clock)
    process_job(
        generate_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=lambda: model, clock=fixed_clock,
    )
    after_generate = repository.get(generate_job.job_id)

    assert after_generate.status == "queued"
    assert after_generate.job_type == "automated_review"
    assert len(model.calls) == 3, "must not fall back before the real retry cap is exhausted"

    def review_result() -> SemanticReviewResult:
        return SemanticReviewResult(
            grounding_assessment=GroundingAssessment(
                grounded=True, independently_supported_answer=True,
                supporting_reference_ids=[], grounding_confidence=0.9,
            ),
            answer_assessment=AnswerAssessment(
                selected_option_text="placeholder", matches_declared_answer=True,
                multiple_defensible_answers=False, obviously_signalled_answer=False,
                answer_confidence=0.9,
            ),
            objective_assessment=ObjectiveAssessment(
                measures_declared_skill=True, satisfies_intent_blueprint=True,
                matches_objective_verb=True, cognitive_demand="apply",
                duplicates_another_intent=False,
            ),
            difficulty_assessment=DifficultyAssessment(
                difficulty_justified=True, explanation_depth_matches_difficulty=True,
                is_definition_recall_only=False,
            ),
            duplicate_assessment=DuplicateAssessment(),
            reviewer_model_id="fake-reviewer",
            reviewer_model_revision="fake-rev-1",
            reviewer_prompt_version="review-v1",
            reviewer_prompt_template_hash="d" * 64,
            rendered_review_request_hash="d" * 64,
        )

    class MatchAnyReviewer:
        """A FakeContentReviewer that answers for whatever stem the template
        deterministically produced, without hardcoding the exact wording."""

        model_id = "fake-reviewer"
        model_revision = "fake-rev-1"

        def review(self, question, skill, intent, references, *, seed):
            outcome = review_result()
            return outcome.model_copy(
                update={"answer_assessment": outcome.answer_assessment.model_copy(
                    update={"selected_option_text": question.correct_answer}
                )}
            )

    review_job = repository.get(generate_job.job_id)
    process_job(
        review_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        reviewer_factory=lambda: MatchAnyReviewer(), clock=fixed_clock,
    )
    after_review = repository.get(generate_job.job_id)

    # The candidate came from a template, not the model -- it must still have gone
    # through the real automated reviewer (a report now exists) and must still stop
    # at a human-actionable boundary, never completing/promoting on its own.
    assert after_review.status in (
        "waiting_for_question_review", "waiting_for_full_human_review",
    )
    assert after_review.status != "completed"

    review_path = Path(after_review.metadata["review_path"])
    from authoring.grounded_review import GroundedReviewStore

    review = GroundedReviewStore(review_path).load()
    assert all(item.final_review_status == "pending" for item in review.items)

    # Attempting to promote before any human decision must find nothing approved --
    # a template-authored candidate gets no special promotion path.
    premature_promote_job = repository.mark_queued(review_job.job_id, job_type="promote_approved_items")
    process_job(
        premature_promote_job, manifest, job_repository=repository,
        search_provider=None, fetcher=None, clock=fixed_clock,
    )
    after_premature_promote = repository.get(review_job.job_id)
    assert after_premature_promote.status == "permanent_failure"
    assert after_premature_promote.error_code == "no_approved_items"
