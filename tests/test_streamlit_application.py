from datetime import datetime, timezone

import pytest

from api.bank import BankItem
from api.schemas import QuizQuestion
from bkt.service import BKTService
from recommendation.policy import RecommendationPolicyConfig
from recommendation.service import RecommendationService, RecommendationUnavailable
from recommendation.sqlite_repository import SQLiteRecommendationRepository
from taxonomy.schemas import SkillDefinition

from app.controller import (
    ApplicationController,
    InvalidOptionError,
    NoApprovedItemError,
    NoRecommendationError,
)
from app.session import (
    begin_submission,
    get_session_state,
    identify_learner,
    next_question,
    retain_question,
    submission_succeeded,
)


NOW = datetime(2026, 8, 10, tzinfo=timezone.utc)
FOUNDATION = "AI-APP-01"
DEPENDENT = "AI-APP-02"


class DeterministicModel:
    model_version = "app-test-v1"

    def update_mastery(self, attempts):
        return 0.8 if attempts[-1].correct else 0.2


def skill(skill_id, prerequisites=()):
    return SkillDefinition(
        skill_id=skill_id,
        topic="Application integration",
        subtopic="Adaptive loop",
        name=f"Skill {skill_id}",
        learning_objective="Complete a deterministic integration question.",
        cognitive_process="understand",
        generation_strategy="hand_authored",
        prerequisite_skill_ids=list(prerequisites),
    )


def item(item_id, skill_id):
    return BankItem(
        item_id=item_id,
        skill_id=skill_id,
        provenance="hand_authored",
        question=QuizQuestion(
            question=f"Question for {skill_id}?",
            options=["Correct", "Wrong A", "Wrong B", "Wrong C"],
            correct_answer="Correct",
            explanation="The deterministic answer is Correct.",
            concept="Adaptive integration",
            difficulty="introductory",
        ),
    )


def build_controller(database, *, items=None, token="token-1"):
    skills = [skill(FOUNDATION), skill(DEPENDENT, [FOUNDATION])]
    items = items if items is not None else [
        item("foundation-item", FOUNDATION),
        item("dependent-item", DEPENDENT),
    ]
    repository = SQLiteRecommendationRepository(
        database, skills=skills, items=items
    )
    repository.initialize_schema()
    policy = RecommendationPolicyConfig(
        prerequisite_mastery_threshold=0.75,
        policy_version="app-policy-v1",
    )
    controller = ApplicationController(
        skills=skills,
        items=items,
        repository=repository,
        recommendation_service=RecommendationService(
            repository, model_version="app-test-v1", config=policy
        ),
        bkt_service=BKTService(
            DeterministicModel(), repository, clock=lambda: NOW
        ),
        clock=lambda: NOW,
        presentation_token_factory=lambda: token,
    )
    return controller, repository


def option(question, text):
    return next(choice.option_id for choice in question.options if choice.text == text)


def test_new_learner_initialization_and_initial_progress(tmp_path):
    controller, repository = build_controller(tmp_path / "app.sqlite3")

    assert controller.start_learner_session(" learner-1 ") == "learner-1"
    assert repository.learner_exists("learner-1")
    progress = controller.get_progress("learner-1", FOUNDATION)
    assert progress.mastery_probability == 0.0
    assert progress.attempt_count == 0
    assert progress.difficulty == "introductory"


def test_question_view_is_safe_and_contains_display_options(tmp_path):
    controller, _ = build_controller(tmp_path / "app.sqlite3")

    question = controller.recommend_question("learner-1", [])
    payload = question.model_dump()

    assert "correct_answer" not in repr(payload)
    assert payload["question_text"]
    assert all(set(choice) == {"option_id", "text"} for choice in payload["options"])
    assert question.recommendation_reason.endswith(".")


@pytest.mark.parametrize(
    ("answer_text", "expected_correct", "expected_mastery"),
    [("Correct", True, 0.8), ("Wrong A", False, 0.2)],
)
def test_server_side_scoring_and_submission_lifecycles(
    tmp_path, answer_text, expected_correct, expected_mastery
):
    controller, repository = build_controller(tmp_path / f"{answer_text}.sqlite3")
    question = controller.recommend_question("learner-1", [])

    result = controller.submit_answer(
        "learner-1", question.presentation_id, option(question, answer_text)
    )

    assert result.correct is expected_correct
    assert result.updated_mastery == expected_mastery
    assert result.previous_mastery == 0.0
    assert result.attempt_count == 1
    assert repository.get_attempt(result.attempt_id).correct is expected_correct


def test_duplicate_rerun_is_idempotent(tmp_path):
    controller, repository = build_controller(tmp_path / "app.sqlite3")
    question = controller.recommend_question("learner-1", [])
    selected = option(question, "Correct")

    first = controller.submit_answer("learner-1", question.presentation_id, selected)
    duplicate = controller.submit_answer("learner-1", question.presentation_id, selected)

    assert duplicate == first
    assert len(repository.list_attempts(learner_id="learner-1")) == 1
    assert len(repository.list_mastery(learner_id="learner-1")) == 1


def test_feedback_is_retained_across_reruns_and_next_resets_question_state(tmp_path):
    controller, _ = build_controller(tmp_path / "app.sqlite3")
    backing = {}
    session = get_session_state(backing)
    identify_learner(session, "learner-1")
    question = controller.recommend_question("learner-1", [])
    retain_question(session, question)
    selected = option(question, "Correct")
    generated_attempt_id = begin_submission(session, selected)
    result = controller.submit_answer("learner-1", question.presentation_id, selected)
    submission_succeeded(session, result)

    rerun = get_session_state(backing)
    assert rerun.feedback == result
    assert rerun.feedback_state == "visible"
    assert rerun.attempt_id == generated_attempt_id == result.attempt_id

    next_question(rerun)
    assert rerun.learner_id == "learner-1"
    assert rerun.seen_item_ids == [question.item_id]
    assert rerun.question is rerun.feedback is None
    assert rerun.presentation_id is rerun.item_id is rerun.attempt_id is None
    assert rerun.feedback_state == "hidden"


def test_learners_are_isolated(tmp_path):
    controller, repository = build_controller(tmp_path / "app.sqlite3")
    first = controller.recommend_question("learner-1", [])
    controller.submit_answer(
        "learner-1", first.presentation_id, option(first, "Correct")
    )

    assert controller.get_progress("learner-1", FOUNDATION).mastery_probability == 0.8
    assert controller.get_progress("learner-2", FOUNDATION).mastery_probability == 0.0
    assert repository.list_attempts(learner_id="learner-2") == []


def test_sqlite_persists_across_controller_reconstruction(tmp_path):
    database = tmp_path / "app.sqlite3"
    first_controller, first_repository = build_controller(database)
    question = first_controller.recommend_question("learner-1", [])
    result = first_controller.submit_answer(
        "learner-1", question.presentation_id, option(question, "Correct")
    )
    first_repository.close()

    reconstructed, repository = build_controller(database, token="token-2")
    assert reconstructed.get_progress("learner-1", FOUNDATION).attempt_count == 1
    assert repository.get_attempt(result.attempt_id) is not None


def test_mastery_update_unlocks_prerequisite_and_changes_recommendation(tmp_path):
    controller, _ = build_controller(tmp_path / "app.sqlite3")
    first = controller.recommend_question("learner-1", [])
    assert first.skill_id == FOUNDATION
    controller.submit_answer(
        "learner-1", first.presentation_id, option(first, "Correct")
    )

    second = controller.recommend_question("learner-1", [first.item_id])
    assert second.skill_id == DEPENDENT
    assert second.item_id != first.item_id


def test_invalid_option_and_missing_approved_item_are_readable(tmp_path):
    controller, _ = build_controller(
        tmp_path / "app.sqlite3", items=[item("foundation-item", FOUNDATION)]
    )
    question = controller.recommend_question("learner-1", [])
    with pytest.raises(InvalidOptionError, match="valid"):
        controller.submit_answer("learner-1", question.presentation_id, "forged")
    with pytest.raises(NoApprovedItemError):
        controller.recommend_question("learner-1", [question.item_id])


def test_full_taxonomy_with_sparse_approved_bank_recommends_available_item(tmp_path):
    controller, _ = build_controller(
        tmp_path / "app.sqlite3", items=[item("foundation-item", FOUNDATION)]
    )

    question = controller.recommend_question("learner-1", [])

    assert question.skill_id == FOUNDATION
    assert question.item_id == "foundation-item"


def test_empty_usable_inventory_has_friendly_application_error(tmp_path):
    controller, _ = build_controller(tmp_path / "app.sqlite3", items=[])

    with pytest.raises(NoApprovedItemError) as raised:
        controller.recommend_question("learner-1", [])

    assert raised.value.user_message == (
        "No approved question is available for the selected skill."
    )


def test_no_recommendation_is_mapped_to_application_error(tmp_path):
    controller, _ = build_controller(tmp_path / "app.sqlite3")

    class UnavailableService:
        config = controller.recommendation_service.config

        def recommend(self, request):
            raise RecommendationUnavailable("no_eligible_skill")

    controller.recommendation_service = UnavailableService()
    with pytest.raises(NoRecommendationError):
        controller.recommend_question("learner-1", [])
