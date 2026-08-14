"""PostgreSQL integration coverage for the adaptive-loop repositories.

Mirrors the scenarios in test_sqlite_adaptive_repository.py, but against a
real PostgreSQL database via QUIZ_TEST_POSTGRES_URL -- this is what proves
the SQLAlchemy abstraction in database.py actually works on the production
dialect, not just SQLite. Skipped locally unless that env var is set;
GitHub Actions always sets it against the postgres service container.
"""

import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import text

from api.bank import BankItem
from api.schemas import QuizQuestion
from bkt import AttemptConflictError, AttemptEvent, MasterySnapshot, SQLiteBKTRepository
from recommendation import RecommendationPolicyConfig, RecommendationRequest, RecommendationService
from recommendation.sqlite_repository import SQLiteRecommendationRepository
from taxonomy.schemas import SkillDefinition

DSN_ENV_VAR = "QUIZ_TEST_POSTGRES_URL"

pytestmark = pytest.mark.skipif(
    not os.getenv(DSN_ENV_VAR),
    reason=f"set {DSN_ENV_VAR} to a PostgreSQL DSN to run these integration tests",
)

NOW = datetime(2026, 8, 9, 12, tzinfo=timezone.utc)


def _dsn() -> str:
    return os.environ[DSN_ENV_VAR]


@pytest.fixture(autouse=True)
def _clean_database():
    """Each test gets a schema reset -- one shared Postgres instance/database
    is reused across tests rather than creating a new database per test."""
    engine_owner = SQLiteBKTRepository(_dsn())
    with engine_owner._engine.begin() as connection:
        connection.execute(
            text(
                "DROP TABLE IF EXISTS question_presentations, learner_sessions, "
                "content_gap_events, recommendation_events, mastery_snapshots, "
                "bkt_model_metadata, attempt_events CASCADE"
            )
        )
    engine_owner.close()
    yield


def _attempt(attempt_id: str, *, order: int = 1, correct: bool = True) -> AttemptEvent:
    return AttemptEvent(
        attempt_id=attempt_id,
        presentation_id=f"presentation-{attempt_id}",
        learner_id="learner-1",
        item_id="item-AI-PG-01",
        skill_id="AI-PG-01",
        selected_option_id="option-correct",
        correct=correct,
        attempt_order=order,
        occurred_at=NOW,
    )


def _snapshot(attempt: AttemptEvent, *, probability: float = 0.5) -> MasterySnapshot:
    return MasterySnapshot(
        learner_id=attempt.learner_id,
        skill_id=attempt.skill_id,
        mastery_probability=probability,
        attempt_count=attempt.attempt_order,
        source_attempt_id=attempt.attempt_id,
        model_version="postgres-model-v1",
        updated_at=NOW,
    )


def _skill(skill_id: str, prerequisites: list[str] | None = None) -> SkillDefinition:
    return SkillDefinition(
        skill_id=skill_id,
        topic="Persistence",
        subtopic="PostgreSQL",
        name=skill_id,
        learning_objective="Answer the synthetic persistence question.",
        cognitive_process="understand",
        generation_strategy="hand_authored",
        prerequisite_skill_ids=prerequisites or [],
    )


def _item(skill_id: str) -> BankItem:
    return BankItem(
        item_id=f"item-{skill_id}",
        skill_id=skill_id,
        provenance="hand_authored",
        question=QuizQuestion(
            question=f"Question for {skill_id}?",
            options=["Correct", "Wrong A", "Wrong B", "Wrong C"],
            correct_answer="Correct",
            explanation="Synthetic PostgreSQL integration data.",
            concept="Persistence",
            difficulty="introductory",
        ),
    )


EXPECTED_TABLES = {
    "attempt_events",
    "mastery_snapshots",
    "recommendation_events",
    "content_gap_events",
    "bkt_model_metadata",
    "learner_sessions",
    "question_presentations",
}


def test_schema_creation_succeeds_on_an_empty_database_and_repeated_init_is_safe():
    repository = SQLiteBKTRepository(_dsn())
    repository.initialize_schema()
    repository.initialize_schema()  # idempotent, must not raise
    with repository._engine.connect() as connection:
        tables = {
            row[0]
            for row in connection.execute(
                text("SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'")
            )
        }
    assert EXPECTED_TABLES <= tables
    repository.close()


def test_a_learner_can_be_created_and_retrieved():
    repository = SQLiteRecommendationRepository(_dsn(), skills=[], items=[])
    repository.initialize_schema()
    repository.save_learner("learner-pg-1", created_at=NOW)
    assert repository.learner_exists("learner-pg-1") is True
    assert repository.learner_exists("learner-pg-unknown") is False
    # idempotent -- an app/controller.py re-activation must not raise
    repository.save_learner("learner-pg-1", created_at=NOW)
    repository.close()


def test_a_quiz_session_survives_repository_reconstruction():
    dsn = _dsn()
    repository = SQLiteRecommendationRepository(dsn, skills=[], items=[])
    repository.initialize_schema()
    repository.save_learner("learner-1", created_at=NOW)
    repository.save_presentation(
        presentation_id="presentation-1",
        learner_id="learner-1",
        item_id="item-AI-PG-01",
        presentation_seed=42,
        recommendation_reason="lowest_mastery_eligible_skill",
        created_at=NOW,
    )
    repository.close()

    reopened = SQLiteRecommendationRepository(dsn, skills=[], items=[])
    row = reopened.get_presentation("presentation-1")
    assert row["learner_id"] == "learner-1"
    assert row["item_id"] == "item-AI-PG-01"
    reopened.close()


def test_submitted_responses_persist():
    repository = SQLiteBKTRepository(_dsn())
    repository.initialize_schema()
    attempt = _attempt("pg-response-1")
    repository.save_attempt_and_mastery(attempt, _snapshot(attempt))

    reopened = SQLiteBKTRepository(_dsn())
    assert reopened.get_attempt("pg-response-1") == attempt
    assert reopened.list_attempts(learner_id="learner-1") == [attempt]
    reopened.close()


def test_bkt_mastery_persists():
    repository = SQLiteBKTRepository(_dsn())
    repository.initialize_schema()
    attempt = _attempt("pg-mastery-1")
    repository.save_attempt_and_mastery(attempt, _snapshot(attempt, probability=0.73))

    reopened = SQLiteBKTRepository(_dsn())
    mastery = reopened.get_mastery("learner-1", "AI-PG-01")
    assert mastery.mastery_probability == pytest.approx(0.73)
    reopened.close()


def test_next_question_recommendation_works():
    foundational = _skill("AI-PG-01")
    dependent = _skill("AI-PG-02", [foundational.skill_id])
    repository = SQLiteRecommendationRepository(
        _dsn(),
        skills=[foundational, dependent],
        items=[_item(foundational.skill_id), _item(dependent.skill_id)],
    )
    repository.initialize_schema()
    mastered = _attempt("pg-mastered-foundation")
    repository.save_attempt_and_mastery(mastered, _snapshot(mastered, probability=0.8))

    service = RecommendationService(
        repository,
        model_version="postgres-model-v1",
        config=RecommendationPolicyConfig(policy_version="postgres-policy-v1"),
    )
    recommendation = service.recommend(
        RecommendationRequest(
            learner_id="learner-1",
            available_skill_ids=[foundational.skill_id, dependent.skill_id],
        )
    )
    assert recommendation.skill_id == dependent.skill_id

    events = repository.list_recommendations(learner_id="learner-1")
    assert len(events) == 1
    assert events[0].policy_version == "postgres-policy-v1"
    repository.close()


def test_duplicate_selection_remains_prevented():
    """The UNIQUE (learner_id, skill_id, attempt_order) constraint is the
    DB-level guard against crediting the same slot twice."""
    repository = SQLiteBKTRepository(_dsn())
    repository.initialize_schema()
    first = _attempt("pg-dup-1", order=1)
    repository.save_attempt_and_mastery(first, _snapshot(first))

    conflicting = _attempt("pg-dup-2", order=1)  # same learner/skill/order, different id
    with pytest.raises(AttemptConflictError):
        repository.save_attempt_and_mastery(conflicting, _snapshot(conflicting))

    # Replaying the exact same attempt_id is idempotent, not a conflict.
    replay = repository.save_attempt_and_mastery(first, _snapshot(first))
    assert replay == _snapshot(first)
    repository.close()


def test_data_written_by_one_process_is_readable_by_a_separate_process():
    """The scenario SQLite file-locking on one process can't distinguish
    from a same-process reopen: two independently constructed repository
    instances against the same Postgres DSN, standing in for the Streamlit
    process and the replenishment worker CLI process."""
    dsn = _dsn()
    writer = SQLiteBKTRepository(dsn)
    writer.initialize_schema()
    attempt = _attempt("pg-cross-process-1")
    writer.save_attempt_and_mastery(attempt, _snapshot(attempt))

    reader = SQLiteBKTRepository(dsn)
    assert reader.get_attempt("pg-cross-process-1") == attempt
    assert reader.get_mastery("learner-1", "AI-PG-01") == _snapshot(attempt)
    writer.close()
    reader.close()
