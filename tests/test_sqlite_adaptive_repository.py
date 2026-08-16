from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError, OperationalError

from api.bank import BankItem
from api.presentation import present_bank_item
from api.schemas import QuizQuestion
from bkt import (
    AttemptConflictError,
    AttemptEvent,
    BKTModelMetadata,
    BKTService,
    MasterySnapshot,
    SQLiteBKTRepository,
)
from recommendation import (
    ContentGapResult,
    RecommendationPolicyConfig,
    RecommendationRequest,
    RecommendationService,
    SQLiteRecommendationRepository,
)
from taxonomy.schemas import SkillDefinition


NOW = datetime(2026, 8, 9, 12, tzinfo=timezone.utc)


class DeterministicModel:
    model_version = "sqlite-model-v1"

    def update_mastery(self, attempts):
        return sum(int(attempt.correct) for attempt in attempts) / (
            len(attempts) + 1
        )


def _attempt(
    attempt_id: str,
    *,
    learner_id: str = "learner-1",
    skill_id: str = "AI-SQL-01",
    order: int = 1,
    occurred_at: datetime = NOW,
    correct: bool = True,
) -> AttemptEvent:
    return AttemptEvent(
        attempt_id=attempt_id,
        course_id="test-course",
        presentation_id=f"presentation-{attempt_id}",
        learner_id=learner_id,
        item_id=f"item-{skill_id}",
        skill_id=skill_id,
        selected_option_id="option-correct",
        correct=correct,
        attempt_order=order,
        occurred_at=occurred_at,
    )


def _snapshot(
    attempt: AttemptEvent,
    *,
    probability: float = 0.5,
    model_version: str = "sqlite-model-v1",
    updated_at: datetime = NOW,
) -> MasterySnapshot:
    return MasterySnapshot(
        learner_id=attempt.learner_id,
        course_id=attempt.course_id,
        skill_id=attempt.skill_id,
        mastery_probability=probability,
        attempt_count=attempt.attempt_order,
        source_attempt_id=attempt.attempt_id,
        model_version=model_version,
        updated_at=updated_at,
    )


def _skill(skill_id: str, prerequisites: list[str] | None = None):
    return SkillDefinition(
        skill_id=skill_id,
        topic="Persistence",
        subtopic="SQLite",
        name=skill_id,
        learning_objective="Answer the synthetic persistence question.",
        cognitive_process="understand",
        generation_strategy="hand_authored",
        prerequisite_skill_ids=prerequisites or [],
    )


def _item(skill_id: str):
    return BankItem(
        item_id=f"item-{skill_id}",
        skill_id=skill_id,
        provenance="hand_authored",
        question=QuizQuestion(
            question=f"Question for {skill_id}?",
            options=["Correct", "Wrong A", "Wrong B", "Wrong C"],
            correct_answer="Correct",
            explanation="Synthetic SQLite integration data.",
            concept="Persistence",
            difficulty="introductory",
        ),
    )


def _submission(attempt_id: str, *, order: int = 1):
    item = _item("AI-SQL-01")
    presentation = present_bank_item(
        item, learner_id="learner-1", attempt_id=attempt_id
    )
    selected = next(
        option
        for option in presentation.presented_options
        if option.value == item.question.correct_answer
    )
    event = AttemptEvent(
        attempt_id=attempt_id,
        course_id="test-course",
        presentation_id=presentation.presentation_id,
        learner_id="learner-1",
        item_id=item.item_id,
        skill_id=item.skill_id,
        selected_option_id=selected.option_id,
        correct=False,
        attempt_order=order,
        occurred_at=NOW + timedelta(minutes=order),
    )
    return event, item, presentation


def test_schema_is_explicit_foreign_keys_are_enabled_and_data_survives_reopen(
    tmp_path,
):
    database = tmp_path / "adaptive.sqlite3"
    repository = SQLiteBKTRepository(database, course_id="test-course")

    with repository._engine.connect() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).fetchone()[0] == 1
    with pytest.raises(OperationalError, match="no such table"):
        repository.list_attempts()

    repository.initialize_schema()
    event = _attempt("attempt-1")
    repository.save_attempt_and_mastery(event, _snapshot(event))
    metadata = BKTModelMetadata(
        model_version="sqlite-model-v1",
        course_id="test-course",
        fitted_at=NOW,
        training_attempt_count=10,
        skill_ids=["AI-SQL-01"],
    )
    repository.save_model_metadata(metadata)
    repository.close()

    reopened = SQLiteBKTRepository(database, course_id="test-course")
    with reopened._engine.connect() as connection:
        assert connection.execute(text("PRAGMA foreign_keys")).fetchone()[0] == 1
    loaded_attempt = reopened.get_attempt(event.attempt_id)
    loaded_mastery = reopened.get_mastery(event.learner_id, event.skill_id)

    assert loaded_attempt == event
    assert loaded_mastery == _snapshot(event)
    assert reopened.get_model_metadata("sqlite-model-v1") == metadata
    assert loaded_attempt.occurred_at.tzinfo is not None
    assert loaded_mastery.updated_at.tzinfo is not None
    with pytest.raises(ValidationError, match="frozen"):
        loaded_attempt.correct = False

    # mastery_snapshots(course_id, source_attempt_id) FKs
    # attempt_events(course_id, attempt_id) -- composite, so a snapshot
    # cannot reference an attempt from a different course either, not just
    # a nonexistent one.
    missing_attempt = _attempt("missing")
    with pytest.raises(IntegrityError, match="FOREIGN KEY"):
        reopened.save_mastery(_snapshot(missing_attempt))
    reopened.close()


def test_sqlite_processing_is_idempotent_and_conflicting_attempts_are_rejected(
    tmp_path,
):
    repository = SQLiteBKTRepository(tmp_path / "idempotency.sqlite3", course_id="test-course")
    repository.initialize_schema()
    service = BKTService(DeterministicModel(), repository, clock=lambda: NOW)
    event, item, presentation = _submission("attempt-1")

    first = service.process_attempt(event, item=item, presentation=presentation)
    duplicate = service.process_attempt(event, item=item, presentation=presentation)

    assert duplicate == first
    assert repository.get_attempt(event.attempt_id).correct is True
    assert len(repository.list_attempts()) == 1
    assert len(repository.list_mastery()) == 1

    conflicting = event.model_copy(update={"attempt_order": 2})
    with pytest.raises(AttemptConflictError, match="already used"):
        service.process_attempt(
            conflicting,
            item=item,
            presentation=presentation,
        )

    assert len(repository.list_attempts()) == 1
    assert len(repository.list_mastery()) == 1
    repository.close()


def test_atomic_failure_rolls_back_both_attempt_and_mastery(tmp_path, monkeypatch):
    repository = SQLiteBKTRepository(tmp_path / "rollback.sqlite3", course_id="test-course")
    repository.initialize_schema()
    event = _attempt("rolled-back")
    snapshot = _snapshot(event)

    def fail_after_attempt_insert(_connection, _snapshot):
        raise RuntimeError("synthetic mastery write failure")

    monkeypatch.setattr(repository, "_insert_mastery", fail_after_attempt_insert)

    with pytest.raises(RuntimeError, match="synthetic mastery write failure"):
        repository.save_attempt_and_mastery(event, snapshot)

    assert repository.get_attempt(event.attempt_id) is None
    assert repository.list_mastery() == []
    repository.close()


def test_attempt_order_latest_mastery_and_learner_skill_isolation(tmp_path):
    repository = SQLiteBKTRepository(tmp_path / "ordering.sqlite3", course_id="test-course")
    repository.initialize_schema()
    attempts = [
        _attempt("learner-1-skill-1-order-2", order=2),
        _attempt("learner-2-skill-1", learner_id="learner-2"),
        _attempt("learner-1-skill-2", skill_id="AI-SQL-02"),
        _attempt(
            "learner-1-skill-1-order-1",
            order=1,
            occurred_at=NOW - timedelta(seconds=1),
        ),
    ]
    for event in attempts:
        repository.save_attempt(event)

    learner_skill_history = repository.list_attempts(
        learner_id="learner-1", skill_id="AI-SQL-01"
    )
    assert [event.attempt_id for event in learner_skill_history] == [
        "learner-1-skill-1-order-1",
        "learner-1-skill-1-order-2",
    ]
    assert len(repository.list_attempts(learner_id="learner-1")) == 3
    assert len(repository.list_attempts(learner_id="learner-2")) == 1

    first, second = learner_skill_history
    repository.save_mastery(_snapshot(first, probability=0.4, updated_at=NOW))
    repository.save_mastery(_snapshot(second, probability=0.6, updated_at=NOW))
    assert repository.get_mastery("learner-1", "AI-SQL-01").source_attempt_id == (
        second.attempt_id
    )

    repository.save_mastery(
        _snapshot(
            first,
            probability=0.8,
            model_version="sqlite-model-v2",
            updated_at=NOW + timedelta(seconds=1),
        )
    )
    latest = repository.get_mastery("learner-1", "AI-SQL-01")
    assert latest.source_attempt_id == first.attempt_id
    assert latest.model_version == "sqlite-model-v2"

    other_learner_attempt = next(
        event for event in attempts if event.learner_id == "learner-2"
    )
    other_skill_attempt = next(
        event for event in attempts if event.skill_id == "AI-SQL-02"
    )
    repository.save_mastery(_snapshot(other_learner_attempt, probability=0.2))
    repository.save_mastery(_snapshot(other_skill_attempt, probability=0.3))

    assert repository.get_mastery(
        "learner-2", "AI-SQL-01"
    ).mastery_probability == 0.2
    assert repository.get_mastery(
        "learner-1", "AI-SQL-02"
    ).mastery_probability == 0.3
    assert repository.get_mastery(
        "learner-1", "AI-SQL-01"
    ).mastery_probability == 0.8
    repository.close()


def test_replay_reconstructs_mastery_from_ordered_sqlite_attempts(tmp_path):
    repository = SQLiteBKTRepository(tmp_path / "replay.sqlite3", course_id="test-course")
    repository.initialize_schema()
    service = BKTService(DeterministicModel(), repository, clock=lambda: NOW)
    submissions = [_submission(f"attempt-{order}", order=order) for order in range(1, 4)]
    online = [
        service.process_attempt(event, item=item, presentation=presentation)
        for event, item, presentation in submissions
    ]

    replayed = service.replay(learner_id="learner-1")

    assert [snapshot.mastery_probability for snapshot in replayed] == pytest.approx(
        [snapshot.mastery_probability for snapshot in online]
    )
    assert replayed[-1].mastery_probability == pytest.approx(
        online[-1].mastery_probability
    )
    assert len(repository.list_attempts()) == 3
    assert len(repository.list_mastery()) == 3
    repository.close()


def test_recommendations_are_persisted_with_versions_and_learners_are_isolated(
    tmp_path,
):
    foundational = _skill("AI-SQL-01")
    dependent = _skill("AI-SQL-02", [foundational.skill_id])
    repository = SQLiteRecommendationRepository(
        tmp_path / "recommendations.sqlite3",
        course_id="test-course",
        skills=[foundational, dependent],
        items=[_item(foundational.skill_id), _item(dependent.skill_id)],
    )
    repository.initialize_schema()
    foundation_attempt = _attempt("mastered-foundation")
    repository.save_attempt_and_mastery(
        foundation_attempt,
        _snapshot(foundation_attempt, probability=0.8),
    )
    service = RecommendationService(
        repository,
        course_id="test-course",
        model_version="sqlite-model-v1",
        config=RecommendationPolicyConfig(policy_version="sqlite-policy-v1"),
    )
    skill_ids = [foundational.skill_id, dependent.skill_id]

    experienced = service.recommend(
        RecommendationRequest(
            learner_id="learner-1", available_skill_ids=skill_ids
        )
    )
    unseen = service.recommend(
        RecommendationRequest(
            learner_id="learner-2", available_skill_ids=skill_ids
        )
    )
    repository.close()

    reopened = SQLiteRecommendationRepository(
        tmp_path / "recommendations.sqlite3",
        course_id="test-course",
        skills=[foundational, dependent],
        items=[_item(foundational.skill_id), _item(dependent.skill_id)],
    )
    experienced_events = reopened.list_recommendations(learner_id="learner-1")
    unseen_events = reopened.list_recommendations(learner_id="learner-2")

    assert experienced.skill_id == dependent.skill_id
    assert unseen.skill_id == foundational.skill_id
    assert len(experienced_events) == len(unseen_events) == 1
    assert experienced_events[0].reason == "lowest_mastery_eligible_skill"
    assert experienced_events[0].model_version == "sqlite-model-v1"
    assert experienced_events[0].policy_version == "sqlite-policy-v1"
    assert experienced_events[0].created_at.tzinfo is not None
    assert unseen_events[0].reason == "foundational_unseen_skill"
    assert reopened.list_attempts(learner_id="learner-2") == []
    assert reopened.list_latest_mastery("learner-2") == []
    reopened.close()


def test_content_gap_result_is_persisted_with_mastery_and_threshold(tmp_path):
    database = tmp_path / "content-gaps.sqlite3"
    repository = SQLiteRecommendationRepository(
        database,
        course_id="test-course",
        skills=[_skill("AI-SQL-01")],
        items=[_item("AI-SQL-01")],
    )
    repository.initialize_schema()
    result = ContentGapResult(
        learner_id="learner-1",
        course_id="test-course",
        completed_skill_id="AI-SQL-01",
        completed_skill_name="SQL foundation",
        newly_unlocked_skill_id="AI-SQL-02",
        newly_unlocked_skill_name="SQL application",
        current_mastery_probability=0.8,
        prerequisite_mastery_threshold=0.75,
        model_version="sqlite-model-v1",
        policy_version="sqlite-policy-v1",
    )
    repository.save_content_gap(result)
    repository.close()

    reopened = SQLiteRecommendationRepository(
        database,
        course_id="test-course",
        skills=[_skill("AI-SQL-01")],
        items=[_item("AI-SQL-01")],
    )
    events = reopened.list_content_gaps(learner_id="learner-1")

    assert len(events) == 1
    assert events[0].completed_skill_id == "AI-SQL-01"
    assert events[0].newly_unlocked_skill_id == "AI-SQL-02"
    assert events[0].missing_approved_content is True
    assert events[0].current_mastery_probability == 0.8
    assert events[0].prerequisite_mastery_threshold == 0.75
    assert events[0].created_at.tzinfo is not None
    reopened.close()
