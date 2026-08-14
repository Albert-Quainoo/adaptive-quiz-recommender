from datetime import datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from api.bank import BankItem
from api.presentation import present_bank_item
from api.schemas import QuizQuestion
from bkt import AttemptEvent, BKTModel, BKTService, InMemoryBKTRepository
from recommendation import (
    InMemoryRecommendationRepository,
    RecommendationPolicyConfig,
    RecommendationRequest,
    RecommendationService,
    RecommendationUnavailable,
)
from taxonomy.schemas import SkillDefinition


NOW = datetime(2026, 8, 9, tzinfo=timezone.utc)
LEARNER_ID = "adaptive-learner"
OTHER_LEARNER_ID = "unseen-learner"
FOUNDATIONAL_SKILL_ID = "AI-ALG-01"
DEPENDENT_SKILL_ID = "AI-ALG-02"
MODEL_VERSION = "adaptive-loop-synthetic-v1"
POLICY_VERSION = "adaptive-loop-policy-v1"
MODEL_SEED = 17


def _skill(skill_id: str, *, prerequisites: list[str] | None = None) -> SkillDefinition:
    return SkillDefinition(
        skill_id=skill_id,
        topic="Algorithms",
        subtopic="Foundations",
        name=skill_id,
        learning_objective="Answer a deterministic synthetic question.",
        cognitive_process="understand",
        generation_strategy="hand_authored",
        prerequisite_skill_ids=prerequisites or [],
    )


def _item(item_id: str, skill_id: str) -> BankItem:
    return BankItem(
        item_id=item_id,
        skill_id=skill_id,
        provenance="hand_authored",
        question=QuizQuestion(
            question=f"Synthetic question for {skill_id}?",
            options=["Correct", "Wrong A", "Wrong B", "Wrong C"],
            correct_answer="Correct",
            explanation="This is deterministic synthetic test data.",
            concept=skill_id,
            difficulty="introductory",
        ),
    )


def _training_attempts() -> list[AttemptEvent]:
    sequences = (
        (False, False, False, True, False, True, True, True, True, True),
        (False, False, True, False, True, True, True, True, True, True),
    )
    return [
        AttemptEvent(
            attempt_id=f"training-{skill_id}-{learner_index}-{attempt_order}",
            presentation_id=(
                f"training-presentation-{skill_id}-{learner_index}-{attempt_order}"
            ),
            learner_id=f"training-learner-{learner_index}",
            item_id=f"training-item-{skill_id}",
            skill_id=skill_id,
            selected_option_id="training-option",
            correct=correct,
            attempt_order=attempt_order,
            occurred_at=NOW + timedelta(seconds=attempt_order),
        )
        for skill_id in (FOUNDATIONAL_SKILL_ID, DEPENDENT_SKILL_ID)
        for learner_index in range(8)
        for attempt_order, correct in enumerate(
            sequences[learner_index % len(sequences)], start=1
        )
    ]


def test_end_to_end_adaptive_loop_unlocks_dependent_skill_after_mastery():
    skills = [
        _skill(FOUNDATIONAL_SKILL_ID),
        _skill(DEPENDENT_SKILL_ID, prerequisites=[FOUNDATIONAL_SKILL_ID]),
    ]
    items = [
        _item("item-foundational", FOUNDATIONAL_SKILL_ID),
        _item("item-dependent", DEPENDENT_SKILL_ID),
    ]
    item_by_id = {item.item_id: item for item in items}
    bkt_repository = InMemoryBKTRepository()
    bkt_service = BKTService(
        BKTModel(
            model_version=MODEL_VERSION,
            seed=MODEL_SEED,
            num_fits=1,
        ),
        bkt_repository,
        clock=lambda: NOW,
    )
    metadata = bkt_service.fit(_training_attempts())
    policy = RecommendationPolicyConfig(
        prerequisite_mastery_threshold=0.75,
        policy_version=POLICY_VERSION,
    )

    def recommend(learner_id: str, available_skill_ids: list[str]):
        # The recommendation repository is the in-memory read view of the current
        # immutable attempt log and versioned mastery history.
        repository = InMemoryRecommendationRepository(
            skills=skills,
            items=items,
            mastery=bkt_repository.list_mastery(),
            attempts=bkt_repository.list_attempts(),
        )
        return RecommendationService(
            repository,
            model_version=MODEL_VERSION,
            config=policy,
        ).recommend(
            RecommendationRequest(
                learner_id=learner_id,
                available_skill_ids=available_skill_ids,
            )
        )

    all_skill_ids = [FOUNDATIONAL_SKILL_ID, DEPENDENT_SKILL_ID]
    first_recommendation = recommend(LEARNER_ID, all_skill_ids)

    assert first_recommendation.skill_id == FOUNDATIONAL_SKILL_ID
    assert first_recommendation.reason == "foundational_unseen_skill"
    with pytest.raises(RecommendationUnavailable, match="no_prerequisite_eligible_skill"):
        recommend(LEARNER_ID, [DEPENDENT_SKILL_ID])

    online_snapshots = []
    for attempt_order in range(1, 11):
        attempt_id = f"online-attempt-{attempt_order:02d}"
        item = item_by_id[first_recommendation.item_id]
        presentation = present_bank_item(
            item,
            learner_id=LEARNER_ID,
            attempt_id=attempt_id,
        )
        correct_option = next(
            option
            for option in presentation.presented_options
            if option.value == item.question.correct_answer
        )
        submitted_event = AttemptEvent(
            attempt_id=attempt_id,
            presentation_id=presentation.presentation_id,
            learner_id=LEARNER_ID,
            item_id=item.item_id,
            skill_id=FOUNDATIONAL_SKILL_ID,
            selected_option_id=correct_option.option_id,
            correct=False,  # The service must score the selected option itself.
            attempt_order=attempt_order,
            occurred_at=NOW + timedelta(minutes=attempt_order),
        )

        snapshot_count = len(bkt_repository.list_mastery(learner_id=LEARNER_ID))
        snapshot = bkt_service.process_attempt(
            submitted_event,
            item=item,
            presentation=presentation,
        )
        online_snapshots.append(snapshot)

        stored_attempt = bkt_repository.get_attempt(attempt_id)
        assert stored_attempt is not None
        assert stored_attempt.correct is True
        assert len(bkt_repository.list_attempts(learner_id=LEARNER_ID)) == attempt_order
        assert len(bkt_repository.list_mastery(learner_id=LEARNER_ID)) == (
            snapshot_count + 1
        )
        assert snapshot.attempt_count == attempt_order
        assert snapshot.source_attempt_id == attempt_id
        assert snapshot.model_version == MODEL_VERSION

        with pytest.raises(ValidationError, match="frozen"):
            stored_attempt.correct = False

        if snapshot.mastery_probability >= policy.prerequisite_mastery_threshold:
            break

    assert online_snapshots[-1].mastery_probability >= (
        policy.prerequisite_mastery_threshold
    )
    assert [attempt.attempt_order for attempt in bkt_repository.list_attempts()] == list(
        range(1, len(online_snapshots) + 1)
    )

    second_recommendation = recommend(LEARNER_ID, all_skill_ids)
    dependent_only_recommendation = recommend(LEARNER_ID, [DEPENDENT_SKILL_ID])

    assert second_recommendation.skill_id == DEPENDENT_SKILL_ID
    assert dependent_only_recommendation.skill_id == DEPENDENT_SKILL_ID
    assert second_recommendation.item_id != first_recommendation.item_id
    assert second_recommendation.reason == "lowest_mastery_eligible_skill"
    for recommendation in (first_recommendation, second_recommendation):
        assert recommendation.reason
        assert recommendation.model_version == MODEL_VERSION
        assert recommendation.policy_version == POLICY_VERSION
    assert metadata.model_version == MODEL_VERSION

    final_submission_count = len(bkt_repository.list_attempts(learner_id=LEARNER_ID))
    final_snapshot_count = len(bkt_repository.list_mastery(learner_id=LEARNER_ID))
    duplicate = bkt_service.process_attempt(
        submitted_event,
        item=item,
        presentation=presentation,
    )

    assert duplicate == online_snapshots[-1]
    assert len(bkt_repository.list_attempts(learner_id=LEARNER_ID)) == (
        final_submission_count
    )
    assert len(bkt_repository.list_mastery(learner_id=LEARNER_ID)) == (
        final_snapshot_count
    )

    replayed = bkt_service.replay(learner_id=LEARNER_ID)

    assert [snapshot.mastery_probability for snapshot in replayed] == pytest.approx(
        [snapshot.mastery_probability for snapshot in online_snapshots]
    )
    assert replayed[-1].mastery_probability == pytest.approx(
        online_snapshots[-1].mastery_probability
    )
    assert len(bkt_repository.list_attempts(learner_id=LEARNER_ID)) == (
        final_submission_count
    )

    other_learner_recommendation = recommend(OTHER_LEARNER_ID, all_skill_ids)

    assert bkt_repository.list_attempts(learner_id=OTHER_LEARNER_ID) == []
    assert bkt_repository.list_mastery(learner_id=OTHER_LEARNER_ID) == []
    assert other_learner_recommendation.skill_id == FOUNDATIONAL_SKILL_ID
    assert other_learner_recommendation.reason == "foundational_unseen_skill"
