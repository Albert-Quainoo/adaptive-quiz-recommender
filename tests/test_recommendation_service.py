from datetime import datetime, timezone

import pytest

from api.bank import BankItem
from api.schemas import QuizQuestion
from bkt.schemas import AttemptEvent, MasterySnapshot
from recommendation import (
    InMemoryRecommendationRepository,
    RecommendationPolicyConfig,
    RecommendationRequest,
    RecommendationService,
    RecommendationUnavailable,
)
from taxonomy.schemas import SkillDefinition


NOW = datetime(2026, 8, 8, tzinfo=timezone.utc)


def skill(skill_id: str, prerequisites: list[str] | None = None) -> SkillDefinition:
    return SkillDefinition(
        skill_id=skill_id,
        topic="Search",
        subtopic="Foundations",
        name=skill_id,
        learning_objective="Demonstrate the skill.",
        cognitive_process="understand",
        generation_strategy="hand_authored",
        prerequisite_skill_ids=prerequisites or [],
    )


def item(item_id: str, skill_id: str, difficulty: str) -> BankItem:
    return BankItem(
        item_id=item_id,
        skill_id=skill_id,
        provenance="hand_authored",
        question=QuizQuestion(
            question=f"Question {item_id}?",
            options=["A", "B", "C", "D"],
            correct_answer="A",
            explanation="A is correct.",
            concept="Search",
            difficulty=difficulty,
        ),
    )


def mastery(learner_id: str, skill_id: str, probability: float) -> MasterySnapshot:
    return MasterySnapshot(
        learner_id=learner_id,
        skill_id=skill_id,
        mastery_probability=probability,
        attempt_count=1,
        model_version="bkt-v3",
        source_attempt_id=f"snapshot-{learner_id}-{skill_id}",
        updated_at=NOW,
    )


def attempt(learner_id: str, item_id: str, order: int = 1) -> AttemptEvent:
    return AttemptEvent(
        attempt_id=f"attempt-{learner_id}-{item_id}",
        presentation_id=f"presentation-{learner_id}-{item_id}",
        learner_id=learner_id,
        item_id=item_id,
        skill_id="AI-SRC-01",
        selected_option_id="option-a",
        correct=True,
        attempt_order=order,
        occurred_at=NOW,
    )


def service(repository: InMemoryRecommendationRepository) -> RecommendationService:
    return RecommendationService(
        repository,
        model_version="bkt-v3",
        config=RecommendationPolicyConfig(policy_version="policy-v1"),
    )


def request(learner_id: str = "learner-1", **updates) -> RecommendationRequest:
    values = {
        "learner_id": learner_id,
        "available_skill_ids": ["AI-SRC-01"],
        "excluded_item_ids": [],
    }
    values.update(updates)
    return RecommendationRequest(**values)


def test_answered_item_is_avoided_when_an_alternative_exists():
    repository = InMemoryRecommendationRepository(
        skills=[skill("AI-SRC-01")],
        items=[
            item("item-a", "AI-SRC-01", "introductory"),
            item("item-b", "AI-SRC-01", "introductory"),
        ],
        attempts=[attempt("learner-1", "item-a")],
    )

    result = service(repository).recommend(request())

    assert result.item_id == "item-b"


def test_unavailable_difficulty_uses_documented_nearest_fallback():
    repository = InMemoryRecommendationRepository(
        skills=[skill("AI-SRC-01")],
        items=[item("item-a", "AI-SRC-01", "intermediate")],
    )

    result = service(repository).recommend(
        request(requested_difficulty="advanced")
    )

    assert result.difficulty == "intermediate"
    assert result.reason == "fallback_difficulty_used"


def test_one_learners_mastery_does_not_affect_another():
    first = skill("AI-SRC-01")
    second = skill("AI-SRC-02")
    repository = InMemoryRecommendationRepository(
        skills=[first, second],
        items=[
            item("item-a", first.skill_id, "introductory"),
            item("item-b", second.skill_id, "introductory"),
        ],
        mastery=[
            mastery("learner-1", first.skill_id, 0.90),
            mastery("learner-1", second.skill_id, 0.10),
        ],
    )
    recommender = service(repository)
    both_skills = [first.skill_id, second.skill_id]

    experienced = recommender.recommend(
        request("learner-1", available_skill_ids=both_skills)
    )
    unseen = recommender.recommend(
        request("learner-2", available_skill_ids=both_skills)
    )

    assert experienced.skill_id == second.skill_id
    assert unseen.skill_id == first.skill_id
    assert unseen.reason == "foundational_unseen_skill"


def test_reason_and_model_and_policy_versions_are_recorded():
    repository = InMemoryRecommendationRepository(
        skills=[skill("AI-SRC-01")],
        items=[item("item-a", "AI-SRC-01", "intermediate")],
        mastery=[mastery("learner-1", "AI-SRC-01", 0.50)],
    )

    result = service(repository).recommend(
        request(requested_difficulty="intermediate")
    )

    assert result.reason == "requested_difficulty_used"
    assert result.model_version == "bkt-v3"
    assert result.policy_version == "policy-v1"
    assert result.mastery_probability == 0.50


def test_excluded_items_are_never_recommended():
    repository = InMemoryRecommendationRepository(
        skills=[skill("AI-SRC-01")],
        items=[
            item("item-a", "AI-SRC-01", "introductory"),
            item("item-b", "AI-SRC-01", "introductory"),
        ],
    )

    result = service(repository).recommend(
        request(excluded_item_ids=["item-a"])
    )

    assert result.item_id == "item-b"


def test_no_eligible_item_raises_specific_unavailable_error():
    repository = InMemoryRecommendationRepository(
        skills=[skill("AI-SRC-01")],
        items=[item("item-a", "AI-SRC-01", "introductory")],
    )

    with pytest.raises(RecommendationUnavailable) as raised:
        service(repository).recommend(request(excluded_item_ids=["item-a"]))

    assert raised.value.reason == "no_eligible_item"
