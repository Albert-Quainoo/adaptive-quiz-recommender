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


def test_new_learner_falls_back_from_introductory_to_available_intermediate():
    repository = InMemoryRecommendationRepository(
        skills=[skill("AI-SRC-01")],
        items=[item("item-a", "AI-SRC-01", "intermediate")],
    )

    result = service(repository).recommend(request())

    assert result.difficulty == "intermediate"
    assert result.reason == "fallback_difficulty_used"


def test_taxonomy_skill_without_items_is_not_considered_available():
    without_item = skill("AI-SRC-01")
    with_item = skill("AI-SRC-02")
    repository = InMemoryRecommendationRepository(
        skills=[without_item, with_item],
        items=[item("item-b", with_item.skill_id, "introductory")],
    )

    result = service(repository).recommend(
        request(available_skill_ids=[without_item.skill_id, with_item.skill_id])
    )

    assert result.skill_id == with_item.skill_id


def test_exhausted_first_skill_falls_back_to_next_eligible_skill(caplog):
    first = skill("AI-SRC-01")
    second = skill("AI-SRC-02")
    repository = InMemoryRecommendationRepository(
        skills=[first, second],
        items=[
            item("item-a", first.skill_id, "introductory"),
            item("item-b", second.skill_id, "intermediate"),
        ],
    )

    with caplog.at_level("INFO", logger="recommendation.service"):
        result = service(repository).recommend(
            request(
                available_skill_ids=[first.skill_id, second.skill_id],
                excluded_item_ids=["item-a"],
            )
        )

    assert result.skill_id == second.skill_id
    assert result.item_id == "item-b"
    assert result.reason == "fallback_difficulty_used"
    assert "selected_skill=AI-SRC-01" in caplog.text
    assert "available_difficulties=[]" in caplog.text
    assert "fallback_result=unavailable" in caplog.text
    assert "selected_skill=AI-SRC-02" in caplog.text
    assert "requested_difficulty=introductory" in caplog.text
    assert "available_difficulties=['intermediate']" in caplog.text
    assert "fallback_result=fallback:intermediate" in caplog.text


def test_exhausted_foundation_below_mastery_unlocks_dependent_skill():
    foundation = skill("AI-SRC-01")
    dependent = skill("AI-SRC-02", [foundation.skill_id])
    repository = InMemoryRecommendationRepository(
        skills=[foundation, dependent],
        items=[
            item("item-a", foundation.skill_id, "introductory"),
            item("item-b", dependent.skill_id, "introductory"),
        ],
    )

    result = service(repository).recommend(
        request(
            available_skill_ids=[foundation.skill_id, dependent.skill_id],
            excluded_item_ids=["item-a"],
        )
    )

    assert result.skill_id == dependent.skill_id
    assert result.item_id == "item-b"
    assert result.reason == "prerequisite_exhausted_unlock"


def test_exhausted_foundation_below_mastery_with_no_further_content_is_unavailable():
    foundation = skill("AI-SRC-01")
    repository = InMemoryRecommendationRepository(
        skills=[foundation],
        items=[item("item-a", foundation.skill_id, "introductory")],
    )

    with pytest.raises(RecommendationUnavailable) as raised:
        service(repository).recommend(
            request(
                available_skill_ids=[foundation.skill_id],
                excluded_item_ids=["item-a"],
            )
        )

    assert raised.value.reason == "bank_exhausted_below_mastery"


def test_empty_usable_inventory_is_unavailable():
    repository = InMemoryRecommendationRepository(
        skills=[skill("AI-SRC-01")],
        items=[],
    )

    with pytest.raises(RecommendationUnavailable) as raised:
        service(repository).recommend(request())

    assert raised.value.reason == "bank_empty"


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

    assert raised.value.reason == "bank_exhausted_below_mastery"


def test_all_items_attempted_at_full_mastery_is_distinguished_from_exhaustion():
    repository = InMemoryRecommendationRepository(
        skills=[skill("AI-SRC-01")],
        items=[item("item-a", "AI-SRC-01", "introductory")],
        mastery=[mastery("learner-1", "AI-SRC-01", 0.90)],
    )

    with pytest.raises(RecommendationUnavailable) as raised:
        service(repository).recommend(request(excluded_item_ids=["item-a"]))

    assert raised.value.reason == "all_eligible_items_attempted"


def test_unlocked_missing_prerequisite_content_is_recorded_as_a_content_gap():
    foundation = skill("AI-FND-01")
    bridge = skill("AI-AGT-01", [foundation.skill_id])
    target = skill("AI-SRC-01", [bridge.skill_id])
    repository = InMemoryRecommendationRepository(
        skills=[foundation, bridge, target],
        items=[
            item("foundation-item", foundation.skill_id, "introductory"),
            item("target-item", target.skill_id, "intermediate"),
        ],
        mastery=[mastery("learner-1", foundation.skill_id, 0.80)],
    )

    with pytest.raises(RecommendationUnavailable) as raised:
        service(repository).recommend(
            request(
                available_skill_ids=[foundation.skill_id, target.skill_id]
            )
        )

    gap = raised.value.content_gap
    assert raised.value.reason == "missing_approved_content"
    assert gap.completed_skill_id == foundation.skill_id
    assert gap.newly_unlocked_skill_id == bridge.skill_id
    assert gap.missing_approved_content is True
    assert gap.current_mastery_probability == 0.80
    assert gap.prerequisite_mastery_threshold == 0.75
    assert repository.list_recommendations() == []
    assert repository.list_content_gaps()[0].model_dump(exclude={"content_gap_id", "created_at"}) == gap.model_dump()


def test_missing_bridge_content_remains_gated_below_mastery_threshold():
    foundation = skill("AI-FND-01")
    bridge = skill("AI-AGT-01", [foundation.skill_id])
    target = skill("AI-SRC-01", [bridge.skill_id])
    repository = InMemoryRecommendationRepository(
        skills=[foundation, bridge, target],
        items=[
            item("foundation-item", foundation.skill_id, "introductory"),
            item("target-item", target.skill_id, "intermediate"),
        ],
        mastery=[mastery("learner-1", foundation.skill_id, 0.74)],
    )

    result = service(repository).recommend(
        request(available_skill_ids=[foundation.skill_id, target.skill_id])
    )

    assert result.skill_id == foundation.skill_id
    assert repository.list_content_gaps() == []
