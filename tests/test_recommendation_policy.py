import pytest

from api.bank import BankItem
from api.schemas import QuizQuestion
from recommendation.policy import (
    RecommendationPolicyConfig,
    difficulty_for_mastery,
    select_item,
    select_skill,
)
from taxonomy.schemas import SkillDefinition


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


def test_locked_skill_is_not_selected_until_its_prerequisite_is_mastered():
    foundation = skill("AI-SRC-01")
    dependent = skill("AI-SRC-02", [foundation.skill_id])
    config = RecommendationPolicyConfig(prerequisite_mastery_threshold=0.75)

    locked = select_skill(
        [foundation, dependent],
        {foundation.skill_id, dependent.skill_id},
        {foundation.skill_id: 0.74, dependent.skill_id: 0.05},
        config,
    )
    unlocked = select_skill(
        [foundation, dependent],
        {foundation.skill_id, dependent.skill_id},
        {foundation.skill_id: 0.75, dependent.skill_id: 0.05},
        config,
    )

    assert locked.skill.skill_id == foundation.skill_id
    assert unlocked.skill.skill_id == dependent.skill_id


def test_exhausted_prerequisite_unlocks_dependent_skill_with_a_distinct_reason():
    foundation = skill("AI-SRC-01")
    dependent = skill("AI-SRC-02", [foundation.skill_id])
    config = RecommendationPolicyConfig(prerequisite_mastery_threshold=0.75)

    still_locked = select_skill(
        [foundation, dependent],
        {dependent.skill_id},
        {foundation.skill_id: 0.30},
        config,
    )
    bypassed = select_skill(
        [foundation, dependent],
        {dependent.skill_id},
        {foundation.skill_id: 0.30},
        config,
        {foundation.skill_id},
    )

    assert still_locked is None
    assert bypassed.skill.skill_id == dependent.skill_id
    assert bypassed.reason == "prerequisite_exhausted_unlock"


def test_missing_prerequisite_mastery_uses_configured_initial_mastery():
    dependent = skill("AI-SRC-02", ["AI-SRC-01"])

    selection = select_skill(
        [dependent],
        {dependent.skill_id},
        {},
        RecommendationPolicyConfig(
            prerequisite_mastery_threshold=0.75,
            initial_mastery_probability=0.20,
        ),
    )

    assert selection is None


def test_lowest_mastery_eligible_skill_is_selected():
    first = skill("AI-SRC-01")
    second = skill("AI-SRC-02")

    selection = select_skill(
        [first, second],
        {first.skill_id, second.skill_id},
        {first.skill_id: 0.60, second.skill_id: 0.30},
        RecommendationPolicyConfig(),
    )

    assert selection.skill.skill_id == second.skill_id
    assert selection.reason == "lowest_mastery_eligible_skill"


def test_cold_start_selects_foundational_skill_and_ties_use_taxonomy_order():
    first_in_taxonomy = skill("AI-SRC-09")
    second_in_taxonomy = skill("AI-SRC-01")

    selection = select_skill(
        [first_in_taxonomy, second_in_taxonomy],
        {first_in_taxonomy.skill_id, second_in_taxonomy.skill_id},
        {},
        RecommendationPolicyConfig(),
    )

    assert selection.skill.skill_id == first_in_taxonomy.skill_id
    assert selection.reason == "foundational_unseen_skill"


def test_item_ties_use_item_id_and_fallback_prefers_nearest_easier_difficulty():
    items = [
        item("item-b", "AI-SRC-01", "introductory"),
        item("item-a", "AI-SRC-01", "introductory"),
        item("item-c", "AI-SRC-01", "advanced"),
    ]

    selection = select_item(
        items,
        skill_id="AI-SRC-01",
        desired_difficulty="intermediate",
        excluded_item_ids=set(),
        last_answered_item_id=None,
    )

    assert selection.item.item_id == "item-a"
    assert selection.reason == "fallback_difficulty_used"


def test_item_ties_are_randomized_per_learner_but_stable_within_a_learner():
    items = [item(f"item-{letter}", "AI-SRC-01", "introductory") for letter in "abcdefgh"]

    def pick(learner_id: str) -> str:
        return select_item(
            items,
            skill_id="AI-SRC-01",
            desired_difficulty="introductory",
            excluded_item_ids=set(),
            last_answered_item_id=None,
            learner_id=learner_id,
        ).item.item_id

    first_call = pick("learner-1")
    assert pick("learner-1") == first_call  # stable across repeated calls

    other_learner_picks = {pick(f"learner-{n}") for n in range(2, 12)}
    assert other_learner_picks != {first_call}  # not everyone gets the same item


@pytest.mark.parametrize(
    ("preferred", "available", "expected"),
    [
        ("introductory", ["intermediate", "advanced"], "intermediate"),
        ("intermediate", ["introductory", "advanced"], "introductory"),
        ("advanced", ["introductory", "intermediate"], "intermediate"),
    ],
)
def test_difficulty_fallback_order_is_deterministic(preferred, available, expected):
    selection = select_item(
        [
            item(f"item-{difficulty}", "AI-SRC-01", difficulty)
            for difficulty in reversed(available)
        ],
        skill_id="AI-SRC-01",
        desired_difficulty=preferred,
        excluded_item_ids=set(),
        last_answered_item_id=None,
    )

    assert selection.item.question.difficulty == expected
    assert selection.reason == "fallback_difficulty_used"


def test_default_difficulty_thresholds_are_transparent_and_configurable():
    config = RecommendationPolicyConfig()

    assert difficulty_for_mastery(0.39, config) == "introductory"
    assert difficulty_for_mastery(0.40, config) == "intermediate"
    assert difficulty_for_mastery(0.74, config) == "intermediate"
    assert difficulty_for_mastery(0.75, config) == "advanced"


def test_restrict_to_weak_narrows_to_below_threshold_skills():
    strong = skill("AI-SRC-01")
    weak = skill("AI-SRC-02")
    config = RecommendationPolicyConfig()

    selection = select_skill(
        [strong, weak],
        {strong.skill_id, weak.skill_id},
        {strong.skill_id: 0.90, weak.skill_id: 0.10},
        config,
        restrict_to_weak=True,
    )

    assert selection.skill.skill_id == weak.skill_id


def test_restrict_to_weak_falls_back_to_normal_selection_when_nothing_is_weak():
    """A "focus on weak areas" round-start choice must never hard-fail just
    because the learner currently has nothing below the weak threshold."""
    first = skill("AI-SRC-01")
    second = skill("AI-SRC-02")
    config = RecommendationPolicyConfig()

    selection = select_skill(
        [first, second],
        {first.skill_id, second.skill_id},
        {first.skill_id: 0.90, second.skill_id: 0.85},
        config,
        restrict_to_weak=True,
    )

    assert selection is not None
    assert selection.skill.skill_id == second.skill_id  # still the lower-mastery one


def test_select_item_prioritizes_lifetime_unseen_items_over_attempted_ones():
    attempted = item("item-attempted", "AI-SRC-01", "introductory")
    unseen = item("item-unseen", "AI-SRC-01", "introductory")

    selection = select_item(
        [attempted, unseen],
        skill_id="AI-SRC-01",
        desired_difficulty="introductory",
        excluded_item_ids=set(),
        last_answered_item_id=None,
        attempted_item_ids=frozenset({attempted.item_id}),
    )

    assert selection.item.item_id == unseen.item_id


def test_select_item_falls_back_to_attempted_items_once_unseen_is_exhausted():
    only_attempted = item("item-attempted", "AI-SRC-01", "introductory")

    selection = select_item(
        [only_attempted],
        skill_id="AI-SRC-01",
        desired_difficulty="introductory",
        excluded_item_ids=set(),
        last_answered_item_id=None,
        attempted_item_ids=frozenset({only_attempted.item_id}),
    )

    assert selection.item.item_id == only_attempted.item_id
