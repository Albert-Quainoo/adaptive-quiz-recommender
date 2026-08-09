from dataclasses import dataclass
from typing import Mapping, Sequence

from api.bank import BankItem
from api.schemas import difficulty_level
from taxonomy.schemas import SkillDefinition

from recommendation.schemas import RecommendationReason


DIFFICULTIES: tuple[difficulty_level, ...] = (
    "introductory",
    "intermediate",
    "advanced",
)


@dataclass(frozen=True)
class RecommendationPolicyConfig:
    prerequisite_mastery_threshold: float = 0.75
    initial_mastery_probability: float = 0.0
    introductory_mastery_threshold: float = 0.40
    advanced_mastery_threshold: float = 0.75
    policy_version: str = "recommendation-policy-v1"

    def __post_init__(self) -> None:
        probabilities = (
            self.prerequisite_mastery_threshold,
            self.initial_mastery_probability,
            self.introductory_mastery_threshold,
            self.advanced_mastery_threshold,
        )
        if any(probability < 0.0 or probability > 1.0 for probability in probabilities):
            raise ValueError("mastery thresholds must be between 0 and 1")
        if self.introductory_mastery_threshold >= self.advanced_mastery_threshold:
            raise ValueError("difficulty mastery thresholds must be increasing")
        if not self.policy_version.strip():
            raise ValueError("policy_version cannot be empty")


@dataclass(frozen=True)
class SkillSelection:
    skill: SkillDefinition
    mastery_probability: float
    reason: RecommendationReason


@dataclass(frozen=True)
class ItemSelection:
    item: BankItem
    reason: RecommendationReason | None


def mastery_for(
    skill_id: str,
    mastery_by_skill: Mapping[str, float],
    config: RecommendationPolicyConfig,
) -> float:
    return mastery_by_skill.get(skill_id, config.initial_mastery_probability)


def prerequisites_are_mastered(
    skill: SkillDefinition,
    mastery_by_skill: Mapping[str, float],
    config: RecommendationPolicyConfig,
) -> bool:
    return all(
        mastery_for(prerequisite_id, mastery_by_skill, config)
        >= config.prerequisite_mastery_threshold
        for prerequisite_id in skill.prerequisite_skill_ids
    )


def select_skill(
    skills: Sequence[SkillDefinition],
    available_skill_ids: set[str],
    mastery_by_skill: Mapping[str, float],
    config: RecommendationPolicyConfig,
) -> SkillSelection | None:
    """Choose an eligible skill; input order is the taxonomy tie-breaker."""

    eligible = [
        (taxonomy_order, skill)
        for taxonomy_order, skill in enumerate(skills)
        if skill.skill_id in available_skill_ids
        and prerequisites_are_mastered(skill, mastery_by_skill, config)
    ]
    if not eligible:
        return None

    if not mastery_by_skill:
        foundational = [entry for entry in eligible if not entry[1].prerequisite_skill_ids]
        if foundational:
            taxonomy_order, skill = min(
                foundational, key=lambda entry: (entry[0], entry[1].skill_id)
            )
            return SkillSelection(
                skill=skill,
                mastery_probability=mastery_for(skill.skill_id, mastery_by_skill, config),
                reason="foundational_unseen_skill",
            )

    _, skill = min(
        eligible,
        key=lambda entry: (
            mastery_for(entry[1].skill_id, mastery_by_skill, config),
            entry[0],
            entry[1].skill_id,
        ),
    )
    return SkillSelection(
        skill=skill,
        mastery_probability=mastery_for(skill.skill_id, mastery_by_skill, config),
        reason="lowest_mastery_eligible_skill",
    )


def difficulty_for_mastery(
    mastery_probability: float, config: RecommendationPolicyConfig
) -> difficulty_level:
    if mastery_probability < config.introductory_mastery_threshold:
        return "introductory"
    if mastery_probability < config.advanced_mastery_threshold:
        return "intermediate"
    return "advanced"


def select_item(
    items: Sequence[BankItem],
    *,
    skill_id: str,
    desired_difficulty: difficulty_level,
    excluded_item_ids: set[str],
    last_answered_item_id: str | None,
) -> ItemSelection | None:
    candidates = [
        item
        for item in items
        if item.skill_id == skill_id
        and item.item_id is not None
        and item.item_id not in excluded_item_ids
    ]
    if not candidates:
        return None

    non_repeats = [item for item in candidates if item.item_id != last_answered_item_id]
    if non_repeats:
        candidates = non_repeats

    desired_index = DIFFICULTIES.index(desired_difficulty)
    selected = min(
        candidates,
        key=lambda item: (
            abs(DIFFICULTIES.index(item.question.difficulty) - desired_index),
            DIFFICULTIES.index(item.question.difficulty),
            item.item_id,
        ),
    )
    reason = (
        None
        if selected.question.difficulty == desired_difficulty
        else "fallback_difficulty_used"
    )
    return ItemSelection(item=selected, reason=reason)
