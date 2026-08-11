import logging
from collections.abc import Sequence

from api.bank import BankItem
from api.schemas import difficulty_level
from recommendation.policy import (
    DIFFICULTIES,
    RecommendationPolicyConfig,
    difficulty_for_mastery,
    select_item,
    select_skill,
    prerequisites_are_mastered,
)
from recommendation.repository import RecommendationRepository
from recommendation.schemas import (
    ContentGapResult,
    RecommendationRequest,
    RecommendationResult,
)


LOGGER = logging.getLogger(__name__)


class RecommendationUnavailable(RuntimeError):
    def __init__(
        self, reason: str, *, content_gap: ContentGapResult | None = None
    ) -> None:
        self.reason = reason
        self.content_gap = content_gap
        super().__init__(reason)


class RecommendationService:
    def __init__(
        self,
        repository: RecommendationRepository,
        *,
        model_version: str,
        config: RecommendationPolicyConfig | None = None,
    ) -> None:
        if not model_version.strip():
            raise ValueError("model_version cannot be empty")
        self.repository = repository
        self.model_version = model_version
        self.config = config or RecommendationPolicyConfig()

    def recommend(self, request: RecommendationRequest) -> RecommendationResult:
        snapshots = self.repository.list_latest_mastery(request.learner_id)
        mastery_by_skill = {
            snapshot.skill_id: snapshot.mastery_probability for snapshot in snapshots
        }
        skills = self.repository.list_skills()
        items = self.repository.list_approved_buildable_items()
        inventory_skill_ids = {
            item.skill_id
            for item in items
            if item.item_id is not None and item.skill_id is not None
        }
        content_gap = self._content_gap(
            request.learner_id,
            skills=skills,
            inventory_skill_ids=inventory_skill_ids,
            mastery_by_skill=mastery_by_skill,
            snapshots=snapshots,
        )
        if content_gap is not None:
            self.repository.save_content_gap(content_gap)
            LOGGER.info(
                "Recommendation content gap: completed_skill=%s "
                "newly_unlocked_skill=%s missing_approved_content=%s "
                "current_mastery=%s prerequisite_threshold=%s",
                content_gap.completed_skill_id,
                content_gap.newly_unlocked_skill_id,
                content_gap.missing_approved_content,
                content_gap.current_mastery_probability,
                content_gap.prerequisite_mastery_threshold,
            )
            raise RecommendationUnavailable(
                "missing_approved_content", content_gap=content_gap
            )
        remaining_skill_ids = set(request.available_skill_ids) & inventory_skill_ids
        if not remaining_skill_ids:
            raise RecommendationUnavailable("no_eligible_item")

        attempts = self.repository.list_attempts(learner_id=request.learner_id)
        last_answered_item_id = attempts[-1].item_id if attempts else None
        excluded_item_ids = set(request.excluded_item_ids)
        found_eligible_skill = False
        skill_selection = None
        item_selection = None

        while remaining_skill_ids:
            skill_selection = select_skill(
                skills,
                remaining_skill_ids,
                mastery_by_skill,
                self.config,
            )
            if skill_selection is None:
                break
            found_eligible_skill = True
            skill_id = skill_selection.skill.skill_id
            desired_difficulty = request.requested_difficulty or difficulty_for_mastery(
                skill_selection.mastery_probability, self.config
            )
            available_difficulties = self._available_difficulties(
                items,
                skill_id=skill_id,
                excluded_item_ids=excluded_item_ids,
            )
            item_selection = select_item(
                items,
                skill_id=skill_id,
                desired_difficulty=desired_difficulty,
                excluded_item_ids=excluded_item_ids,
                last_answered_item_id=last_answered_item_id,
            )
            if item_selection is None:
                LOGGER.info(
                    "Recommendation item resolution: selected_skill=%s "
                    "requested_difficulty=%s available_difficulties=%s "
                    "fallback_result=unavailable",
                    skill_id,
                    desired_difficulty,
                    available_difficulties,
                )
                remaining_skill_ids.remove(skill_id)
                continue

            fallback_result = (
                "requested"
                if item_selection.item.question.difficulty == desired_difficulty
                else f"fallback:{item_selection.item.question.difficulty}"
            )
            LOGGER.info(
                "Recommendation item resolution: selected_skill=%s "
                "requested_difficulty=%s available_difficulties=%s fallback_result=%s",
                skill_id,
                desired_difficulty,
                available_difficulties,
                fallback_result,
            )
            break

        if skill_selection is None or item_selection is None:
            reason = "no_eligible_item" if found_eligible_skill else "no_eligible_skill"
            raise RecommendationUnavailable(reason)

        reason = skill_selection.reason
        if item_selection.reason is not None:
            reason = item_selection.reason
        elif request.requested_difficulty is not None:
            reason = "requested_difficulty_used"

        result = RecommendationResult(
            learner_id=request.learner_id,
            skill_id=skill_selection.skill.skill_id,
            item_id=item_selection.item.item_id,
            difficulty=item_selection.item.question.difficulty,
            mastery_probability=skill_selection.mastery_probability,
            reason=reason,
            model_version=self.model_version,
            policy_version=self.config.policy_version,
        )
        self.repository.save_recommendation(result)
        return result

    def _content_gap(
        self,
        learner_id: str,
        *,
        skills,
        inventory_skill_ids: set[str],
        mastery_by_skill: dict[str, float],
        snapshots,
    ) -> ContentGapResult | None:
        by_id = {skill.skill_id: skill for skill in skills}
        required_skill_ids = set(inventory_skill_ids)
        pending = list(inventory_skill_ids)
        while pending:
            skill = by_id.get(pending.pop())
            if skill is None:
                continue
            for prerequisite_id in skill.prerequisite_skill_ids:
                if prerequisite_id not in required_skill_ids:
                    required_skill_ids.add(prerequisite_id)
                    pending.append(prerequisite_id)

        snapshot_by_skill = {snapshot.skill_id: snapshot for snapshot in snapshots}
        for skill in skills:
            if (
                skill.skill_id not in required_skill_ids
                or skill.skill_id in inventory_skill_ids
                or not skill.prerequisite_skill_ids
                or not prerequisites_are_mastered(
                    skill, mastery_by_skill, self.config
                )
            ):
                continue
            completed_skill_id = max(
                skill.prerequisite_skill_ids,
                key=lambda prerequisite_id: (
                    snapshot_by_skill[prerequisite_id].updated_at.timestamp()
                    if prerequisite_id in snapshot_by_skill
                    else float("-inf"),
                    prerequisite_id,
                ),
            )
            completed = by_id[completed_skill_id]
            return ContentGapResult(
                learner_id=learner_id,
                completed_skill_id=completed_skill_id,
                completed_skill_name=completed.name,
                newly_unlocked_skill_id=skill.skill_id,
                newly_unlocked_skill_name=skill.name,
                current_mastery_probability=mastery_by_skill.get(
                    completed_skill_id, self.config.initial_mastery_probability
                ),
                prerequisite_mastery_threshold=(
                    self.config.prerequisite_mastery_threshold
                ),
                model_version=self.model_version,
                policy_version=self.config.policy_version,
            )
        return None

    @staticmethod
    def _available_difficulties(
        items: Sequence[BankItem],
        *,
        skill_id: str,
        excluded_item_ids: set[str],
    ) -> list[difficulty_level]:
        represented = {
            item.question.difficulty
            for item in items
            if item.skill_id == skill_id
            and item.item_id is not None
            and item.item_id not in excluded_item_ids
        }
        return [difficulty for difficulty in DIFFICULTIES if difficulty in represented]
