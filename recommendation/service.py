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
)
from recommendation.repository import RecommendationRepository
from recommendation.schemas import RecommendationRequest, RecommendationResult


LOGGER = logging.getLogger(__name__)


class RecommendationUnavailable(RuntimeError):
    def __init__(self, reason: str) -> None:
        self.reason = reason
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
