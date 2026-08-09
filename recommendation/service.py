from recommendation.policy import (
    RecommendationPolicyConfig,
    difficulty_for_mastery,
    select_item,
    select_skill,
)
from recommendation.repository import RecommendationRepository
from recommendation.schemas import RecommendationRequest, RecommendationResult


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
        skill_selection = select_skill(
            self.repository.list_skills(),
            set(request.available_skill_ids),
            mastery_by_skill,
            self.config,
        )
        if skill_selection is None:
            raise RecommendationUnavailable("no_eligible_skill")

        desired_difficulty = request.requested_difficulty or difficulty_for_mastery(
            skill_selection.mastery_probability, self.config
        )
        attempts = self.repository.list_attempts(learner_id=request.learner_id)
        last_answered_item_id = attempts[-1].item_id if attempts else None
        item_selection = select_item(
            self.repository.list_approved_buildable_items(),
            skill_id=skill_selection.skill.skill_id,
            desired_difficulty=desired_difficulty,
            excluded_item_ids=set(request.excluded_item_ids),
            last_answered_item_id=last_answered_item_id,
        )
        if item_selection is None:
            raise RecommendationUnavailable("no_eligible_item")

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
