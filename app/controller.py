"""Application boundary coordinating recommendation, scoring, BKT, and SQLite."""

import logging
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from functools import wraps
from threading import RLock
from uuid import uuid4

from api.bank import BankItem
from api.presentation import option_id, present_bank_item, presentation_from_seed
from bkt.repository import AttemptConflictError
from bkt.schemas import AttemptEvent, MasterySnapshot
from bkt.service import BKTService
from recommendation.policy import difficulty_for_mastery
from recommendation.schemas import RecommendationRequest
from recommendation.service import RecommendationService, RecommendationUnavailable
from taxonomy.schemas import SkillDefinition

from app.perf import phase
from app.session import attempt_id_for
from app.view_models import (
    ContentGapViewModel,
    ProgressViewModel,
    QuestionOptionViewModel,
    QuestionViewModel,
    SubmissionResultViewModel,
    readable_recommendation_reason,
)


LOGGER = logging.getLogger(__name__)


def _synchronized(method):
    @wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapped


class ApplicationError(RuntimeError):
    """Expected learner-facing failure with a safe message."""

    user_message = "The quiz could not complete that action. Please try again."

    def __init__(self, message: str | None = None) -> None:
        super().__init__(message or self.user_message)


class NoRecommendationError(ApplicationError):
    user_message = "No eligible recommendation is available right now."


class NoApprovedItemError(ApplicationError):
    user_message = "No approved question is available for the selected skill."


class BankExhaustedBelowMasteryError(ApplicationError):
    user_message = (
        "You've answered every available practice question for this skill "
        "without yet reaching mastery. Check back once more questions are added."
    )


class AllEligibleItemsAttemptedError(ApplicationError):
    user_message = (
        "You've mastered every skill with available practice questions right "
        "now. There's nothing new to practice until more content is added."
    )


class ContentGapError(ApplicationError):
    user_message = (
        "You unlocked the next skill, but approved practice content is not available yet."
    )

    def __init__(self, content_gap: ContentGapViewModel) -> None:
        self.content_gap = content_gap
        super().__init__(self.user_message)


class InvalidOptionError(ApplicationError):
    user_message = "That answer option is not valid for this question."


class PresentationNotFoundError(ApplicationError):
    user_message = "This question is no longer available. Request a new question."


class ConflictingAttemptError(ApplicationError):
    user_message = "This submission conflicts with an answer already recorded."


class BKTUpdateError(ApplicationError):
    user_message = "Your answer could not be saved. Please try submitting it again."


class ApplicationController:
    def __init__(
        self,
        *,
        course_id: str,
        skills: Sequence[SkillDefinition],
        items: Sequence[BankItem],
        repository,
        recommendation_service: RecommendationService,
        bkt_service: BKTService,
        clock: Callable[[], datetime] | None = None,
        presentation_token_factory: Callable[[], str] | None = None,
        low_supply_reporter: Callable[[str], None] | None = None,
    ) -> None:
        if not course_id.strip():
            raise ValueError("course_id cannot be empty")
        self.course_id = course_id
        self._skills = {skill.skill_id: skill.model_copy(deep=True) for skill in skills}
        self._items = {
            item.item_id: item.model_copy(deep=True)
            for item in items
            if item.item_id is not None
        }
        self.repository = repository
        self.recommendation_service = recommendation_service
        self.bkt_service = bkt_service
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._presentation_token_factory = presentation_token_factory or (
            lambda: str(uuid4())
        )
        self._low_supply_reporter = low_supply_reporter
        self._lock = RLock()

    def _report_low_supply(self, skill_id: str) -> None:
        # Fire-and-forget: an idempotent local insert only, never network or
        # model calls, and never allowed to affect the learner response.
        if self._low_supply_reporter is None:
            return
        try:
            self._low_supply_reporter(skill_id)
        except Exception:
            LOGGER.warning("low-supply reporting failed for skill %s", skill_id, exc_info=True)

    @_synchronized
    def start_learner_session(self, learner_id: str) -> str:
        learner_id = self._normalise_learner_id(learner_id)
        self.repository.save_learner(learner_id, created_at=self._clock())
        return learner_id

    @_synchronized
    def answered_item_ids(self, learner_id: str) -> list[str]:
        """Scoped to this controller's own course: attempt_events is a
        shared table across every course's controller (see
        app/multi_course.py), so an unfiltered read here would leak another
        course's item_ids into this course's excluded-item/seen-item state."""
        learner_id = self._normalise_learner_id(learner_id)
        return list(
            dict.fromkeys(
                attempt.item_id
                for attempt in self.repository.list_attempts(learner_id=learner_id)
                if attempt.item_id in self._items
            )
        )

    @_synchronized
    def recommend_question(
        self, learner_id: str, excluded_item_ids: Sequence[str]
    ) -> QuestionViewModel:
        learner_id = self.start_learner_session(learner_id)
        available_skill_ids = sorted(
            {
                item.skill_id
                for item in self.repository.list_approved_buildable_items()
                if item.item_id in self._items and item.skill_id in self._skills
            }
        )
        if not available_skill_ids:
            LOGGER.info(
                "Recommendation unavailable for learner %s: empty usable item inventory",
                learner_id,
            )
            raise NoApprovedItemError
        try:
            with phase("recommendation_calculation", course_id=self.course_id):
                recommendation = self.recommendation_service.recommend(
                    RecommendationRequest(
                        learner_id=learner_id,
                        available_skill_ids=available_skill_ids,
                        excluded_item_ids=list(excluded_item_ids),
                    )
                )
        except RecommendationUnavailable as exc:
            LOGGER.info(
                "Recommendation unavailable for learner %s: %s", learner_id, exc.reason
            )
            if exc.content_gap is not None:
                gap = exc.content_gap
                self._report_low_supply(gap.newly_unlocked_skill_id)
                raise ContentGapError(
                    ContentGapViewModel(
                        completed_skill_id=gap.completed_skill_id,
                        completed_skill_name=gap.completed_skill_name,
                        newly_unlocked_skill_id=gap.newly_unlocked_skill_id,
                        newly_unlocked_skill_name=gap.newly_unlocked_skill_name,
                        missing_approved_content=gap.missing_approved_content,
                        current_mastery_probability=gap.current_mastery_probability,
                        prerequisite_mastery_threshold=(
                            gap.prerequisite_mastery_threshold
                        ),
                    )
                ) from exc
            if exc.reason == "bank_empty":
                raise NoApprovedItemError from exc
            if exc.reason == "bank_exhausted_below_mastery":
                raise BankExhaustedBelowMasteryError from exc
            if exc.reason == "all_eligible_items_attempted":
                raise AllEligibleItemsAttemptedError from exc
            raise NoRecommendationError from exc

        item = self._items.get(recommendation.item_id)
        skill = self._skills.get(recommendation.skill_id)
        if item is None or skill is None:
            LOGGER.error("Recommendation references unavailable runtime data: %s", recommendation)
            raise NoApprovedItemError

        presentation = present_bank_item(
            item,
            learner_id=learner_id,
            attempt_id=self._presentation_token_factory(),
        )
        with phase("question_persistence", course_id=self.course_id):
            self.repository.save_presentation(
                presentation_id=presentation.presentation_id,
                learner_id=learner_id,
                item_id=item.item_id,
                presentation_seed=presentation.presentation_seed,
                recommendation_reason=recommendation.reason,
                created_at=self._clock(),
            )
        return QuestionViewModel(
            presentation_id=presentation.presentation_id,
            item_id=item.item_id,
            skill_id=item.skill_id,
            topic=skill.topic,
            concept=item.question.concept,
            difficulty=item.question.difficulty,
            question_text=item.question.question,
            options=[
                QuestionOptionViewModel(option_id=choice.option_id, text=choice.value)
                for choice in presentation.presented_options
            ],
            recommendation_reason=readable_recommendation_reason(
                recommendation.reason
            ),
        )

    @_synchronized
    def submit_answer(
        self, learner_id: str, presentation_id: str, selected_option_id: str
    ) -> SubmissionResultViewModel:
        learner_id = self._normalise_learner_id(learner_id)
        with phase("answer_persistence", course_id=self.course_id):
            row = self.repository.get_presentation(presentation_id)
            if row is None or row["learner_id"] != learner_id:
                raise PresentationNotFoundError
            item = self._items.get(row["item_id"])
            if item is None:
                raise PresentationNotFoundError
            presentation = presentation_from_seed(item, int(row["presentation_seed"]))
            valid_option_ids = {
                choice.option_id for choice in presentation.presented_options
            }
            if selected_option_id not in valid_option_ids:
                raise InvalidOptionError

            attempt_id = attempt_id_for(learner_id, presentation_id)
            existing = self.repository.get_attempt(attempt_id)
            if existing is not None:
                if (
                    existing.learner_id != learner_id
                    or existing.presentation_id != presentation_id
                    or existing.selected_option_id != selected_option_id
                ):
                    raise ConflictingAttemptError
                snapshot = self.repository.get_mastery_for_attempt(attempt_id)
                if snapshot is None:
                    LOGGER.error("Attempt %s has no mastery snapshot", attempt_id)
                    raise BKTUpdateError
                return self._submission_result(item, existing, snapshot)

            history = self.repository.list_attempts(
                learner_id=learner_id, skill_id=item.skill_id
            )
            attempt = AttemptEvent(
                attempt_id=attempt_id,
                course_id=self.course_id,
                presentation_id=presentation_id,
                learner_id=learner_id,
                item_id=item.item_id,
                skill_id=item.skill_id,
                selected_option_id=selected_option_id,
                correct=False,
                attempt_order=len(history) + 1,
                occurred_at=self._clock(),
            )
        try:
            snapshot = self.bkt_service.process_attempt(
                attempt, item=item, presentation=presentation
            )
        except AttemptConflictError as exc:
            LOGGER.warning("Conflicting duplicate attempt %s", attempt_id, exc_info=True)
            raise ConflictingAttemptError from exc
        except Exception as exc:
            LOGGER.exception("BKT update failed for attempt %s", attempt_id)
            raise BKTUpdateError from exc

        with phase("answer_persistence_verify", course_id=self.course_id):
            scored_attempt = self.repository.get_attempt(attempt_id)
        if scored_attempt is None:
            LOGGER.error("BKT update returned without storing attempt %s", attempt_id)
            raise BKTUpdateError
        return self._submission_result(item, scored_attempt, snapshot)

    @_synchronized
    def get_progress(self, learner_id: str, skill_id: str) -> ProgressViewModel:
        learner_id = self._normalise_learner_id(learner_id)
        skill = self._skills.get(skill_id)
        if skill is None:
            raise ValueError(f"Unknown skill_id: {skill_id}")
        snapshot = self.repository.get_mastery(learner_id, skill_id)
        mastery = (
            snapshot.mastery_probability
            if snapshot is not None
            else self.recommendation_service.config.initial_mastery_probability
        )
        recommendations = [
            event
            for event in self.repository.list_recommendations(learner_id=learner_id)
            if event.skill_id == skill_id
        ]
        reason = recommendations[-1].reason if recommendations else None
        return ProgressViewModel(
            learner_id=learner_id,
            skill_id=skill_id,
            skill_name=skill.name,
            mastery_probability=mastery,
            attempt_count=snapshot.attempt_count if snapshot else 0,
            difficulty=difficulty_for_mastery(
                mastery, self.recommendation_service.config
            ),
            readable_recommendation_reason=readable_recommendation_reason(reason),
        )

    def _submission_result(
        self, item: BankItem, attempt: AttemptEvent, snapshot: MasterySnapshot
    ) -> SubmissionResultViewModel:
        previous = self.recommendation_service.config.initial_mastery_probability
        if snapshot.attempt_count > 1:
            candidates = [
                stored
                for stored in self.repository.list_mastery(learner_id=attempt.learner_id)
                if stored.skill_id == attempt.skill_id
                and stored.attempt_count == snapshot.attempt_count - 1
            ]
            if candidates:
                previous = candidates[-1].mastery_probability
        return SubmissionResultViewModel(
            attempt_id=attempt.attempt_id,
            correct=attempt.correct,
            correct_option_id=option_id(item.item_id, item.question.correct_answer),
            correct_answer_text=item.question.correct_answer,
            explanation=item.question.explanation,
            previous_mastery=previous,
            updated_mastery=snapshot.mastery_probability,
            attempt_count=snapshot.attempt_count,
        )

    @staticmethod
    def _normalise_learner_id(learner_id: str) -> str:
        learner_id = learner_id.strip()
        if not learner_id:
            raise ValueError("Learner ID is required.")
        return learner_id
