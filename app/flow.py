"""Thin orchestration between Streamlit state and the application controller."""

from app.controller import ApplicationController, ApplicationError
from app.session import (
    LearnerSessionState,
    begin_submission,
    identify_learner,
    retain_question,
    select_course,
    submission_failed,
    submission_succeeded,
)
from app.view_models import QuestionViewModel, SubmissionResultViewModel


def activate_learner(
    controller: ApplicationController,
    session: LearnerSessionState,
    learner_id: str,
) -> str:
    normalized = controller.start_learner_session(learner_id)
    identify_learner(session, normalized)
    return normalized


def activate_course(
    controller: ApplicationController,
    session: LearnerSessionState,
    course_id: str,
) -> None:
    with controller.repository.unit_of_work():
        normalized = controller.start_learner_session(session.learner_id or "")
        select_course(session, course_id, bank_version=controller.bank_version)
        if controller.max_session_questions is None:
            # Unbounded course (e.g. intro-ai): preserve the original
            # behavior exactly -- seed with the learner's full lifetime
            # history so nothing already answered is ever re-served. Round
            # checkpoints don't exist for this course, so there is no later
            # point where that full history would be reset.
            session.seen_item_ids = controller.answered_item_ids(normalized)
        # Round-based courses start each round's excluded set empty --
        # recommendation/policy.py's select_item still prefers the
        # learner's lifetime-unseen items first, so genuinely new content
        # is not repeated ahead of schedule; it just isn't a hard exclusion
        # once a round wants to resurface weak-area content.


def ensure_question(
    controller: ApplicationController,
    session: LearnerSessionState,
) -> QuestionViewModel:
    if not session.learner_id:
        raise ValueError("Learner ID is required.")
    if session.question is None:
        question = controller.recommend_question(
            session.learner_id,
            session.seen_item_ids,
            restrict_to_weak_skills=session.restrict_to_weak_skills,
        )
        retain_question(session, question)
    return session.question


def submit_current_question(
    controller: ApplicationController,
    session: LearnerSessionState,
    selected_option_id: str | None,
) -> SubmissionResultViewModel:
    if session.submission_state == "submitted" and session.feedback is not None:
        return session.feedback
    begin_submission(session, selected_option_id)
    try:
        result = controller.submit_answer(
            session.learner_id or "",
            session.presentation_id or "",
            session.selected_option_id or "",
        )
    except Exception as error:
        message = (
            error.user_message
            if isinstance(error, ApplicationError)
            else "The answer could not be processed. Please try again."
        )
        submission_failed(session, message)
        raise
    submission_succeeded(session, result)
    return result
