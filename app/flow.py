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
    normalized = controller.start_learner_session(session.learner_id or "")
    select_course(session, course_id)
    session.seen_item_ids = controller.answered_item_ids(normalized)


def ensure_question(
    controller: ApplicationController,
    session: LearnerSessionState,
) -> QuestionViewModel:
    if not session.learner_id:
        raise ValueError("Learner ID is required.")
    if session.question is None:
        question = controller.recommend_question(
            session.learner_id, session.seen_item_ids
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
