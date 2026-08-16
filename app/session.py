"""Small, explicit state machine for Streamlit's rerun execution model."""

import hashlib
from typing import Literal, MutableMapping

from pydantic import BaseModel, ConfigDict, Field

from app.view_models import QuestionViewModel, SubmissionResultViewModel


SESSION_KEY = "adaptive_quiz_session"


class LearnerSessionState(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    learner_id: str | None = None
    course_id: str | None = None
    question: QuestionViewModel | None = None
    presentation_id: str | None = None
    item_id: str | None = None
    selected_option_id: str | None = None
    attempt_id: str | None = None
    submission_state: Literal["not_submitted", "processing", "failed", "submitted"] = (
        "not_submitted"
    )
    feedback_state: Literal["hidden", "visible"] = "hidden"
    feedback: SubmissionResultViewModel | None = None
    # Scoped to the *current round* only (see round_number below): reset to
    # [] at the start of every round, so "no repeats" applies within a
    # round, not for the learner's whole lifetime -- a later round may
    # legitimately resurface an item from an earlier round once genuinely
    # unseen content runs low (recommendation/policy.py's select_item
    # prioritizes lifetime-unseen items ahead of that).
    seen_item_ids: list[str] = Field(default_factory=list)
    error_message: str | None = None

    round_number: int = 1
    round_state: Literal["in_progress", "round_complete", "paused", "finished"] = (
        "in_progress"
    )
    # The active bank's fingerprint (ApplicationController.bank_version) at
    # the moment this round started, recorded so a round's content
    # provenance is auditable even though the controller's own item set is
    # already static for its cached lifetime.
    round_bank_version: str | None = None
    # This round's "focus on weak areas" choice, threaded into every
    # recommend_question call in the round.
    restrict_to_weak_skills: bool = False


def attempt_id_for(learner_id: str, presentation_id: str) -> str:
    digest = hashlib.sha256(f"{learner_id}\0{presentation_id}".encode()).hexdigest()
    return f"attempt-{digest[:24]}"


def get_session_state(state: MutableMapping[str, object]) -> LearnerSessionState:
    current = state.get(SESSION_KEY)
    if isinstance(current, LearnerSessionState):
        return current
    created = LearnerSessionState()
    state[SESSION_KEY] = created
    return created


def identify_learner(session: LearnerSessionState, learner_id: str) -> None:
    learner_id = learner_id.strip()
    if not learner_id:
        raise ValueError("Learner ID is required.")
    if session.learner_id != learner_id:
        fresh = LearnerSessionState(learner_id=learner_id)
        for name, value in fresh.model_dump().items():
            setattr(session, name, value)


def clear_learner(session: LearnerSessionState) -> None:
    fresh = LearnerSessionState()
    for name, value in fresh.model_dump().items():
        setattr(session, name, value)


def select_course(
    session: LearnerSessionState, course_id: str, *, bank_version: str | None = None
) -> None:
    session.course_id = course_id
    session.seen_item_ids = []
    session.question = None
    session.presentation_id = None
    session.item_id = None
    session.selected_option_id = None
    session.attempt_id = None
    session.submission_state = "not_submitted"
    session.feedback_state = "hidden"
    session.feedback = None
    session.error_message = None
    session.round_number = 1
    session.round_state = "in_progress"
    session.round_bank_version = bank_version
    session.restrict_to_weak_skills = False


def mark_round_complete(session: LearnerSessionState) -> None:
    """Called when the controller raises RoundCompleteError: presents the
    checkpoint (Continue / Focus on weak areas / Pause / Finish) instead of
    a terminal error. Mastery and attempt history are untouched -- only
    display state changes here."""
    session.round_state = "round_complete"


def start_new_round(
    session: LearnerSessionState, *, bank_version: str | None, focus_weak_areas: bool = False
) -> None:
    """Starts the next round: resets only this round's seen-item exclusion
    set and question-display state. Never touches mastery or attempt
    history, both of which live durably in the repository keyed by
    learner_id/skill_id, independent of round or session boundaries."""
    session.round_number += 1
    session.round_state = "in_progress"
    session.round_bank_version = bank_version
    session.restrict_to_weak_skills = focus_weak_areas
    session.seen_item_ids = []
    session.question = None
    session.presentation_id = None
    session.item_id = None
    session.selected_option_id = None
    session.attempt_id = None
    session.submission_state = "not_submitted"
    session.feedback_state = "hidden"
    session.feedback = None
    session.error_message = None


def pause_session(session: LearnerSessionState) -> None:
    session.round_state = "paused"


def finish_session(session: LearnerSessionState) -> None:
    session.round_state = "finished"
    session.question = None


def retain_question(session: LearnerSessionState, question: QuestionViewModel) -> None:
    if session.feedback_state == "visible":
        raise RuntimeError("Finish reviewing feedback before requesting another question.")
    session.question = question
    session.presentation_id = question.presentation_id
    session.item_id = question.item_id
    session.selected_option_id = None
    session.attempt_id = None
    session.submission_state = "not_submitted"
    session.feedback_state = "hidden"
    session.feedback = None
    session.error_message = None


def begin_submission(session: LearnerSessionState, selected_option_id: str | None) -> str:
    if not session.learner_id or not session.presentation_id:
        raise RuntimeError("There is no question to submit.")
    if session.submission_state == "submitted":
        raise RuntimeError("This question has already been submitted.")
    if not selected_option_id:
        raise ValueError("Select an answer before submitting.")
    if session.attempt_id is None:
        session.attempt_id = attempt_id_for(session.learner_id, session.presentation_id)
    session.selected_option_id = selected_option_id
    session.submission_state = "processing"
    session.error_message = None
    return session.attempt_id


def submission_succeeded(
    session: LearnerSessionState, result: SubmissionResultViewModel
) -> None:
    if session.attempt_id is not None and result.attempt_id != session.attempt_id:
        raise ValueError("Submission result does not match this presentation.")
    session.attempt_id = result.attempt_id
    session.submission_state = "submitted"
    session.feedback_state = "visible"
    session.feedback = result
    session.error_message = None


def submission_failed(session: LearnerSessionState, message: str) -> None:
    session.submission_state = "failed"
    session.error_message = message


def next_question(session: LearnerSessionState) -> None:
    if session.item_id and session.item_id not in session.seen_item_ids:
        session.seen_item_ids.append(session.item_id)
    session.question = None
    session.presentation_id = None
    session.item_id = None
    session.selected_option_id = None
    session.attempt_id = None
    session.submission_state = "not_submitted"
    session.feedback_state = "hidden"
    session.feedback = None
    session.error_message = None
