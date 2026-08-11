"""Streamlit composition root for the adaptive learner loop."""

import logging

import streamlit as st

from app.bootstrap import AppSettings, BootstrapError, build_controller
from app.controller import ApplicationError, ContentGapError
from app.session import (
    begin_submission,
    get_session_state,
    identify_learner,
    next_question,
    retain_question,
    submission_failed,
    submission_succeeded,
)


LOGGER = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Preparing the quiz…")
def get_controller():
    """Cache only shared services; learner workflow remains in session_state/SQLite."""

    return build_controller(AppSettings.from_sources(st.secrets))


def main() -> None:
    st.set_page_config(page_title="Adaptive Quiz", page_icon="✦", layout="wide")
    st.title("Adaptive Quiz")
    st.caption("One question at a time, selected from your current learning state.")

    try:
        from app.ui.feedback import render_feedback
        from app.ui.content_gap import render_content_gap
        from app.ui.login import render_login
        from app.ui.progress import render_progress
        from app.ui.question import render_question
    except ImportError:
        LOGGER.exception("Streamlit UI component modules are unavailable")
        st.error("The quiz interface is not installed yet.")
        return

    try:
        controller = get_controller()
    except BootstrapError as exc:
        st.error(exc.user_message)
        return

    session = get_session_state(st.session_state)
    if session.learner_id is None:
        learner_id = render_login(default_learner_id="")
        if learner_id is None:
            return
        try:
            learner_id = controller.start_learner_session(learner_id)
            identify_learner(session, learner_id)
        except (ApplicationError, ValueError) as exc:
            LOGGER.exception("Learner session could not be started")
            st.error(getattr(exc, "user_message", str(exc)))
            return
        st.rerun()

    if session.question is None and session.feedback_state != "visible":
        try:
            retain_question(
                session,
                controller.recommend_question(
                    session.learner_id, session.seen_item_ids
                ),
            )
        except ContentGapError as exc:
            render_content_gap(exc.content_gap)
            return
        except ApplicationError as exc:
            st.warning(exc.user_message)
            return
        except Exception:
            LOGGER.exception("Question recommendation failed")
            st.error("A question could not be prepared. Please try again.")
            return

    if session.feedback_state == "visible" and session.feedback is not None:
        try:
            render_progress(
                controller.get_progress(
                    session.learner_id, session.question.skill_id
                )
            )
        except Exception:
            LOGGER.exception("Progress rendering data could not be loaded")
            st.warning("Progress is temporarily unavailable.")
        if render_feedback(session.feedback, next_enabled=True):
            next_question(session)
            st.rerun()
        return

    selected_option_id, submit_clicked = render_question(
        session.question,
        selected_option_id=session.selected_option_id,
        disabled=session.submission_state == "processing",
    )
    if selected_option_id is not None:
        session.selected_option_id = selected_option_id
    if session.error_message:
        st.error(session.error_message)
    if not submit_clicked:
        return

    try:
        begin_submission(session, selected_option_id)
        result = controller.submit_answer(
            session.learner_id,
            session.presentation_id,
            session.selected_option_id,
        )
        submission_succeeded(session, result)
        st.rerun()
    except ApplicationError as exc:
        submission_failed(session, exc.user_message)
    except ValueError as exc:
        submission_failed(session, str(exc))
    except Exception:
        LOGGER.exception("Unexpected answer submission failure")
        submission_failed(session, "Your answer could not be submitted. Please try again.")


if __name__ == "__main__":
    main()
