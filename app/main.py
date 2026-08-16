"""Learner-facing Streamlit quiz application."""

import sys
from pathlib import Path

# Streamlit executes this file with app/ as the script directory. Add the
# repository root so package imports behave the same as `python -m` execution.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from app.bootstrap import AppSettings, BootstrapError
from app.controller import (
    AllEligibleItemsAttemptedError,
    ApplicationError,
    BankExhaustedBelowMasteryError,
    ContentGapError,
    NoApprovedItemError,
    NoRecommendationError,
)
from app.flow import activate_course, ensure_question, submit_current_question
from app.multi_course import build_course_catalog
from app.session import clear_learner, get_session_state, identify_learner, next_question
from app.ui.content_gap import render_content_gap
from app.ui.course_selector import render_course_selector
from app.ui.feedback import render_feedback
from app.ui.login import render_learner_switcher, render_login
from app.ui.progress import render_progress
from app.ui.question import render_question
from app.ui.theme import inject_theme


@st.cache_resource(show_spinner="Loading approved questions and learner models…")
def load_application(settings: AppSettings):
    return build_course_catalog(settings)


@st.cache_resource(show_spinner=False)
def load_replenishment_job_repository(database_path: str) -> "SQLiteReplenishmentJobRepository":
    from authoring.replenishment.jobs import SQLiteReplenishmentJobRepository

    repository = SQLiteReplenishmentJobRepository(database_path)
    repository.initialize_schema()
    return repository


def render_admin_status(database_path: str) -> None:
    """Read-only content-status sidebar. Never touches Brave or Llama, and
    any failure here must never disturb the learner-facing quiz. The
    replenishment subsystem is optional: import it lazily so the core app
    still starts when authoring/replenishment/ is unavailable."""
    try:
        from app.ui.replenishment_admin import render_replenishment_admin
        from authoring.replenishment.inventory import compute_course_inventory
        from authoring.replenishment.manifest import load_preparation_eligible_manifests

        job_repository = load_replenishment_job_repository(database_path)
        for manifest in load_preparation_eligible_manifests():
            inventory = compute_course_inventory(manifest, job_repository)
            render_replenishment_admin(manifest.course_id, list(inventory.values()))
    except Exception:
        pass


def configured_settings() -> AppSettings:
    try:
        secrets = dict(st.secrets)
    except FileNotFoundError:
        secrets = {}
    return AppSettings.from_sources(secrets)


def render_recommendation_error(error: ApplicationError, *, has_seen: bool) -> None:
    if isinstance(error, NoApprovedItemError) and has_seen:
        st.info("You have completed all currently available questions for this learner.")
    elif isinstance(error, NoRecommendationError):
        st.info("No eligible question is available for this learner right now.")
    elif isinstance(error, (BankExhaustedBelowMasteryError, AllEligibleItemsAttemptedError)):
        st.info(error.user_message)
    else:
        st.error(error.user_message)


def main() -> None:
    st.set_page_config(
        page_title="Adaptive Quiz",
        page_icon=None,
        layout="centered",
    )
    inject_theme()

    try:
        settings = configured_settings()
        catalogue = load_application(settings)
    except (BootstrapError, ValueError) as error:
        st.title("Adaptive Quiz")
        st.error(str(error))
        st.caption("Check the configured bank, database, and BKT model settings.")
        return

    render_admin_status(str(settings.database_path))

    session = get_session_state(st.session_state)
    if session.learner_id is None:
        learner_id = render_login()
        if learner_id is None:
            return
        # Session-state identification only -- no course is known yet, so no
        # course's controller (and therefore no bank/taxonomy/BKT-model I/O)
        # is built at this point. The learner's row in the shared
        # learner_sessions table is written once a course, and therefore a
        # real controller, is selected (see activate_course, below).
        try:
            identify_learner(session, learner_id)
        except ValueError as error:
            st.error(str(error))
            return
        st.rerun()

    if render_learner_switcher(session.learner_id):
        clear_learner(session)
        st.rerun()

    if session.course_id is None:
        course_id = render_course_selector(catalogue)
        if course_id is None:
            return
        activate_course(catalogue.resolve_active(course_id), session, course_id)
        st.rerun()

    controller = catalogue.resolve_active(session.course_id)

    st.title("Adaptive Quiz")
    st.caption("One question at a time · progress is saved automatically")

    if session.question is None:
        try:
            ensure_question(controller, session)
        except ContentGapError as error:
            render_content_gap(error.content_gap)
            return
        except (
            NoApprovedItemError,
            NoRecommendationError,
            BankExhaustedBelowMasteryError,
            AllEligibleItemsAttemptedError,
        ) as error:
            render_recommendation_error(error, has_seen=bool(session.seen_item_ids))
            return
        except ApplicationError as error:
            st.error(error.user_message)
            return

    question = session.question
    if question is None:
        st.error("The recommended question could not be displayed.")
        return

    try:
        render_progress(controller.get_progress(session.learner_id, question.skill_id))
    except Exception:
        st.error("Saved learner progress could not be loaded.")
        return

    selected, submitted = render_question(
        question,
        selected_option_id=session.selected_option_id,
        disabled=session.submission_state == "submitted",
    )

    if submitted and session.submission_state != "submitted":
        try:
            submit_current_question(controller, session, selected)
            st.rerun()
        except ValueError as error:
            st.warning(str(error))
        except ApplicationError:
            pass
        except Exception:
            pass

    if session.error_message and session.submission_state == "failed":
        st.error(session.error_message)

    if session.feedback_state == "visible" and session.feedback is not None:
        if render_feedback(session.feedback):
            next_question(session)
            st.rerun()


if __name__ == "__main__":
    main()
