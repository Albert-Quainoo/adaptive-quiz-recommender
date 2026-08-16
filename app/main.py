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


# The admin sidebar's inventory scan reads every prep-eligible course's
# taxonomy (skills.csv) and approved bank file to compute per-skill supply
# counts -- real disk I/O, not just a DB query. Caching the *computed
# result* (not just the job-repository connection) with a bounded TTL means
# that I/O happens at most once per window, shared across every learner
# session hitting the app, instead of once per page render. st.cache_data
# is process-global (not per-session), which is exactly the sharing this
# needs: course inventory isn't learner-specific state.
ADMIN_STATUS_CACHE_TTL_SECONDS = 60


@st.cache_data(ttl=ADMIN_STATUS_CACHE_TTL_SECONDS, show_spinner=False)
def _compute_admin_status_snapshot(database_path: str) -> list[tuple[str, list]]:
    from authoring.replenishment.inventory import compute_course_inventory
    from authoring.replenishment.manifest import load_preparation_eligible_manifests

    job_repository = load_replenishment_job_repository(database_path)
    return [
        (manifest.course_id, list(compute_course_inventory(manifest, job_repository).values()))
        for manifest in load_preparation_eligible_manifests()
    ]


def render_admin_status(database_path: str) -> None:
    """Read-only content-status sidebar for operators/admins -- never shown
    to learners by default (see AppSettings.admin_status_enabled, and
    main()'s gate on it). Never touches Brave or Llama, and any failure
    here must never disturb the learner-facing quiz. The replenishment
    subsystem is optional: import it lazily so the core app still starts
    when authoring/replenishment/ is unavailable."""
    try:
        from app.ui.replenishment_admin import render_replenishment_admin

        for course_id, inventory_rows in _compute_admin_status_snapshot(database_path):
            render_replenishment_admin(course_id, inventory_rows)
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

    # Disabled by default (see AppSettings.admin_status_enabled): the
    # public learner interface must never call into the replenishment
    # inventory scan unless an operator has explicitly opted in.
    if settings.admin_status_enabled:
        render_admin_status(str(settings.database_path))

    # Disabled by default (see AppSettings.maintenance_mode). An operator
    # sets QUIZ_MAINTENANCE_MODE explicitly during a migration deployment
    # (see RUNBOOK_course_id_migration.md) to block learner writes while
    # still allowing the settings/catalogue load above -- basic startup
    # diagnostics -- to run.
    if settings.maintenance_mode:
        st.title("Adaptive Quiz")
        st.warning(
            "The quiz service is temporarily in maintenance mode for a "
            "scheduled database migration. Please try again shortly."
        )
        return

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
        try:
            controller = catalogue.resolve_active(course_id)
        except BootstrapError as error:
            st.error(str(error))
            return
        activate_course(controller, session, course_id)
        st.rerun()

    try:
        controller = catalogue.resolve_active(session.course_id)
    except BootstrapError as error:
        st.error(str(error))
        return

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
