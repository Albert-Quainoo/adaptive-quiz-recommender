import streamlit as st

from app.multi_course import ActiveCourse, CourseCatalogController, UnavailableCourse

BEING_PREPARED_MESSAGE = "This course is being prepared and is not available for practice yet."


def render_course_selector(
    catalogue: CourseCatalogController, *, default_query: str = ""
) -> str | None:
    st.title("Choose a course")

    active_courses = catalogue.list_active_courses()
    if active_courses:
        st.caption(
            "Available now: " + ", ".join(title for _, title in active_courses)
        )
    preparing_courses = catalogue.list_preparing_courses()
    if preparing_courses:
        st.caption(
            "Coming soon: " + ", ".join(title for _, title in preparing_courses)
        )

    with st.form("course_selector_form"):
        query = st.text_input("Which course?", value=default_query)
        submitted = st.form_submit_button("Continue")

    if not submitted:
        return None

    query = query.strip()
    if not query:
        st.error("Enter a course name to continue.")
        return None

    result = catalogue.resolve_for_learner(query)
    if isinstance(result, ActiveCourse):
        return result.course_id
    if isinstance(result, UnavailableCourse):
        st.info(BEING_PREPARED_MESSAGE)
        return None
    st.error(f"{query!r} is not a recognized course.")
    return None
