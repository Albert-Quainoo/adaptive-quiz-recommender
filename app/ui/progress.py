import streamlit as st

from app.view_models import ProgressViewModel


def render_progress(progress: ProgressViewModel) -> None:
    with st.sidebar:
        st.write(f"Learner: {progress.learner_id}")
        st.write(f"Skill: {progress.skill_name}")
        st.progress(
            progress.mastery_probability,
            text=f"Mastery: {progress.mastery_probability:.0%}",
        )
        st.caption(
            f"Attempts: {progress.attempt_count}  ·  Difficulty: {progress.difficulty}"
        )
        st.info(progress.readable_recommendation_reason)
