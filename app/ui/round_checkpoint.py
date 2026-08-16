"""Round-checkpoint screens: the learner's explicit choice after completing
a round, instead of a hard session-ending error."""

import streamlit as st


def render_round_checkpoint(round_number: int) -> str | None:
    st.title("Round complete")
    st.write(f"You finished round {round_number}. What would you like to do next?")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Continue", use_container_width=True, type="primary"):
            return "continue"
        if st.button("Focus on weak areas", use_container_width=True):
            return "focus_weak"
    with col2:
        if st.button("Pause", use_container_width=True):
            return "pause"
        if st.button("Finish", use_container_width=True):
            return "finish"
    return None


def render_paused() -> bool:
    st.title("Paused")
    st.write("Your progress is saved. Come back anytime to pick up where you left off.")
    return st.button("Resume", use_container_width=True, type="primary")


def render_finished() -> bool:
    st.title("Session finished")
    st.write("Great work today. Your progress is saved.")
    return st.button("Start a new session", use_container_width=True, type="primary")
