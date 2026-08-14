import streamlit as st


def render_login(*, default_learner_id: str = "") -> str | None:
    st.title("Adaptive Quiz")
    st.write("Enter your learner ID to continue from your saved progress.")

    with st.form("login_form"):
        learner_id = st.text_input("Learner ID", value=default_learner_id)
        submitted = st.form_submit_button("Start")

    if not submitted:
        return None

    learner_id = learner_id.strip()
    if not learner_id:
        st.error("Learner ID is required.")
        return None
    return learner_id


def render_learner_switcher(current_learner_id: str) -> bool:
    with st.sidebar:
        st.subheader("Learner")
        st.caption(f"Signed in as {current_learner_id}")
        return st.button("Switch learner", use_container_width=True)
