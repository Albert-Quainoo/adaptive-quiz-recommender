import streamlit as st


def render_login(*, default_learner_id: str = "") -> str | None:
    st.title("Login")
    st.write("Please login with your learner ID")

    with st.form("login_form"):
        learner_id = st.text_input("Learner ID", value=default_learner_id)
        submitted = st.form_submit_button("Start")

    if not submitted:
        return None

    learner_id = learner_id.strip()
    if not learner_id:
        st.error("Learner ID is required!")
        return None
    return learner_id

