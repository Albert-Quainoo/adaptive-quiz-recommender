import streamlit as st

from app.view_models import SubmissionResultViewModel


def render_feedback(
    result: SubmissionResultViewModel, *, next_enabled: bool = True
) -> bool:
    if result.correct:
        st.success("Correct")
    else:
        st.error("Incorrect")
        st.write(f"Correct answer: {result.correct_answer_text}")
    st.write(result.explanation)

    col1, col2 = st.columns(2)
    col1.metric("Previous mastery", f"{result.previous_mastery:.0%}")
    col2.metric(
        "Updated mastery",
        f"{result.updated_mastery:.0%}",
        delta=f"{result.updated_mastery - result.previous_mastery:+.0%}",
    )

    return st.button("Next Question", disabled=not next_enabled)
