import streamlit as st

from app.view_models import QuestionViewModel


def render_question(
    question: QuestionViewModel,
    *,
    selected_option_id: str | None = None,
    disabled: bool = False,
) -> tuple[str | None, bool]:
    st.caption(f"{question.topic} · {question.concept} · {question.difficulty}")
    # st.subheader renders as a single markdown heading, where a lone "\n"
    # collapses to a space (standard Markdown line-break rules) -- multi-line
    # stems (e.g. a matrix written across several lines) would otherwise
    # flatten onto one line. "  \n" is markdown's hard line break.
    question_text = question.question_text.replace("\n", "  \n")
    st.markdown(f"### {question_text}")
    option_ids = [option.option_id for option in question.options]
    selected_index = (
        option_ids.index(selected_option_id)
        if selected_option_id in option_ids
        else None
    )
    with st.form(f"question_form_{question.presentation_id}"):
        selected = st.radio(
            "Options",
            options=option_ids,
            format_func=lambda option_id: next(
                option.text
                for option in question.options
                if option.option_id == option_id
            ),
            index=selected_index,
            label_visibility="collapsed",
            disabled=disabled,
        )
        submitted = st.form_submit_button("Submit", disabled=disabled)
    return selected, submitted
