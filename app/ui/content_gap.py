"""Render an actionable curriculum content-gap result."""

import streamlit as st

from app.view_models import ContentGapViewModel


def render_content_gap(gap: ContentGapViewModel) -> None:
    st.warning(
        "The next skill is unlocked, but it does not yet have approved practice content."
    )
    st.write(
        f"Completed/mastered skill: {gap.completed_skill_name} "
        f"({gap.completed_skill_id})"
    )
    st.write(
        f"Newly unlocked skill: {gap.newly_unlocked_skill_name} "
        f"({gap.newly_unlocked_skill_id})"
    )
    st.write(
        "Missing approved content: "
        f"{'Yes' if gap.missing_approved_content else 'No'} — "
        f"{gap.newly_unlocked_skill_id}"
    )
    st.write(f"Current mastery: {gap.current_mastery_probability:.0%}")
    st.write(
        f"Prerequisite threshold: {gap.prerequisite_mastery_threshold:.0%}"
    )
