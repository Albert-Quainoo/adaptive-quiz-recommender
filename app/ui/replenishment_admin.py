"""Compact authoring/admin content-status view, separate from the learner quiz.

Read-only: it names the existing CLI command an operator should run next. It
never triggers retrieval or inference itself, so it carries no risk of a
learner request ever waiting on either.
"""

import streamlit as st

from authoring.replenishment.inventory import SkillInventory

_HEALTHY_READINESS = {"ready", "template_ready"}


def _next_step(row: SkillInventory) -> str:
    if row.readiness in ("taxonomy_only", "content_exhausted"):
        return "run `python -m authoring.replenishment.cli scan` to enqueue replenishment"
    if row.readiness == "retrieval_pending":
        return "worker is retrieving references (`python -m authoring.replenishment.cli worker --once`)"
    if row.readiness == "reference_review":
        return (
            f"approve references ({row.pending_reference_candidates} pending): "
            "`python -m authoring.retrieval.cli approve <candidate_id> --reviewer you`"
        )
    if row.readiness == "generation_pending":
        return "worker is generating questions (`python -m authoring.replenishment.cli worker --once`)"
    if row.readiness == "question_review":
        return (
            f"approve questions ({row.pending_generated_questions} pending): "
            "`python -m scripts.review_grounded_batch ... approve QUESTION_ID REVISION_ID`"
        )
    if row.readiness == "replenishment_failed" and row.latest_job is not None:
        return (
            f"job failed [{row.latest_job.error_code}]: {row.latest_job.error_message} -- "
            "fix the cause, then `python -m authoring.replenishment.cli worker --once`"
        )
    return "no action needed"


def render_replenishment_admin(course_id: str, rows: list[SkillInventory]) -> None:
    attention = [row for row in rows if row.readiness not in _HEALTHY_READINESS]

    with st.sidebar.expander(f"Content status ({course_id})", expanded=False):
        if not attention:
            st.caption("Every active skill has enough approved content.")
            return

        for row in sorted(attention, key=lambda item: item.skill_id):
            stage = row.latest_job.job_type if row.latest_job is not None else "-"
            st.markdown(f"**{row.skill_id}** -- {row.readiness}")
            st.caption(
                f"approved={row.total_approved_items}  "
                f"pending_refs={row.pending_reference_candidates}  "
                f"pending_qs={row.pending_generated_questions}  job_stage={stage}"
            )
            st.caption(_next_step(row))
