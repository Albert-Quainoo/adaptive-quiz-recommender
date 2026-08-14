"""Compact, terminal-renderable review card for a human reviewer.

Displays only structured evidence -- recommendation, risk, passed/blocking checks,
warnings, the reviewer's independent answer, grounding evidence, and any proposed
revision. Never displays hidden model reasoning, since none of that ever reaches this
module: AutomatedReviewReport itself carries no raw model text.
"""

from api.schemas import QuizQuestion
from authoring.grounded_review import CurationItem
from authoring.review.models import AutomatedReviewReport

# Sort key: lower is shown first when prioritizing review attention -- low-risk items
# recommended for approval surface before anything needing revision or full review.
_RECOMMENDATION_PRIORITY = {
    "recommend_human_approval": 0,
    "propose_revision": 1,
    "require_full_human_review": 2,
    "reject": 3,
}
_RISK_PRIORITY = {"low": 0, "medium": 1, "high": 2, "critical": 3}


def review_priority(report: AutomatedReviewReport | None) -> tuple[int, int]:
    if report is None:
        return (_RECOMMENDATION_PRIORITY["require_full_human_review"], _RISK_PRIORITY["critical"])
    return (
        _RECOMMENDATION_PRIORITY.get(report.recommendation, 4),
        _RISK_PRIORITY.get(report.risk_level, 4),
    )


def format_review_card(
    question: QuizQuestion,
    item: CurationItem,
    report: AutomatedReviewReport | None,
    *,
    supporting_passages: dict[str, str] | None = None,
) -> str:
    lines = [
        f"=== {item.original_question_id} ({item.intent_id} / {item.skill_id}) ===",
        f"Question: {question.question}",
        "Options:",
        *[f"  - {option}" for option in question.options],
        f"Declared answer: {question.correct_answer}",
        f"Explanation: {question.explanation}",
    ]

    supporting_passages = supporting_passages or {}
    if report is None:
        lines.append("Automated review: not yet available.")
        return "\n".join(lines)

    lines.extend(
        [
            "",
            f"Automated recommendation: {report.recommendation}",
            f"Risk: {report.risk_level} (score={report.risk_score:.2f})",
            f"Reviewer: {report.reviewer_model_id}@{report.reviewer_model_revision} "
            f"prompt={report.reviewer_prompt_version} policy={report.review_policy_version}",
        ]
    )

    passed_checks = [check.code for check in report.deterministic_checks.checks if check.passed]
    if passed_checks:
        lines.append(f"Passed checks: {', '.join(passed_checks)}")
    if report.blocking_reasons:
        lines.append("Blocking reasons:")
        lines.extend(f"  - {reason}" for reason in report.blocking_reasons)
    if report.warnings:
        lines.append("Warnings:")
        lines.extend(f"  - {warning}" for warning in report.warnings)

    if report.answer_assessment is not None:
        no_defensible_note = (
            " -- no option is defensible" if report.answer_assessment.no_defensible_option else ""
        )
        lines.append(
            "Reviewer's selected answer: "
            f"{report.answer_assessment.selected_option_text}{no_defensible_note} "
            f"(matches declared: {report.answer_assessment.matches_declared_answer}, "
            f"confidence={report.answer_assessment.answer_confidence:.2f})"
        )
    if report.grounding_assessment is not None:
        lines.append(
            f"Grounding: grounded={report.grounding_assessment.grounded} "
            f"confidence={report.grounding_assessment.grounding_confidence:.2f}"
        )
        for reference_id in report.grounding_assessment.supporting_reference_ids:
            passage = supporting_passages.get(reference_id)
            lines.append(f"  - [{reference_id}] {passage or '(passage not supplied)'}")

    pending_revisions = [
        revision for revision in item.revisions if revision.final_review_status == "pending"
    ]
    if pending_revisions:
        latest = pending_revisions[-1]
        lines.extend(
            [
                "",
                f"Proposed revision ({latest.revision_id}) by {latest.editor}:",
                f"  Question: {latest.question.question}",
                f"  Correct answer: {latest.question.correct_answer}",
                f"  Review note: {latest.review_note}",
            ]
        )

    return "\n".join(lines)
