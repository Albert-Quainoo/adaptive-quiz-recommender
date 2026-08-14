"""Deterministic structural checks for generated and revised MCQs.

These checks catch recognizable defects; they do not establish factual truth.
"""

import re

from pydantic import BaseModel

from api.schemas import QuizQuestion


class QualityIssue(BaseModel):
    code: str
    message: str


def _normalise(text: str) -> str:
    return " ".join(text.casefold().split())


# Matches text ending in a list conjunction (", and a ", "and an ", "or the ", etc.)
# immediately before the answer: the stem is enumerating candidate terms (a common,
# legitimate "which of these is missing/excluded" construction), not asserting the
# answer outright.
_ENUMERATION_PRECEDER = re.compile(r"(?:,|\band\b|\bor\b)\s*(?:a|an|the)?\s*$")


def generic_quality_issues(
    question: QuizQuestion,
    *,
    intent_id: str,
    accepted_intent_ids: set[str] | None = None,
) -> list[QualityIssue]:
    """Find deterministic warning signs; human review remains the factual gate."""
    issues: list[QualityIssue] = []
    stem = question.question.strip()
    answer = _normalise(question.correct_answer)
    normalised_stem = _normalise(stem)
    normalised_options = [_normalise(option) for option in question.options]
    if len(answer) >= 4 and answer in normalised_stem:
        occurrences = [match.start() for match in re.finditer(re.escape(answer), normalised_stem)]
        if any(
            not _ENUMERATION_PRECEDER.search(normalised_stem[:start]) for start in occurrences
        ):
            issues.append(
                QualityIssue(
                    code="answer_revealed",
                    message="stem contains the correct answer verbatim",
                )
            )
    imperative = re.match(
        r"^(choose|select|identify|determine|calculate|compute|formulate|explain)\b",
        stem,
        re.I,
    )
    if not stem.endswith("?") and not imperative:
        issues.append(
            QualityIssue(
                code="non_question_stem",
                message="stem is neither a question nor a clear MCQ directive",
            )
        )
    if accepted_intent_ids is not None and intent_id in accepted_intent_ids:
        issues.append(
            QualityIssue(
                code="repeated_intent",
                message="intent was already accepted in this skill batch",
            )
        )
    if sum(option == question.correct_answer for option in question.options) != 1:
        issues.append(
            QualityIssue(
                code="correct_answer_match",
                message="correct_answer must match exactly one option",
            )
        )
    if len(normalised_options) != len(set(normalised_options)):
        issues.append(
            QualityIssue(
                code="duplicate_options",
                message="options duplicate after normalization",
            )
        )
    count_words = {"four": 4, "five": 5, "six": 6}
    counts = {
        int(value) if value.isdigit() else count_words[value.casefold()]
        for value in re.findall(
            r"\b(4|5|6|four|five|six)\s+(?:components|elements)\b",
            " ".join((stem, question.correct_answer, question.explanation)),
            re.I,
        )
    }
    if len(counts) > 1:
        issues.append(
            QualityIssue(
                code="contradictory_counts",
                message="question fields assert contradictory component counts",
            )
        )
    explanation = _normalise(question.explanation)
    if re.search(
        rf"{re.escape(answer)}\s+(?:is|are)\s+(?:incorrect|false|not correct)",
        explanation,
    ):
        issues.append(
            QualityIssue(
                code="explanation_disagreement",
                message="explanation negates the selected option",
            )
        )
    return issues
