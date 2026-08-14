"""Pure, deterministic structural checks for a generated question candidate.

Nothing here makes a network or model call. A blocking failure here is what lets the
review service skip the reviewer model entirely -- see authoring/review/service.py.
These checks establish structural soundness, not factual truth; they extend (and call,
rather than duplicate) authoring.question_quality.generic_quality_issues.
"""

import re
from collections.abc import Iterable

from authoring.grounded_batch import PendingQuestion, normalise_question
from authoring.question_intents import QuestionIntent
from authoring.question_quality import generic_quality_issues
from authoring.review.models import DeterministicCheckResult, DeterministicChecks
from taxonomy.schemas import ReferenceProvenance, SkillDefinition

MAX_QUESTION_CHARS = 600
MAX_OPTION_CHARS = 250
MAX_EXPLANATION_CHARS = 900

_PLACEHOLDER_PATTERN = re.compile(
    r"\{\{.*?\}\}|\[insert[^\]]*\]|\bTODO\b|\bFIXME\b|\bXXX\b|<PLACEHOLDER>",
    re.IGNORECASE,
)

# Below this Jaccard overlap of normalized word sets, two stems are considered
# unrelated; above it, near-duplicate enough to warn a human before promotion.
NEAR_DUPLICATE_JACCARD_THRESHOLD = 0.85


def _check(
    code: str, passed: bool, *, blocking: bool, passed_message: str, failed_message: str
) -> DeterministicCheckResult:
    """message always describes the actual outcome -- what was true (passed) or what
    was violated (failed) -- never a restatement of the requirement regardless of
    outcome."""
    return DeterministicCheckResult(
        code=code,
        passed=passed,
        message=passed_message if passed else failed_message,
        blocking=blocking,
    )


def _word_set(text: str) -> frozenset[str]:
    return frozenset(normalise_question(text).split())


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def run_deterministic_checks(
    candidate: PendingQuestion,
    skill: SkillDefinition,
    intent: QuestionIntent,
    approved_references: Iterable[ReferenceProvenance],
    *,
    existing_item_ids: Iterable[str] = (),
    existing_question_texts: Iterable[str] = (),
    accepted_answer_positions: Iterable[int] = (),
) -> DeterministicChecks:
    question = candidate.question
    approved_reference_ids = {reference.reference_id for reference in approved_references}
    existing_item_ids = set(existing_item_ids)
    existing_question_texts = set(existing_question_texts)
    checks: list[DeterministicCheckResult] = []

    checks.append(
        _check(
            "exactly_four_options",
            len(question.options) == 4,
            blocking=True,
            passed_message="question has exactly four options",
            failed_message=f"question has {len(question.options)} option(s), expected exactly four",
        )
    )

    normalized_options = [normalise_question(option) for option in question.options]
    duplicate_options = sorted(
        {option for option in normalized_options if normalized_options.count(option) > 1}
    )
    checks.append(
        _check(
            "options_distinct_after_normalization",
            len(normalized_options) == len(set(normalized_options)),
            blocking=True,
            passed_message="all options are distinct after normalization",
            failed_message=(
                "options are not distinct after normalization, duplicates: "
                + ", ".join(duplicate_options)
            ),
        )
    )

    checks.append(
        _check(
            "correct_answer_matches_one_option",
            question.correct_answer in question.options,
            blocking=True,
            passed_message="correct_answer exactly equals one option",
            failed_message=f"correct_answer {question.correct_answer!r} does not exactly equal any option",
        )
    )

    checks.append(
        _check(
            "explanation_non_empty",
            bool(question.explanation.strip()),
            blocking=True,
            passed_message="explanation is non-empty",
            failed_message="explanation is empty",
        )
    )

    checks.append(
        _check(
            "skill_matches_candidate",
            skill.skill_id == candidate.skill_id,
            blocking=True,
            passed_message=f"candidate skill_id equals {skill.skill_id}",
            failed_message=(
                f"candidate skill_id {candidate.skill_id!r} does not equal {skill.skill_id!r}"
            ),
        )
    )

    checks.append(
        _check(
            "intent_matches_candidate",
            intent.intent_id == candidate.intent_id and intent.skill_id == candidate.skill_id,
            blocking=True,
            passed_message=f"candidate intent_id equals {intent.intent_id}",
            failed_message=(
                f"candidate intent_id {candidate.intent_id!r}/skill_id {candidate.skill_id!r} "
                f"does not equal {intent.intent_id!r}/{intent.skill_id!r}"
            ),
        )
    )

    checks.append(
        _check(
            "difficulty_allowed",
            question.difficulty in ("introductory", "intermediate", "advanced"),
            blocking=True,
            passed_message=f"difficulty {question.difficulty!r} is an allowed level",
            failed_message=f"difficulty {question.difficulty!r} is not an allowed level",
        )
    )

    missing_references = [
        reference_id
        for reference_id in candidate.reference_ids
        if reference_id not in approved_reference_ids
    ]
    checks.append(
        _check(
            "approved_references_only",
            not missing_references and bool(candidate.reference_ids),
            blocking=True,
            passed_message="every cited reference_id is an approved reference",
            failed_message=(
                "candidate cites unapproved reference_id(s): " + ", ".join(missing_references)
                if missing_references
                else "candidate cites no reference_id"
            ),
        )
    )

    provenance_fields = {
        "model_id": candidate.model_id,
        "model_revision": candidate.model_revision,
        "prompt_version": candidate.prompt_version,
        "prompt_hash": candidate.prompt_hash,
    }
    missing_provenance = sorted(name for name, value in provenance_fields.items() if not value.strip())
    checks.append(
        _check(
            "provenance_present",
            not missing_provenance,
            blocking=True,
            passed_message="candidate carries complete model and prompt provenance",
            failed_message="candidate is missing provenance field(s): " + ", ".join(missing_provenance),
        )
    )

    checks.append(
        _check(
            "no_duplicate_item_id",
            candidate.question_id not in existing_item_ids,
            blocking=True,
            passed_message=f"question_id {candidate.question_id} does not already exist",
            failed_message=f"question_id {candidate.question_id} already exists",
        )
    )

    normalized_stem = normalise_question(question.question)
    checks.append(
        _check(
            "no_exact_duplicate_question",
            normalized_stem not in existing_question_texts,
            blocking=True,
            passed_message="question text does not duplicate an existing item exactly",
            failed_message="question text duplicates an existing item exactly",
        )
    )

    stem_words = _word_set(question.question)
    near_duplicate = next(
        (
            existing
            for existing in existing_question_texts
            if existing != normalized_stem
            and _jaccard(stem_words, _word_set(existing)) >= NEAR_DUPLICATE_JACCARD_THRESHOLD
        ),
        None,
    )
    checks.append(
        _check(
            "no_near_duplicate_question",
            near_duplicate is None,
            blocking=True,
            passed_message="question text is not a near-duplicate of an existing item",
            failed_message=(
                f"question text is a near-duplicate (jaccard >= {NEAR_DUPLICATE_JACCARD_THRESHOLD}) "
                "of an existing item"
            ),
        )
    )

    answer_positions = list(accepted_answer_positions)
    if answer_positions and len(answer_positions) >= 8:
        most_common_share = max(answer_positions.count(position) for position in set(answer_positions)) / len(
            answer_positions
        )
        checks.append(
            _check(
                "answer_position_distribution",
                most_common_share <= 0.5,
                blocking=False,
                passed_message=(
                    f"no single answer position exceeds 50% of the batch so far "
                    f"(most common: {most_common_share:.0%})"
                ),
                failed_message=(
                    f"one answer position accounts for {most_common_share:.0%} of the batch so far"
                ),
            )
        )

    placeholder_fields = " ".join((question.question, *question.options, question.explanation))
    checks.append(
        _check(
            "no_unresolved_placeholders",
            not _PLACEHOLDER_PATTERN.search(placeholder_fields),
            blocking=True,
            passed_message="question contains no forbidden formatting or unresolved placeholders",
            failed_message="question contains forbidden formatting or an unresolved placeholder",
        )
    )

    checks.append(
        _check(
            "question_length_within_limit",
            len(question.question) <= MAX_QUESTION_CHARS,
            blocking=True,
            passed_message=f"question text is within the {MAX_QUESTION_CHARS} character limit",
            failed_message=(
                f"question text is {len(question.question)} characters, "
                f"exceeding the {MAX_QUESTION_CHARS} character limit"
            ),
        )
    )
    longest_option = max((len(option) for option in question.options), default=0)
    checks.append(
        _check(
            "option_lengths_within_limit",
            all(len(option) <= MAX_OPTION_CHARS for option in question.options),
            blocking=True,
            passed_message=f"every option is within the {MAX_OPTION_CHARS} character limit",
            failed_message=(
                f"an option is {longest_option} characters, exceeding the "
                f"{MAX_OPTION_CHARS} character limit"
            ),
        )
    )
    checks.append(
        _check(
            "explanation_length_within_limit",
            len(question.explanation) <= MAX_EXPLANATION_CHARS,
            blocking=True,
            passed_message=f"explanation is within the {MAX_EXPLANATION_CHARS} character limit",
            failed_message=(
                f"explanation is {len(question.explanation)} characters, exceeding the "
                f"{MAX_EXPLANATION_CHARS} character limit"
            ),
        )
    )

    quality_issues = generic_quality_issues(question, intent_id=candidate.intent_id)
    for issue in quality_issues:
        checks.append(
            _check(
                f"quality_{issue.code}",
                False,
                blocking=True,
                passed_message="unreachable: only appended when the issue is present",
                failed_message=issue.message,
            )
        )

    return DeterministicChecks(checks=checks)
