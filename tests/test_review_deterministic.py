"""Pure deterministic-check tests -- no reviewer, no I/O."""

from authoring.review.deterministic import run_deterministic_checks
from tests.review_fixtures import APPROVED_REFERENCES, CORRECTED_CANDIDATE, INTENT, SKILL


def test_valid_candidate_has_no_blocking_failures():
    checks = run_deterministic_checks(CORRECTED_CANDIDATE, SKILL, INTENT, APPROVED_REFERENCES)
    assert checks.blocking_failures == []
    assert checks.all_passed


def test_unapproved_reference_is_blocking():
    candidate = CORRECTED_CANDIDATE.model_copy(update={"reference_ids": ["not-approved-ref"]})
    checks = run_deterministic_checks(candidate, SKILL, INTENT, APPROVED_REFERENCES)
    codes = {check.code for check in checks.blocking_failures}
    assert "approved_references_only" in codes


def test_duplicate_item_id_is_blocking():
    checks = run_deterministic_checks(
        CORRECTED_CANDIDATE,
        SKILL,
        INTENT,
        APPROVED_REFERENCES,
        existing_item_ids={CORRECTED_CANDIDATE.question_id},
    )
    codes = {check.code for check in checks.blocking_failures}
    assert "no_duplicate_item_id" in codes


def test_exact_duplicate_question_text_is_blocking():
    from authoring.grounded_batch import normalise_question

    checks = run_deterministic_checks(
        CORRECTED_CANDIDATE,
        SKILL,
        INTENT,
        APPROVED_REFERENCES,
        existing_question_texts={normalise_question(CORRECTED_CANDIDATE.question.question)},
    )
    codes = {check.code for check in checks.blocking_failures}
    assert "no_exact_duplicate_question" in codes


def test_exact_duplicate_options_after_normalization_is_blocking():
    """Preserved deterministic exact-duplicate detection: this is the complement to
    duplicate_option_pairs (the reviewer-reported *semantic* signal) -- normalization
    (casefold + whitespace collapse) is enough to catch this case without any model
    call at all."""
    candidate = CORRECTED_CANDIDATE.model_copy(deep=True)
    candidate.question.options[1] = candidate.question.options[0].upper()
    checks = run_deterministic_checks(candidate, SKILL, INTENT, APPROVED_REFERENCES)
    codes = {check.code for check in checks.blocking_failures}
    assert "options_distinct_after_normalization" in codes


def test_unresolved_placeholder_is_blocking():
    candidate = CORRECTED_CANDIDATE.model_copy(deep=True)
    candidate.question.explanation = "TODO: finish this explanation."
    checks = run_deterministic_checks(candidate, SKILL, INTENT, APPROVED_REFERENCES)
    codes = {check.code for check in checks.blocking_failures}
    assert "no_unresolved_placeholders" in codes


def test_mismatched_skill_is_blocking():
    candidate = CORRECTED_CANDIDATE.model_copy(update={"skill_id": "AI-SRC-09"})
    checks = run_deterministic_checks(candidate, SKILL, INTENT, APPROVED_REFERENCES)
    codes = {check.code for check in checks.blocking_failures}
    assert "skill_matches_candidate" in codes


def test_answer_position_skew_is_a_non_blocking_warning():
    checks = run_deterministic_checks(
        CORRECTED_CANDIDATE,
        SKILL,
        INTENT,
        APPROVED_REFERENCES,
        accepted_answer_positions=[0] * 8,
    )
    skew_check = next(check for check in checks.checks if check.code == "answer_position_distribution")
    assert skew_check.blocking is False
    assert skew_check.passed is False
