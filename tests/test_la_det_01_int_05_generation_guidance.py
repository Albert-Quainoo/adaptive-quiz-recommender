"""Regression test for the LA-DET-01-INT-05 blueprint fix.

The pilot's live Modal generation attempts for LA-DET-01-INT-05 (cofactor-
expansion determinant computation) failed all 3/3 attempts with distinct
construction defects: a mislabeled/arithmetically-wrong correct_answer, ANSI
escape codes breaking JSON on a non-square (undefined-determinant) matrix, and
verbatim-duplicate options. The fix adds intent-specific generation_constraints
and prohibited_ambiguity_patterns to this one blueprint intent (not the shared
system prompt) so the guidance reaches the model only for this archetype.
build_grounded_quiz_messages JSON-dumps the intent object verbatim into the
prompt, so asserting on the loaded intent's fields is equivalent to asserting
on prompt content.
"""

from authoring.question_intents import intents_by_skill, load_blueprint_for_batch

BATCH_ID = "grounded-linear-algebra-v1"
SKILL_ID = "LA-DET-01"
INTENT_ID = "LA-DET-01-INT-05"


def _target_intent():
    blueprint = load_blueprint_for_batch(BATCH_ID)
    pool = intents_by_skill(blueprint)[SKILL_ID]
    return next(intent for intent in pool if intent.intent_id == INTENT_ID)


def test_la_det_01_int_05_forbids_non_square_matrices():
    intent = _target_intent()
    assert any(
        "square" in constraint.lower() and "non-square" in constraint.lower()
        for constraint in intent.generation_constraints
    )


def test_la_det_01_int_05_requires_independent_arithmetic_verification_and_unlabeled_answer():
    intent = _target_intent()
    assert any(
        "recompute the determinant" in constraint.lower()
        for constraint in intent.generation_constraints
    )
    assert any(
        "correct_answer:" in constraint
        for constraint in intent.generation_constraints
    )


def test_la_det_01_int_05_forbids_duplicate_options():
    intent = _target_intent()
    assert any(
        "textually identical" in constraint.lower()
        for constraint in intent.generation_constraints
    )


def test_la_det_01_int_05_forbids_ansi_and_markdown_markup():
    intent = _target_intent()
    assert any(
        "ansi escape codes" in pattern.lower()
        for pattern in intent.prohibited_ambiguity_patterns
    )


def test_other_la_det_01_intents_are_unchanged():
    """Only LA-DET-01-INT-05 was touched -- every other slot in this pool keeps
    its original two generation_constraints and one prohibited_ambiguity_pattern."""
    blueprint = load_blueprint_for_batch(BATCH_ID)
    pool = intents_by_skill(blueprint)[SKILL_ID]
    others = [intent for intent in pool if intent.intent_id != INTENT_ID]
    assert others, "expected sibling LA-DET-01 intents to exist"
    for intent in others:
        assert len(intent.generation_constraints) == 2
