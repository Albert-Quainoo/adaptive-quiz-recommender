"""Deterministic, application-side fallback templates for question archetypes where
free-form model generation has repeatedly failed on construction, not on
infrastructure (see authoring/grounded_batch.py's generate_batch and
authoring/replenishment/worker.py's generate-questions stage).

Each template is a pure function, keyed by intent_id in DETERMINISTIC_TEMPLATES.
A template never calls a model and never invents pedagogy -- it only assembles a
candidate deterministically from a reviewed blueprint intent and its approved
reference material, using trusted application-side arithmetic. It is never
exempt from the checks a live model attempt would face: generate_batch runs a
template's output through the identical validate_question/validate_pilot_question/
generic_quality_issues gate before accepting it, and everything downstream
(deterministic review checks, the automated reviewer, human approval, promotion)
treats it exactly like a model-generated candidate. The only distinguishing trace
is provenance: PendingQuestion.generation_method records "deterministic_template",
and model_id/model_revision are set to this module's own identity rather than a
live model's, so a template-authored item is never mistaken for one the model
actually produced.
"""

import random
from collections.abc import Callable

from api.schemas import difficulty_level
from authoring.grounded_batch import IntentQuestion
from authoring.question_intents import QuestionIntent
from taxonomy.schemas import ReferenceProvenance, SkillDefinition

TemplateFn = Callable[
    [QuestionIntent, SkillDefinition, list[ReferenceProvenance], int, difficulty_level],
    IntentQuestion,
]


def _random_matrix(rng: random.Random, size: int = 3, low: int = -6, high: int = 6) -> list[list[int]]:
    return [[rng.randint(low, high) for _ in range(size)] for _ in range(size)]


def _minor(matrix: list[list[int]], row: int, col: int) -> list[list[int]]:
    return [entries[:col] + entries[col + 1 :] for index, entries in enumerate(matrix) if index != row]


def _det2(matrix: list[list[int]]) -> int:
    return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]


def determinant(matrix: list[list[int]]) -> int:
    """Trusted, application-side determinant via recursive cofactor expansion
    along the first row of every square submatrix -- never delegated to a model."""
    size = len(matrix)
    if size == 1:
        return matrix[0][0]
    if size == 2:
        return _det2(matrix)
    total = 0
    for col in range(size):
        sign = 1 if col % 2 == 0 else -1
        total += sign * matrix[0][col] * determinant(_minor(matrix, 0, col))
    return total


def _format_matrix(matrix: list[list[int]]) -> str:
    return "[" + ", ".join("[" + ", ".join(str(value) for value in row) + "]" for row in matrix) + "]"


def generate_determinant_question(
    intent: QuestionIntent,
    skill: SkillDefinition,
    references: list[ReferenceProvenance],
    seed: int,
    difficulty: difficulty_level,
) -> IntentQuestion:
    """Deterministically construct a cofactor-expansion determinant MCQ for a 3x3
    integer matrix.

    The correct answer is the trusted determinant() above, computed by
    cofactor expansion along the first row. The three distractors are the
    controlled mathematical error patterns the blueprint's own
    expected_misconception_or_distractor_strategy names: forgetting the
    cofactor sign alternation, dropping one expansion term, and an arithmetic
    slip on one term's 2x2 minor (ad + bc mistaken for ad - bc). A matrix is
    redrawn (deterministically, from the same seeded RNG) whenever these four
    values are not all distinct, bounded at 50 attempts.
    """
    rng = random.Random(seed)
    matrix = candidates = correct = None
    for _ in range(50):
        matrix = _random_matrix(rng)
        col_signs = [1 if col % 2 == 0 else -1 for col in range(3)]
        minors = [_minor(matrix, 0, col) for col in range(3)]
        unsigned_terms = [matrix[0][col] * _det2(minors[col]) for col in range(3)]
        correct = sum(sign * term for sign, term in zip(col_signs, unsigned_terms))

        # Common mistake 1: every term added, the cofactor sign alternation forgotten.
        unsigned_sum = sum(unsigned_terms)
        # Common mistake 2: the last expansion term dropped entirely.
        dropped_last_term = sum(
            sign * term for sign, term in zip(col_signs[:2], unsigned_terms[:2])
        )
        # Common mistake 3: the first term's 2x2 minor determinant computed as
        # ad + bc instead of ad - bc.
        slipped_minor = minors[0][0][0] * minors[0][1][1] + minors[0][0][1] * minors[0][1][0]
        arithmetic_slip = (
            col_signs[0] * matrix[0][0] * slipped_minor
            + col_signs[1] * unsigned_terms[1]
            + col_signs[2] * unsigned_terms[2]
        )

        candidates = [correct, unsigned_sum, dropped_last_term, arithmetic_slip]
        if len(set(candidates)) == 4:
            break
    else:
        raise ValueError("could not construct four distinct determinant options after 50 attempts")

    options = [str(value) for value in candidates]
    rng.shuffle(options)

    matrix_text = _format_matrix(matrix)
    stem = (
        f"Compute the determinant of the matrix A = {matrix_text} "
        "by cofactor expansion along the first row."
    )
    explanation = (
        f"Expanding along the first row of A: det(A) = "
        f"({matrix[0][0]})*det({_format_matrix(minors[0])}) "
        f"- ({matrix[0][1]})*det({_format_matrix(minors[1])}) "
        f"+ ({matrix[0][2]})*det({_format_matrix(minors[2])}) = "
        f"({unsigned_terms[0]}) - ({unsigned_terms[1]}) + ({unsigned_terms[2]}) = {correct}."
    )

    return IntentQuestion(
        question=stem,
        options=options,
        correct_answer=str(correct),
        explanation=explanation,
        concept=", ".join(intent.required_concepts) or skill.name,
        difficulty=difficulty,
        intent_id=intent.intent_id,
    )


DETERMINISTIC_TEMPLATES: dict[str, TemplateFn] = {
    "LA-DET-01-INT-05": generate_determinant_question,
}
