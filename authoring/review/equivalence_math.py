"""Symbolic mathematical equivalence between two option texts, using a constrained
SymPy parsing path.

Safety comes from what is allowed to reach the parser, not from anything SymPy itself
enforces: _NUMERIC_EXPRESSION regex only accepts digits, '.', '/', '+', '-', '*', '(',
')', and whitespace, so no identifier, function name, or attribute access can ever
appear in a string this module passes to sympy.sympify -- there is no symbol table to
exploit because there can be no symbols. Text that doesn't match is "not_applicable",
the ordinary case (most options are not bare arithmetic expressions), not an error.

Bounded: the length cap below rejects pathological inputs before parsing; sympify with
rational=True on a short pure-arithmetic string is a bounded, sub-millisecond operation
with no recursion or search proportional to unbounded input.
"""

import re

from authoring.review.models import OptionPairEvidence

_MAX_EXPRESSION_LENGTH = 40
_NUMERIC_EXPRESSION = re.compile(r"^[0-9+\-*/(). ]+$")


def _parse_numeric_expression(text: str):
    stripped = text.strip()
    if not stripped or len(stripped) > _MAX_EXPRESSION_LENGTH:
        return None
    if not _NUMERIC_EXPRESSION.match(stripped):
        return None
    if not any(character.isdigit() for character in stripped):
        return None
    import sympy

    try:
        return sympy.sympify(stripped, rational=True, evaluate=True)
    except (sympy.SympifyError, TypeError, ValueError, RecursionError):
        return None


def check_math_equivalence(
    option_index_a: int, option_index_b: int, text_a: str, text_b: str
) -> OptionPairEvidence:
    value_a = _parse_numeric_expression(text_a)
    value_b = _parse_numeric_expression(text_b)
    if value_a is None or value_b is None:
        return OptionPairEvidence(
            option_index_a=option_index_a,
            option_index_b=option_index_b,
            detector="symbolic_math",
            verdict="not_applicable",
            score_or_normalized_form="n/a",
            reason="one or both options are not bare numeric expressions",
        )

    import sympy

    difference = sympy.simplify(value_a - value_b)
    is_equal = difference == 0
    return OptionPairEvidence(
        option_index_a=option_index_a,
        option_index_b=option_index_b,
        detector="symbolic_math",
        verdict="equivalent" if is_equal else "not_equivalent",
        score_or_normalized_form=f"{sympy.nsimplify(value_a)} vs {sympy.nsimplify(value_b)}",
        reason=(
            "both options simplify to the same numeric value"
            if is_equal
            else "options simplify to different numeric values"
        ),
    )
