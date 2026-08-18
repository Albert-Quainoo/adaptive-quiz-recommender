"""authoring/review/equivalence_math.py: constrained symbolic math equivalence."""

from authoring.review.equivalence_math import check_math_equivalence


def test_equal_fraction_and_decimal_are_equivalent():
    result = check_math_equivalence(0, 1, "0.75", "3/4")
    assert result.verdict == "equivalent"


def test_different_numbers_are_not_equivalent():
    result = check_math_equivalence(0, 1, "0.75", "0.57")
    assert result.verdict == "not_equivalent"


def test_free_text_is_not_applicable():
    result = check_math_equivalence(0, 1, "Machine learning", "Graph search")
    assert result.verdict == "not_applicable"


def test_text_with_units_is_not_applicable_here():
    """Units are the unit_conversion detector's job, not symbolic_math's -- a bare
    numeric expression parser must not also try to interpret "cups"/"meters" etc."""
    result = check_math_equivalence(0, 1, "0.75 cups", "75/100 cups")
    assert result.verdict == "not_applicable"


def test_malicious_looking_text_is_rejected_by_the_regex_gate_not_evaluated():
    """No identifier can ever reach sympy.sympify -- the regex only allows digits,
    '.', '/', '+', '-', '*', '(', ')', and whitespace."""
    for hostile in ["__import__('os')", "open('/etc/passwd')", "1; DROP TABLE x", "os.system('ls')"]:
        result = check_math_equivalence(0, 1, hostile, "1")
        assert result.verdict == "not_applicable"


def test_overlong_expression_is_rejected_before_parsing():
    long_expression = "1+" * 100 + "1"
    result = check_math_equivalence(0, 1, long_expression, "1")
    assert result.verdict == "not_applicable"


def test_option_pair_evidence_requires_distinct_indices():
    import pytest
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        check_math_equivalence(0, 0, "1", "1")
