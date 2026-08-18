"""authoring/review/equivalence_units.py: constrained unit-conversion equivalence."""

from authoring.review.equivalence_units import check_unit_equivalence


def test_equal_quantities_in_different_units_are_equivalent():
    result = check_unit_equivalence(0, 1, "1.5 kilometers", "1500000 millimeters")
    assert result.verdict == "equivalent"


def test_fractional_quantity_matches_decimal_quantity():
    result = check_unit_equivalence(0, 1, "0.75 cups", "75/100 cups")
    assert result.verdict == "equivalent"


def test_different_quantities_are_not_equivalent():
    result = check_unit_equivalence(0, 1, "1.5 kilometers", "15 kilometers")
    assert result.verdict == "not_equivalent"


def test_incompatible_dimensions_are_not_applicable():
    result = check_unit_equivalence(0, 1, "5 meters", "5 seconds")
    assert result.verdict == "not_applicable"


def test_free_text_is_not_applicable():
    result = check_unit_equivalence(0, 1, "Machine learning", "Graph search")
    assert result.verdict == "not_applicable"


def test_unrecognized_unit_word_is_not_applicable_not_an_error():
    result = check_unit_equivalence(0, 1, "5 frobnicates", "5 meters")
    assert result.verdict == "not_applicable"


def test_extra_words_around_the_quantity_are_not_applicable():
    """The strict '<number> <unit>' pattern deliberately does not attempt a loose
    parse of free text that merely contains a number and a unit-like word."""
    result = check_unit_equivalence(0, 1, "about 5 meters away", "5 meters")
    assert result.verdict == "not_applicable"
