"""Unit equivalence between two option texts, using a constrained unit parser (a strict
"<number> <unit>" regex, never free text) and pint's canonical conversion.

Safety: _QUANTITY_PATTERN only accepts a leading number and a single trailing word of
letters as the unit -- text that doesn't match this exact shape (extra words, symbols,
no unit) is "not_applicable", not passed to pint at all. pint itself never executes
arbitrary code; it looks up the unit token in a fixed registry and raises on anything
unrecognized, which this module treats as "not_applicable" (an unrecognized or
unsupported unit is not an error -- most options naming a unit-like word are not
physical quantities pint's registry knows).
"""

import re

from authoring.review.models import OptionPairEvidence

_QUANTITY_PATTERN = re.compile(r"^\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?)\s*([a-zA-Z]+)\s*$")
_RELATIVE_TOLERANCE = 1e-6

_registry = None


def _unit_registry():
    global _registry
    if _registry is None:
        import pint

        _registry = pint.UnitRegistry()
    return _registry


def _parse_quantity(text: str):
    match = _QUANTITY_PATTERN.match(text.strip())
    if match is None:
        return None
    number_text, unit_text = match.groups()
    try:
        magnitude = float(number_text) if "/" not in number_text else (
            float(number_text.split("/")[0]) / float(number_text.split("/")[1])
        )
    except (ValueError, ZeroDivisionError):
        return None

    registry = _unit_registry()
    try:
        quantity = magnitude * registry(unit_text)
    except Exception:  # noqa: BLE001 -- pint raises several distinct, undocumented-as-a-set
        # exception types (UndefinedUnitError, DimensionalityError, and others) for an
        # unrecognized or malformed unit token; any of them means "not a quantity this
        # registry knows," which is not_applicable, not a detector failure.
        return None
    return quantity


def check_unit_equivalence(
    option_index_a: int, option_index_b: int, text_a: str, text_b: str
) -> OptionPairEvidence:
    quantity_a = _parse_quantity(text_a)
    quantity_b = _parse_quantity(text_b)
    if quantity_a is None or quantity_b is None:
        return OptionPairEvidence(
            option_index_a=option_index_a,
            option_index_b=option_index_b,
            detector="unit_conversion",
            verdict="not_applicable",
            score_or_normalized_form="n/a",
            reason="one or both options are not a strict '<number> <unit>' quantity",
        )

    if not quantity_a.is_compatible_with(quantity_b):
        return OptionPairEvidence(
            option_index_a=option_index_a,
            option_index_b=option_index_b,
            detector="unit_conversion",
            verdict="not_applicable",
            score_or_normalized_form=f"{quantity_a} vs {quantity_b}",
            reason="quantities have incompatible dimensions -- not comparable",
        )

    base_a = quantity_a.to_base_units().magnitude
    base_b = quantity_b.to_base_units().magnitude
    denominator = max(abs(base_a), abs(base_b), 1e-12)
    is_equal = abs(base_a - base_b) / denominator <= _RELATIVE_TOLERANCE
    return OptionPairEvidence(
        option_index_a=option_index_a,
        option_index_b=option_index_b,
        detector="unit_conversion",
        verdict="equivalent" if is_equal else "not_equivalent",
        score_or_normalized_form=f"{quantity_a} vs {quantity_b}",
        reason=(
            "both options convert to the same canonical quantity"
            if is_equal
            else "options convert to different canonical quantities"
        ),
    )
