"""authoring/review/equivalence_nli.py: pairwise semantic-equivalence detector logic,
using FakeNliScorer throughout -- never touches ONNX Runtime or the network. Real-model
integration is covered separately by tests/test_equivalence_gate_acceptance.py, which
skips itself when the pinned model isn't reachable.
"""

from authoring.review.equivalence_nli import EntailmentScores, FakeNliScorer, check_semantic_equivalence


def _scorer_for(stem: str, a: str, b: str, *, forward: float, backward: float) -> FakeNliScorer:
    premise_a = f"Given the question: {stem} Claim: {a}"
    premise_b = f"Given the question: {stem} Claim: {b}"
    return FakeNliScorer(
        {
            (premise_a, premise_b): EntailmentScores(contradiction=0.0, entailment=forward, neutral=1 - forward),
            (premise_b, premise_a): EntailmentScores(contradiction=0.0, entailment=backward, neutral=1 - backward),
        }
    )


def test_high_bidirectional_entailment_is_equivalent():
    scorer = _scorer_for("stem", "a", "b", forward=0.9, backward=0.9)
    result = check_semantic_equivalence(0, 1, "stem", "a", "b", threshold=0.5, scorer=scorer)
    assert result.verdict == "equivalent"


def test_low_entailment_in_either_direction_is_not_equivalent():
    scorer = _scorer_for("stem", "a", "b", forward=0.9, backward=0.1)
    result = check_semantic_equivalence(0, 1, "stem", "a", "b", threshold=0.5, scorer=scorer)
    assert result.verdict == "not_equivalent"


def test_threshold_boundary_is_inclusive():
    scorer = _scorer_for("stem", "a", "b", forward=0.5, backward=0.5)
    result = check_semantic_equivalence(0, 1, "stem", "a", "b", threshold=0.5, scorer=scorer)
    assert result.verdict == "equivalent"


def test_default_fake_scorer_score_is_never_equivalent():
    """FakeNliScorer's default (unconfigured) response is low-entailment/high-neutral
    -- the property tests/conftest.py's no_default_nli_model fixture relies on to
    never spuriously escalate an existing test that knows nothing about this gate."""
    scorer = FakeNliScorer()
    result = check_semantic_equivalence(0, 1, "any stem", "any a", "any b", threshold=0.05, scorer=scorer)
    assert result.verdict == "not_equivalent"
