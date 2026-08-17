"""authoring/review/equivalence_gate.py: pair orchestration, bounded timeouts,
fail-closed error handling, and escalation-reason extraction. Uses FakeNliScorer
throughout -- never touches the network."""

import time

from authoring.review.equivalence_gate import evaluate_option_equivalence, escalation_reasons
from authoring.review.equivalence_nli import EntailmentScores, FakeNliScorer


def test_four_options_produce_six_pairs_times_three_detectors():
    options = ["1", "2", "3", "4"]
    evidence = evaluate_option_equivalence("stem", options, nli_threshold=0.5, nli_scorer=FakeNliScorer())
    assert len(evidence) == 18  # C(4,2)=6 pairs x 3 detectors
    pairs = {(item.option_index_a, item.option_index_b) for item in evidence}
    assert pairs == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}


def test_math_and_unit_evidence_present_alongside_nli_for_every_pair():
    options = ["0.5", "1/2", "a", "b"]
    evidence = evaluate_option_equivalence("stem", options, nli_threshold=0.5, nli_scorer=FakeNliScorer())
    detectors_for_pair_01 = {item.detector for item in evidence if (item.option_index_a, item.option_index_b) == (0, 1)}
    assert detectors_for_pair_01 == {"symbolic_math", "unit_conversion", "nli_semantic"}


def test_escalation_reasons_only_includes_equivalent_and_error():
    options = ["0.75", "0.75", "x", "y"]  # pair (0,1): math-equivalent
    evidence = evaluate_option_equivalence("stem", options, nli_threshold=0.99, nli_scorer=FakeNliScorer())
    reasons = escalation_reasons(evidence)
    assert any("symbolic_math" in reason for reason in reasons)
    verdicts_in_reasons = {reason.split(" ", 1)[0] for reason in reasons}
    assert "not_applicable" not in verdicts_in_reasons
    assert "not_equivalent" not in verdicts_in_reasons


def test_no_signal_produces_no_escalation():
    options = ["Machine learning", "Graph search", "Computer vision", "Robotics"]
    evidence = evaluate_option_equivalence("stem", options, nli_threshold=0.99, nli_scorer=FakeNliScorer())
    assert escalation_reasons(evidence) == []


def test_detector_exception_becomes_error_evidence_not_a_crash():
    class ExplodingScorer:
        def warm_up(self):
            pass

        def score(self, premise, hypothesis):
            raise RuntimeError("simulated ONNX Runtime failure")

    evidence = evaluate_option_equivalence(
        "stem", ["a", "b", "c", "d"], nli_threshold=0.5, nli_scorer=ExplodingScorer()
    )
    nli_evidence = [item for item in evidence if item.detector == "nli_semantic"]
    assert len(nli_evidence) == 6
    assert all(item.verdict == "error" for item in nli_evidence)
    # Fail-closed: an unresolved detector always escalates, exactly like "equivalent".
    reasons = escalation_reasons(evidence)
    assert len(reasons) == 6


def test_detector_timeout_becomes_error_evidence_and_escalates():
    class SlowScorer:
        def warm_up(self):
            pass

        def score(self, premise, hypothesis):
            time.sleep(10)
            return EntailmentScores(contradiction=0.0, entailment=0.0, neutral=1.0)

    started = time.perf_counter()
    evidence = evaluate_option_equivalence(
        "stem", ["a", "b"], nli_threshold=0.5, nli_scorer=SlowScorer()
    )
    elapsed = time.perf_counter() - started
    assert elapsed < 8  # bounded well under the 10s sleep -- the gate must not hang
    nli_evidence = [item for item in evidence if item.detector == "nli_semantic"][0]
    assert nli_evidence.verdict == "error"
    assert "timeout" in nli_evidence.reason
    assert escalation_reasons(evidence) == [
        f"nli_semantic could not evaluate options 0/1: {nli_evidence.reason}"
    ]


def test_never_auto_approves_or_rejects_only_reports_evidence():
    """The gate itself has no concept of approval/rejection/revision/promotion -- it
    only ever returns evidence; authoring/review/risk.py is the only place that turns
    it into a recommendation, and it only ever escalates (see test_review_risk.py)."""
    evidence = evaluate_option_equivalence(
        "stem", ["0.75", "0.75", "x", "y"], nli_threshold=0.5, nli_scorer=FakeNliScorer()
    )
    for item in evidence:
        assert item.verdict in {"not_applicable", "equivalent", "not_equivalent", "error"}
