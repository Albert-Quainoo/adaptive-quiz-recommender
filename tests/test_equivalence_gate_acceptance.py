"""End-to-end acceptance test for the hybrid option-equivalence gate against the real
pinned NLI model -- the frozen benchmark cases from the reviewer-model-evaluation
branch's live comparison, plus the domain-neutral fixtures.

Opt-in only, like tests/postgres_test_safety.py's DSN-gated integration tests: this is
the one place in the suite that legitimately downloads/runs the real ~90MB ONNX model,
so it never runs as part of a normal `pytest tests/` invocation (see
tests/conftest.py's no_default_nli_model fixture, which this test explicitly bypasses
by constructing its own real NliScorer).

    QUIZ_TEST_LIVE_EQUIVALENCE_NLI=1 python -m pytest tests/test_equivalence_gate_acceptance.py
"""

import os

import pytest

from authoring.review.equivalence_gate import escalation_reasons, evaluate_option_equivalence
from authoring.review.equivalence_nli import NliScorer
from tests.review_domain_neutral_fixtures import GEN_CAUSAL_QUESTION, GEN_UNIT_QUESTION
from tests.review_fnd_fixtures import FND03_KNOWN_GOOD_QUESTION, FND04_ORIGINAL_QUESTION, FND04_REVISED_QUESTION
from tests.review_generalized_fixtures import (
    GEN_CLOSE_QUESTION,
    GEN_MATH_QUESTION,
    GEN_NEGATION_QUESTION,
    GEN_PARAPHRASE_QUESTION,
)

_ENV_VAR = "QUIZ_TEST_LIVE_EQUIVALENCE_NLI"

pytestmark = pytest.mark.skipif(
    not os.getenv(_ENV_VAR),
    reason=f"set {_ENV_VAR}=1 to run the real-model equivalence-gate acceptance test",
)

_THRESHOLD = 0.25  # authoring/review/config.py's calibrated default -- see
# evaluation/equivalence_calibration_report.md

_CASES = [
    ("fnd04-original", FND04_ORIGINAL_QUESTION, True),
    ("fnd04-revised", FND04_REVISED_QUESTION, False),
    ("fnd03-known-good", FND03_KNOWN_GOOD_QUESTION, False),
    ("gen-paraphrase", GEN_PARAPHRASE_QUESTION, True),
    ("gen-math-equivalent", GEN_MATH_QUESTION, True),
    ("gen-unit-equivalent", GEN_UNIT_QUESTION, True),
    ("gen-close-distractor", GEN_CLOSE_QUESTION, False),
    ("gen-negation", GEN_NEGATION_QUESTION, False),
    ("gen-causal-distractor", GEN_CAUSAL_QUESTION, False),
]


@pytest.fixture(scope="module")
def real_scorer():
    scorer = NliScorer()
    scorer.warm_up()
    return scorer


@pytest.mark.parametrize("case_id,question,expect_escalation", _CASES)
def test_frozen_case_escalation_matches_expectation(case_id, question, expect_escalation, real_scorer):
    evidence = evaluate_option_equivalence(
        question.question, question.options, nli_threshold=_THRESHOLD, nli_scorer=real_scorer
    )
    escalated = bool(escalation_reasons(evidence))
    assert escalated == expect_escalation, (
        f"{case_id}: expected escalation={expect_escalation}, got {escalated}. "
        f"Evidence: {[e.model_dump() for e in evidence if e.verdict in ('equivalent', 'error')]}"
    )
