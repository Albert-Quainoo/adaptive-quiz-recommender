"""Hybrid option-equivalence gate: evaluates every pair of a candidate's options with
three independent detectors (symbolic math, unit conversion, local NLI) and produces
structured evidence for authoring/review/risk.py to escalate on.

Never runs in the learner path -- this module is only ever imported/called from
authoring/review/service.py's review_candidate(), the authoring-side pipeline. It never
touches banks, BKT models, course state, or the presentation/scoring path.

Bounded: each detector call for each pair runs under a hard wall-clock timeout
(_DETECTOR_TIMEOUT_SECONDS) via a thread pool. A detector that raises unexpectedly or
exceeds its budget produces an "error" verdict -- never silently treated as "no
equivalence" (see authoring/review/models.py's EquivalenceVerdict docstring) -- and
authoring/review/risk.py escalates on "error" exactly like "equivalent".
"""

import itertools
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from authoring.review.equivalence_math import check_math_equivalence
from authoring.review.equivalence_nli import (
    FakeNliScorer,
    NliScorer,
    check_semantic_equivalence,
    get_default_scorer,
)
from authoring.review.equivalence_units import check_unit_equivalence
from authoring.review.models import EquivalenceDetector, OptionPairEvidence

GATE_VERSION = "equivalence-gate-v1"
_DETECTOR_TIMEOUT_SECONDS = 5.0


def _error_evidence(
    option_index_a: int, option_index_b: int, detector: EquivalenceDetector, reason: str
) -> OptionPairEvidence:
    return OptionPairEvidence(
        option_index_a=option_index_a,
        option_index_b=option_index_b,
        detector=detector,
        verdict="error",
        score_or_normalized_form="n/a",
        reason=reason,
    )


def _run_bounded(
    executor: ThreadPoolExecutor,
    detector: EquivalenceDetector,
    option_index_a: int,
    option_index_b: int,
    call,
) -> OptionPairEvidence:
    future = executor.submit(call)
    try:
        return future.result(timeout=_DETECTOR_TIMEOUT_SECONDS)
    except FutureTimeoutError:
        return _error_evidence(
            option_index_a, option_index_b, detector,
            f"detector exceeded {_DETECTOR_TIMEOUT_SECONDS}s timeout",
        )
    except Exception as exc:  # noqa: BLE001 -- any unexpected detector failure must
        # become structured error evidence, never propagate and abort the review, and
        # never be silently swallowed as "no signal."
        return _error_evidence(option_index_a, option_index_b, detector, f"{type(exc).__name__}: {exc}")


def evaluate_option_equivalence(
    stem: str,
    options: list[str],
    *,
    nli_threshold: float,
    nli_scorer: NliScorer | FakeNliScorer | None = None,
) -> list[OptionPairEvidence]:
    """Runs all 3 detectors over every pair of `options`. For 4 options this is 6
    pairs x 3 detectors = 18 evidence entries. `nli_scorer` is an injection point for
    tests -- production callers omit it and get the pinned default (lazy-loaded,
    process-wide, see equivalence_nli.get_default_scorer)."""
    resolved_scorer = nli_scorer or get_default_scorer()
    resolved_scorer.warm_up()  # untimed, one-time cost -- see NliScorer.warm_up's docstring

    evidence: list[OptionPairEvidence] = []
    pairs = list(itertools.combinations(range(len(options)), 2))
    # Deliberately not a `with ThreadPoolExecutor(...) as executor:` block: the context
    # manager's __exit__ calls shutdown(wait=True), which would block this function
    # until every submitted task finishes -- including one _run_bounded already gave up
    # on via future.result(timeout=...), silently undoing the bound this function
    # exists to provide. shutdown(wait=False) lets an already-abandoned (timed-out)
    # call keep running in its own thread without blocking the caller; Python cannot
    # forcibly kill a running thread, so this is the best available bound.
    # A pool of exactly 1 would let one hung call (e.g. an unresponsive ONNX session)
    # starve every later call's own worker thread, cascading a single timeout into
    # every remaining pair -- a small bounded pool keeps one stuck call from blocking
    # the rest.
    executor = ThreadPoolExecutor(max_workers=4)
    try:
        for index_a, index_b in pairs:
            text_a, text_b = options[index_a], options[index_b]
            evidence.append(
                _run_bounded(
                    executor, "symbolic_math", index_a, index_b,
                    lambda a=index_a, b=index_b, ta=text_a, tb=text_b: check_math_equivalence(a, b, ta, tb),
                )
            )
            evidence.append(
                _run_bounded(
                    executor, "unit_conversion", index_a, index_b,
                    lambda a=index_a, b=index_b, ta=text_a, tb=text_b: check_unit_equivalence(a, b, ta, tb),
                )
            )
            evidence.append(
                _run_bounded(
                    executor, "nli_semantic", index_a, index_b,
                    lambda a=index_a, b=index_b, ta=text_a, tb=text_b: check_semantic_equivalence(
                        a, b, stem, ta, tb, threshold=nli_threshold, scorer=resolved_scorer
                    ),
                )
            )
    finally:
        executor.shutdown(wait=False)
    return evidence


def escalation_reasons(evidence: list[OptionPairEvidence]) -> list[str]:
    """Human-readable reasons for every evidence entry that must force
    require_full_human_review: "equivalent" (a credible signal) and "error" (an
    unresolved detector -- see module docstring). "not_applicable"/"not_equivalent"
    never appear here."""
    reasons = []
    for item in evidence:
        if item.verdict == "equivalent":
            reasons.append(
                f"{item.detector} flags options {item.option_index_a}/{item.option_index_b} "
                f"as equivalent ({item.score_or_normalized_form})"
            )
        elif item.verdict == "error":
            reasons.append(
                f"{item.detector} could not evaluate options {item.option_index_a}/"
                f"{item.option_index_b}: {item.reason}"
            )
    return sorted(set(reasons))
