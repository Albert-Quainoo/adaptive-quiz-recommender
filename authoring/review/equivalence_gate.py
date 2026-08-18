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

The NLI scorer's one-time warm-up (tokenizer/ONNX session load, including the
onnx/model.onnx checksum verification -- see equivalence_nli.py) is bounded the same
way, under its own more generous timeout (_WARMUP_TIMEOUT_SECONDS): an OSError,
checksum mismatch, timeout, or any other initialization failure can never propagate
out of evaluate_option_equivalence() or terminate the caller (the replenishment
worker). Because no option pair was ever evaluated in that case, this is never recorded
as pairwise OptionPairEvidence (there is no pair to attach it to, and fabricating
sentinel indices would misrepresent evaluation that never happened) -- instead
evaluate_option_equivalence() returns it as a second, assessment-level return value
(an "initialization_error" string, or None on success) that
authoring/review/service.py records directly on EquivalenceAssessment.
initialization_error and folds into the same escalation-reasons list risk.py already
treats identically to a per-pair "error" verdict. The reason string is sanitized: it
never includes a filesystem path, URL, environment value, or other exception-message
content that an underlying library (huggingface_hub, transformers, onnxruntime) might
embed (see _sanitize_initialization_failure below).
"""

import itertools
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

from authoring.review.equivalence_math import check_math_equivalence
from authoring.review.equivalence_nli import (
    FakeNliScorer,
    NliModelChecksumError,
    NliScorer,
    check_semantic_equivalence,
    get_default_scorer,
)
from authoring.review.equivalence_units import check_unit_equivalence
from authoring.review.models import EquivalenceDetector, OptionPairEvidence

GATE_VERSION = "equivalence-gate-v1"
_DETECTOR_TIMEOUT_SECONDS = 5.0
# More generous than _DETECTOR_TIMEOUT_SECONDS: a cold warm-up pays a one-time
# HF Hub cache lookup/download (~280MB) + checksum stream-hash + ONNX session
# construction cost that a single pair's inference never does, and it must not be
# mistaken for a hang at the same threshold a slow inference call would trip. Still
# bounded -- a genuinely stuck download must not block a worker job tick forever.
_WARMUP_TIMEOUT_SECONDS = 30.0


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


def _sanitize_initialization_failure(exc: Exception) -> str:
    """Reduces any warm-up failure to a reason string safe to store as evidence. Only
    NliModelChecksumError's message is ever included verbatim -- it is authored
    entirely by this codebase (two hex digests, fixed template text), never by an
    underlying library, so it cannot carry a filesystem path, URL, or environment
    value. Every other exception is reduced to just its (or its wrapped cause's) type
    name -- huggingface_hub/transformers/onnxruntime exception messages routinely
    embed a local cache path or a repository URL, which must never reach stored
    evidence."""
    if isinstance(exc, NliModelChecksumError):
        return f"nli model initialization failed: {exc}"
    origin = exc.__cause__ if exc.__cause__ is not None else exc
    return f"nli model initialization failed: {type(origin).__name__}"


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
) -> tuple[list[OptionPairEvidence], str | None]:
    """Runs all 3 detectors over every pair of `options` and returns
    (evidence, initialization_error). For 4 options the normal case is 6 pairs x 3
    detectors = 18 evidence entries and initialization_error=None -- unless the NLI
    scorer's warm-up itself fails, in which case no pair was ever evaluated: this
    returns ([], sanitized_reason) instead, since fabricating pair-indexed evidence
    for pairs that were never evaluated would misrepresent what happened (see this
    module's docstring). The caller (authoring/review/service.py) records
    initialization_error on EquivalenceAssessment and folds it into the same
    escalation-reasons list a per-pair "error" verdict would produce.
    `nli_scorer` is an injection point for tests -- production callers omit it and get
    the pinned default (lazy-loaded, process-wide, see equivalence_nli.get_default_scorer)."""
    resolved_scorer = nli_scorer or get_default_scorer()

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
        # Bounded exactly like a per-pair detector call, just under a more generous
        # timeout and with its own assessment-level (not pairwise) failure shape --
        # see this module's docstring. A successful warm-up on an already-loaded
        # scorer (the common case for every candidate after a worker process's first)
        # is a fast no-op, so this costs nothing on the hot path.
        warmup_future = executor.submit(resolved_scorer.warm_up)
        try:
            warmup_future.result(timeout=_WARMUP_TIMEOUT_SECONDS)
        except FutureTimeoutError:
            return [], f"nli model warm-up exceeded {_WARMUP_TIMEOUT_SECONDS}s timeout"
        except Exception as exc:  # noqa: BLE001 -- any initialization failure (network
            # error, cache corruption, checksum mismatch, unsupported opset, ...) must
            # become one sanitized, assessment-level error -- never propagate and
            # crash the caller (the replenishment worker). See
            # _sanitize_initialization_failure and this module's docstring.
            return [], _sanitize_initialization_failure(exc)

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
    return evidence, None


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
