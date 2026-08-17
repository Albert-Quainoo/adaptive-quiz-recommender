"""Pairwise semantic-equivalence detection via a local NLI cross-encoder, run entirely
through ONNX Runtime -- never torch. This is deliberate: the generation/review Modal
environment's torch/transformers pin is untouched by this feature (see the
reviewer-model-evaluation branch's report for why upgrading it was ruled out), so this
detector's only framework dependency is `onnxruntime` (CPU) plus `transformers`'
tokenizer classes, which load and run without any working PyTorch backend -- verified
directly: AutoTokenizer.from_pretrained works and produces plain numpy arrays even when
transformers has disabled its torch backend for being too old.

Model pin (immutable commit revision, not a mutable branch/tag):
    repository: cross-encoder/nli-deberta-v3-xsmall
    revision:   a150876415327c80daeff35ca6f68f5ed8cf5c24
    file:       onnx/model.onnx (unquantized fp32 -- chosen over the quantized
                qint8_avx512/arm64/avx2 variants for portability: those are tuned for
                specific CPU instruction sets this code cannot verify are present in
                every environment it might run in, e.g. CI runners vs. Modal vs. a
                developer's laptop; a wrong choice there would be silently slow or
                fail on unsupported hardware. fp32 always works.)
    labels:     0=contradiction, 1=entailment, 2=neutral (from the pinned revision's
                config.json id2label -- verified, not assumed)

The threshold this detector's caller applies to these scores is a *separate* pin
(threshold_version, calibrated by scripts/calibrate_equivalence_nli_threshold.py) --
this module only ever returns raw probabilities.
"""

from dataclasses import dataclass

import numpy as np

NLI_MODEL_REPOSITORY = "cross-encoder/nli-deberta-v3-xsmall"
NLI_MODEL_REVISION = "a150876415327c80daeff35ca6f68f5ed8cf5c24"
_ONNX_FILENAME = "onnx/model.onnx"
_LABELS = ("contradiction", "entailment", "neutral")


@dataclass(frozen=True)
class EntailmentScores:
    contradiction: float
    entailment: float
    neutral: float


class NliScorer:
    """Loads the pinned tokenizer + ONNX session once, lazily, and scores premise/
    hypothesis pairs. A fresh instance re-downloads/reloads; authoring/review/
    equivalence_gate.py holds one process-wide instance via get_default_scorer()."""

    def __init__(self, *, repository: str = NLI_MODEL_REPOSITORY, revision: str = NLI_MODEL_REVISION) -> None:
        self.repository = repository
        self.revision = revision
        self._tokenizer = None
        self._session = None

    def warm_up(self) -> None:
        """Loads the tokenizer/ONNX session if not already loaded. Public so
        authoring/review/equivalence_gate.py can pay this one-time cost (a cold first
        call can take several seconds: HF Hub cache lookup + ONNX session
        construction) up front, outside any single pair's bounded timeout -- every
        pair's own inference is sub-100ms once warm (see
        evaluation/equivalence_calibration_report.md)."""
        self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        if self._session is not None:
            return
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        from transformers import AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.repository, revision=self.revision)
        model_path = hf_hub_download(self.repository, _ONNX_FILENAME, revision=self.revision)
        self._session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])

    def score(self, premise: str, hypothesis: str) -> EntailmentScores:
        self._ensure_loaded()
        input_names = {item.name for item in self._session.get_inputs()}
        encoded = self._tokenizer(premise, hypothesis, return_tensors="np", truncation=True, max_length=256)
        feed = {name: value.astype(np.int64) for name, value in encoded.items() if name in input_names}
        logits = self._session.run(None, feed)[0][0]
        exponentiated = np.exp(logits - logits.max())
        probabilities = exponentiated / exponentiated.sum()
        return EntailmentScores(**dict(zip(_LABELS, (float(value) for value in probabilities))))


class FakeNliScorer:
    """Deterministic test double -- never touches ONNX Runtime, a tokenizer, or the
    network. Returns a configured EntailmentScores for exact (premise, hypothesis)
    pairs, defaulting to low-entailment/high-neutral for anything unconfigured, so a
    test suite that injects this by default (tests/conftest.py's no_default_nli_model
    fixture) never spuriously escalates a candidate it isn't specifically testing."""

    def __init__(self, scores: dict[tuple[str, str], EntailmentScores] | None = None) -> None:
        self._scores = dict(scores or {})

    def score(self, premise: str, hypothesis: str) -> EntailmentScores:
        return self._scores.get(
            (premise, hypothesis), EntailmentScores(contradiction=0.0, entailment=0.0, neutral=1.0)
        )

    def warm_up(self) -> None:
        pass


_default_scorer: NliScorer | None = None


def get_default_scorer() -> NliScorer:
    global _default_scorer
    if _default_scorer is None:
        _default_scorer = NliScorer()
    return _default_scorer


def check_semantic_equivalence(
    option_index_a: int,
    option_index_b: int,
    stem: str,
    text_a: str,
    text_b: str,
    *,
    threshold: float,
    scorer: "NliScorer | FakeNliScorer | None" = None,
):
    from authoring.review.models import OptionPairEvidence

    scorer = scorer or get_default_scorer()
    premise_a = f"Given the question: {stem} Claim: {text_a}"
    premise_b = f"Given the question: {stem} Claim: {text_b}"
    forward = scorer.score(premise_a, premise_b)
    backward = scorer.score(premise_b, premise_a)

    is_equivalent = forward.entailment >= threshold and backward.entailment >= threshold
    return OptionPairEvidence(
        option_index_a=option_index_a,
        option_index_b=option_index_b,
        detector="nli_semantic",
        verdict="equivalent" if is_equivalent else "not_equivalent",
        score_or_normalized_form=(
            f"entailment(a->b)={forward.entailment:.3f} entailment(b->a)={backward.entailment:.3f}"
        ),
        reason=(
            f"bidirectional entailment >= threshold {threshold:.3f}"
            if is_equivalent
            else f"bidirectional entailment below threshold {threshold:.3f}"
        ),
    )
