"""authoring/review/equivalence_nli.py: pairwise semantic-equivalence detector logic,
using FakeNliScorer throughout -- never touches ONNX Runtime or the network. Real-model
integration is covered separately by tests/test_equivalence_gate_acceptance.py, which
skips itself when the pinned model isn't reachable.

The NliScorer._ensure_loaded()/checksum tests below mock transformers/huggingface_hub/
onnxruntime -- they exercise the real loading and checksum-verification code paths
without ever downloading anything or touching the network.
"""

import hashlib
from unittest.mock import MagicMock, patch

import authoring.review.equivalence_nli as equivalence_nli
from authoring.review.equivalence_nli import (
    EntailmentScores,
    FakeNliScorer,
    NliModelChecksumError,
    NliModelInitializationError,
    NliScorer,
    _verify_onnx_checksum,
    check_semantic_equivalence,
)


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


def test_verify_onnx_checksum_passes_for_matching_content(tmp_path, monkeypatch):
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"pretend-onnx-bytes")
    monkeypatch.setattr(
        equivalence_nli, "NLI_MODEL_ONNX_SHA256", hashlib.sha256(b"pretend-onnx-bytes").hexdigest()
    )
    _verify_onnx_checksum(str(model_file))  # must not raise


def test_verify_onnx_checksum_raises_on_mismatch(tmp_path):
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"corrupted-or-substituted-bytes")
    try:
        _verify_onnx_checksum(str(model_file))
        raise AssertionError("expected NliModelChecksumError")
    except NliModelChecksumError as exc:
        assert exc.expected == equivalence_nli.NLI_MODEL_ONNX_SHA256
        assert exc.actual == hashlib.sha256(b"corrupted-or-substituted-bytes").hexdigest()


def test_ensure_loaded_rejects_a_cached_file_that_fails_the_pinned_checksum(tmp_path):
    """A cached onnx/model.onnx that doesn't match NLI_MODEL_ONNX_SHA256 (corruption,
    a substituted file, ...) must raise NliModelChecksumError specifically -- not a
    generic NliModelInitializationError -- and must never construct an ONNX Runtime
    session from it."""
    wrong_model_file = tmp_path / "model.onnx"
    wrong_model_file.write_bytes(b"not the pinned model")
    scorer = NliScorer()
    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=MagicMock()),
        patch("huggingface_hub.hf_hub_download", return_value=str(wrong_model_file)),
        patch("onnxruntime.InferenceSession") as inference_session,
    ):
        try:
            scorer.warm_up()
            raise AssertionError("expected NliModelChecksumError")
        except NliModelChecksumError:
            pass
        assert inference_session.call_count == 0
    assert scorer._session is None


def test_tokenizer_load_failure_is_wrapped_never_raised_raw():
    """An arbitrary loader failure (network, missing repo, ...) must surface as
    NliModelInitializationError with the original exception chained as __cause__ --
    never the raw exception -- so authoring/review/equivalence_gate.py's sanitizer can
    recover just the original type name, never its (possibly path-carrying) message."""
    scorer = NliScorer()
    with patch(
        "transformers.AutoTokenizer.from_pretrained", side_effect=OSError("network unreachable")
    ):
        try:
            scorer.warm_up()
            raise AssertionError("expected NliModelInitializationError")
        except NliModelInitializationError as exc:
            assert type(exc) is NliModelInitializationError
            assert isinstance(exc.__cause__, OSError)


def test_successful_cached_initialization_loads_once_and_is_reused_across_candidates(tmp_path, monkeypatch):
    """Simulates a warm process-wide singleton: warm_up() succeeds once (tokenizer +
    ONNX session load + checksum verification), then is called again for two more
    'candidates' -- the loader/checksum work must not repeat."""
    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"pretend-onnx-bytes")
    monkeypatch.setattr(
        equivalence_nli, "NLI_MODEL_ONNX_SHA256", hashlib.sha256(b"pretend-onnx-bytes").hexdigest()
    )

    fake_tokenizer = MagicMock()
    fake_session = MagicMock()
    with (
        patch(
            "transformers.AutoTokenizer.from_pretrained", return_value=fake_tokenizer
        ) as tokenizer_from_pretrained,
        patch("huggingface_hub.hf_hub_download", return_value=str(model_file)) as hf_hub_download,
        patch("onnxruntime.InferenceSession", return_value=fake_session) as inference_session,
    ):
        scorer = NliScorer()
        scorer.warm_up()  # first candidate -- real load
        scorer.warm_up()  # second candidate -- must be a no-op
        scorer.warm_up()  # third candidate -- must still be a no-op

    assert tokenizer_from_pretrained.call_count == 1
    assert hf_hub_download.call_count == 1
    assert inference_session.call_count == 1
    assert scorer._tokenizer is fake_tokenizer
    assert scorer._session is fake_session
