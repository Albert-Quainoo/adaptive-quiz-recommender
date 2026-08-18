"""Hybrid option-equivalence gate config (authoring/review/config.py) -- pinned model
provenance defaults, threshold env-var override, and validation."""

import pytest

from authoring.review.config import ReviewPolicyConfig, load_review_policy_config


def test_equivalence_defaults_are_the_calibrated_pinned_model():
    config = ReviewPolicyConfig()
    assert config.equivalence_nli_model_repository == "cross-encoder/nli-deberta-v3-xsmall"
    assert config.equivalence_nli_model_revision == "a150876415327c80daeff35ca6f68f5ed8cf5c24"
    assert config.equivalence_nli_threshold == 0.25
    assert config.equivalence_threshold_version


def test_load_review_policy_config_reads_equivalence_overrides():
    config = load_review_policy_config(
        environ={
            "QUIZ_REVIEW_EQUIVALENCE_NLI_MODEL_REPOSITORY": "some-org/some-model",
            "QUIZ_REVIEW_EQUIVALENCE_NLI_MODEL_REVISION": "deadbeef",
            "QUIZ_REVIEW_EQUIVALENCE_NLI_THRESHOLD": "0.6",
            "QUIZ_REVIEW_EQUIVALENCE_THRESHOLD_VERSION": "test-version",
        }
    )
    assert config.equivalence_nli_model_repository == "some-org/some-model"
    assert config.equivalence_nli_model_revision == "deadbeef"
    assert config.equivalence_nli_threshold == 0.6
    assert config.equivalence_threshold_version == "test-version"


def test_equivalence_nli_threshold_must_be_in_unit_interval():
    with pytest.raises(ValueError, match="equivalence_nli_threshold"):
        ReviewPolicyConfig(equivalence_nli_threshold=1.5)
    with pytest.raises(ValueError, match="equivalence_nli_threshold"):
        ReviewPolicyConfig(equivalence_nli_threshold=-0.1)
