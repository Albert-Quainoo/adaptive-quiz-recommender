"""Environment-driven configuration for the automated review layer.

Mirrors authoring/replenishment/cli.py's `_policy_config` env-var pattern: every value
has a conservative default and can be overridden by a `QUIZ_REVIEW_*` environment
variable, so ordinary local development and tests never need to set anything.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass

_PREFIX = "QUIZ_REVIEW_"

_TRUTHY = {"1", "true", "yes", "on"}


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in _TRUTHY


@dataclass(frozen=True)
class ReviewPolicyConfig:
    review_policy_version: str = "review-policy-v1"
    reviewer_provider: str = "fake"
    reviewer_passes: int = 1
    low_risk_confirmation_passes: int = 2
    max_pending_per_skill: int = 4
    max_backlog: int = 25
    max_automatic_revisions: int = 2
    max_calls_per_job_tick: int = 4
    # When True, automated review never triggers an automatic model rewrite -- every
    # candidate a human sees is exactly what retrieval/generation/review produced live,
    # with no automation-authored edit in between. Approval/promotion already require an
    # explicit human call regardless of this flag (see worker.py's
    # _handle_promote_approved_items docstring), so this is the only gate shadow mode
    # needs.
    shadow_mode: bool = False
    model_timeout_seconds: float = 180.0
    grounding_confidence_threshold: float = 0.7
    answer_confidence_threshold: float = 0.7
    # A review pass's own output budget, independent of generation's max_new_tokens
    # (which defaults to 800 -- see modal_app.py/worker.py's LlamaBatchModel). The
    # compact review contract is far shorter than a generated question, but this is
    # still tracked separately so raising or lowering one never silently moves the
    # other.
    reviewer_max_new_tokens: int = 1000
    # Hybrid option-equivalence gate (authoring/review/equivalence_gate.py): pinned NLI
    # model provenance + calibrated threshold, recorded into every EquivalenceAssessment
    # so a stored report always shows exactly what produced it (authoring/review/
    # models.py). Defaults are the pinned model this was calibrated against -- see
    # scripts/calibrate_equivalence_nli_threshold.py for the calibration method and
    # evaluation/equivalence_calibration_report.md for the resulting numbers this
    # threshold and threshold_version come from.
    equivalence_nli_model_repository: str = "cross-encoder/nli-deberta-v3-xsmall"
    equivalence_nli_model_revision: str = "a150876415327c80daeff35ca6f68f5ed8cf5c24"
    # Calibrated by scripts/calibrate_equivalence_nli_threshold.py on 2026-08-17 against
    # evaluation/equivalence_nli_labeled_pairs.py's calibration split (zero false
    # positives there); held-out results (precision 1.0, recall 0.5) are in
    # evaluation/equivalence_calibration_report.md.
    equivalence_nli_threshold: float = 0.25
    equivalence_threshold_version: str = "equivalence-threshold-v1-2026-08-17"

    def __post_init__(self) -> None:
        if self.reviewer_passes < 1:
            raise ValueError("reviewer_passes must be at least 1")
        if self.low_risk_confirmation_passes < 1:
            raise ValueError("low_risk_confirmation_passes must be at least 1")
        if self.max_pending_per_skill < 1:
            raise ValueError("max_pending_per_skill must be at least 1")
        if self.max_backlog < 1:
            raise ValueError("max_backlog must be at least 1")
        if self.max_automatic_revisions < 0:
            raise ValueError("max_automatic_revisions must be non-negative")
        if self.max_calls_per_job_tick < 1:
            raise ValueError("max_calls_per_job_tick must be at least 1")
        if self.model_timeout_seconds <= 0:
            raise ValueError("model_timeout_seconds must be positive")
        if self.reviewer_max_new_tokens < 1:
            raise ValueError("reviewer_max_new_tokens must be at least 1")
        if not (0.0 <= self.grounding_confidence_threshold <= 1.0):
            raise ValueError("grounding_confidence_threshold must be in [0, 1]")
        if not (0.0 <= self.answer_confidence_threshold <= 1.0):
            raise ValueError("answer_confidence_threshold must be in [0, 1]")
        if not (0.0 <= self.equivalence_nli_threshold <= 1.0):
            raise ValueError("equivalence_nli_threshold must be in [0, 1]")


def load_review_policy_config(environ: Mapping[str, str] | None = None) -> ReviewPolicyConfig:
    environ = os.environ if environ is None else environ
    defaults = ReviewPolicyConfig()

    def setting(name: str, default: str, cast=str):
        return cast(environ.get(f"{_PREFIX}{name}", default))

    return ReviewPolicyConfig(
        review_policy_version=setting("POLICY_VERSION", defaults.review_policy_version),
        reviewer_provider=setting("REVIEWER_PROVIDER", defaults.reviewer_provider),
        reviewer_passes=setting("PASSES", str(defaults.reviewer_passes), int),
        low_risk_confirmation_passes=setting(
            "LOW_RISK_CONFIRMATION_PASSES", str(defaults.low_risk_confirmation_passes), int
        ),
        max_pending_per_skill=setting(
            "MAX_PENDING_PER_SKILL", str(defaults.max_pending_per_skill), int
        ),
        max_backlog=setting("MAX_BACKLOG", str(defaults.max_backlog), int),
        max_automatic_revisions=setting(
            "MAX_AUTOMATIC_REVISIONS", str(defaults.max_automatic_revisions), int
        ),
        max_calls_per_job_tick=setting(
            "MAX_CALLS_PER_JOB_TICK", str(defaults.max_calls_per_job_tick), int
        ),
        model_timeout_seconds=setting(
            "MODEL_TIMEOUT_SECONDS", str(defaults.model_timeout_seconds), float
        ),
        grounding_confidence_threshold=setting(
            "GROUNDING_CONFIDENCE_THRESHOLD", str(defaults.grounding_confidence_threshold), float
        ),
        answer_confidence_threshold=setting(
            "ANSWER_CONFIDENCE_THRESHOLD", str(defaults.answer_confidence_threshold), float
        ),
        reviewer_max_new_tokens=setting(
            "MAX_NEW_TOKENS", str(defaults.reviewer_max_new_tokens), int
        ),
        shadow_mode=setting(
            "SHADOW_MODE", "true" if defaults.shadow_mode else "false", _parse_bool
        ),
        equivalence_nli_model_repository=setting(
            "EQUIVALENCE_NLI_MODEL_REPOSITORY", defaults.equivalence_nli_model_repository
        ),
        equivalence_nli_model_revision=setting(
            "EQUIVALENCE_NLI_MODEL_REVISION", defaults.equivalence_nli_model_revision
        ),
        equivalence_nli_threshold=setting(
            "EQUIVALENCE_NLI_THRESHOLD", str(defaults.equivalence_nli_threshold), float
        ),
        equivalence_threshold_version=setting(
            "EQUIVALENCE_THRESHOLD_VERSION", defaults.equivalence_threshold_version
        ),
    )
