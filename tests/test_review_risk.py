"""Pure risk-policy tests -- the reviewer LLM never assigns the final decision, this
module's score_risk() does, so it is fully testable with plain fixtures and no model."""

from authoring.review.config import ReviewPolicyConfig
from authoring.review.deterministic import run_deterministic_checks
from authoring.review.risk import score_risk
from tests.review_fixtures import (
    APPROVED_REFERENCES,
    CORRECTED_CANDIDATE,
    CORRECTED_REVIEW_RESULT,
    INTENT,
    ORIGINAL_INACCURATE_CANDIDATE,
    ORIGINAL_REVIEW_RESULT,
    SKILL,
)

CONFIG = ReviewPolicyConfig()


def _clean_checks():
    return run_deterministic_checks(CORRECTED_CANDIDATE, SKILL, INTENT, APPROVED_REFERENCES)


def test_deterministic_blocking_failure_short_circuits_to_reject():
    checks = run_deterministic_checks(
        CORRECTED_CANDIDATE,
        SKILL,
        INTENT,
        APPROVED_REFERENCES,
        existing_item_ids={CORRECTED_CANDIDATE.question_id},
    )
    decision = score_risk(checks, [], config=CONFIG)
    assert decision.risk_level == "critical"
    assert decision.recommendation == "reject"
    assert decision.blocking_reasons


def test_heuristic_regression_original_is_critical_and_not_approved():
    decision = score_risk(_clean_checks(), [ORIGINAL_REVIEW_RESULT], config=CONFIG)
    assert decision.risk_level == "critical"
    assert decision.recommendation != "recommend_human_approval"


def test_heuristic_regression_corrected_can_reach_low_risk():
    decision = score_risk(_clean_checks(), [CORRECTED_REVIEW_RESULT], config=CONFIG)
    assert decision.risk_level == "low"
    assert decision.recommendation == "recommend_human_approval"


def test_reviewer_output_errors_force_critical_require_full_human_review():
    decision = score_risk(
        _clean_checks(),
        [CORRECTED_REVIEW_RESULT],
        config=CONFIG,
        reviewer_output_errors=["reviewer output cites nonexistent reference id(s): bogus"],
    )
    assert decision.risk_level == "critical"
    assert decision.recommendation == "require_full_human_review"


def test_equivalence_escalation_forces_require_full_human_review_never_reject():
    """A candidate that would otherwise be "low"/recommend_human_approval is escalated
    -- never auto-rejected -- by a credible option-equivalence signal from the hybrid
    gate (authoring/review/equivalence_gate.py)."""
    decision = score_risk(
        _clean_checks(),
        [CORRECTED_REVIEW_RESULT],
        config=CONFIG,
        equivalence_escalation_reasons=[
            "unit_conversion flags options 0/1 as equivalent (0.75 cup vs 0.75 cup)"
        ],
    )
    assert decision.risk_level == "high"
    assert decision.recommendation == "require_full_human_review"
    assert "unit_conversion flags options 0/1 as equivalent (0.75 cup vs 0.75 cup)" in decision.blocking_reasons


def test_equivalence_escalation_never_downgrades_an_existing_critical_reject_to_less_severe():
    """A candidate that was already critical/reject for an unrelated reason (here: a
    reviewer disagreeing with the declared answer, via ORIGINAL_REVIEW_RESULT) stays at
    critical severity when equivalence evidence also escalates it -- the recommendation
    moves from reject to require_full_human_review (a human always looks, never a
    silent auto-reject), but never drops below critical."""
    decision = score_risk(
        _clean_checks(),
        [ORIGINAL_REVIEW_RESULT],
        config=CONFIG,
        equivalence_escalation_reasons=["nli_semantic flags options 0/1 as equivalent (entailment=0.9)"],
    )
    assert decision.risk_level == "critical"
    assert decision.recommendation == "require_full_human_review"


def test_no_equivalence_evidence_leaves_decision_unaffected():
    with_none = score_risk(_clean_checks(), [CORRECTED_REVIEW_RESULT], config=CONFIG)
    with_empty = score_risk(
        _clean_checks(), [CORRECTED_REVIEW_RESULT], config=CONFIG, equivalence_escalation_reasons=[]
    )
    assert with_none.recommendation == with_empty.recommendation == "recommend_human_approval"
    assert with_none.risk_level == with_empty.risk_level == "low"


def test_no_passes_is_critical_require_full_human_review():
    decision = score_risk(_clean_checks(), [], config=CONFIG)
    assert decision.risk_level == "critical"
    assert decision.recommendation == "require_full_human_review"


def test_no_defensible_option_alone_is_critical_and_blocking():
    """A reviewer determining that none of the four options is defensible must, on
    its own, force critical risk with a blocking reason explaining the answer is
    absent from the options -- independent of matches_declared_answer."""
    no_option = CORRECTED_REVIEW_RESULT.model_copy(
        update={
            "answer_assessment": CORRECTED_REVIEW_RESULT.answer_assessment.model_copy(
                update={
                    "no_defensible_option": True,
                    "matches_declared_answer": False,
                    "selected_option_text": "an answer not among the options",
                }
            )
        }
    )
    decision = score_risk(_clean_checks(), [no_option], config=CONFIG)
    assert decision.risk_level == "critical"
    assert any(
        "no defensible option" in reason and "absent from the options" in reason
        for reason in decision.blocking_reasons
    )


def test_no_defensible_option_never_reaches_recommend_human_approval():
    no_option = CORRECTED_REVIEW_RESULT.model_copy(
        update={
            "answer_assessment": CORRECTED_REVIEW_RESULT.answer_assessment.model_copy(
                update={
                    "no_defensible_option": True,
                    "matches_declared_answer": False,
                    "selected_option_text": "an answer not among the options",
                }
            )
        }
    )
    decision = score_risk(_clean_checks(), [no_option], config=CONFIG)
    assert decision.recommendation != "recommend_human_approval"
    # A single clean pass that is critical purely on content grounds (not a parser
    # failure or multi-pass disagreement) is a "reject", not "require_full_human_
    # review" -- see score_risk's branches.
    assert decision.recommendation == "reject"


def test_multiple_defensible_answers_is_blocking():
    ambiguous = CORRECTED_REVIEW_RESULT.model_copy(
        update={
            "answer_assessment": CORRECTED_REVIEW_RESULT.answer_assessment.model_copy(
                update={"multiple_defensible_answers": True}
            )
        }
    )
    decision = score_risk(_clean_checks(), [ambiguous], config=CONFIG)
    assert decision.risk_level == "critical"
    assert any("defensible" in reason for reason in decision.blocking_reasons)


def test_unsupported_explanation_raises_risk_to_propose_revision():
    unsupported = CORRECTED_REVIEW_RESULT.model_copy(
        update={
            "grounding_assessment": CORRECTED_REVIEW_RESULT.grounding_assessment.model_copy(
                update={"unsupported_claims": ["explanation asserts a fact no reference states"]}
            )
        }
    )
    decision = score_risk(_clean_checks(), [unsupported], config=CONFIG)
    assert decision.risk_level == "high"
    assert decision.recommendation == "propose_revision"


def test_objective_mismatch_raises_risk():
    mismatched = CORRECTED_REVIEW_RESULT.model_copy(
        update={
            "objective_assessment": CORRECTED_REVIEW_RESULT.objective_assessment.model_copy(
                update={"measures_declared_skill": False}
            )
        }
    )
    decision = score_risk(_clean_checks(), [mismatched], config=CONFIG)
    assert decision.risk_level == "high"


def test_reviewer_disagreement_across_passes_requires_full_human_review():
    first = CORRECTED_REVIEW_RESULT
    second = CORRECTED_REVIEW_RESULT.model_copy(
        update={
            "objective_assessment": CORRECTED_REVIEW_RESULT.objective_assessment.model_copy(
                update={"satisfies_intent_blueprint": False}
            )
        }
    )
    decision = score_risk(_clean_checks(), [first, second], config=CONFIG)
    assert decision.risk_level in ("high", "critical")
    assert decision.recommendation == "require_full_human_review"


def test_severe_disagreement_across_passes_stays_critical_and_requires_full_human_review():
    decision = score_risk(
        _clean_checks(), [CORRECTED_REVIEW_RESULT, ORIGINAL_REVIEW_RESULT], config=CONFIG
    )
    assert decision.risk_level == "critical"
    assert decision.recommendation == "require_full_human_review"


def test_uncalibrated_configuration_blocks_low_risk():
    decision = score_risk(
        _clean_checks(),
        [CORRECTED_REVIEW_RESULT],
        config=CONFIG,
        is_calibrated_configuration=False,
    )
    assert decision.risk_level != "low"


def test_unknown_reviewer_version_is_elevated():
    decision = score_risk(
        _clean_checks(),
        [CORRECTED_REVIEW_RESULT],
        config=CONFIG,
        known_reviewer_versions=frozenset({("some-other-model", "rev-2", "review-v2")}),
    )
    assert decision.risk_level != "low"
