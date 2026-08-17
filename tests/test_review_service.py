"""authoring.review.service.review_candidate() orchestration tests -- the heuristic
h(n)/g(n) regression case, cache reuse, deterministic short-circuit, and reviewer outage
routing, all fakes-only.
"""

from datetime import datetime, timezone

import pytest

from authoring.review.config import ReviewPolicyConfig
from authoring.review.equivalence_nli import EntailmentScores, FakeNliScorer
from authoring.review.reports import AutomatedReviewReportStore
from authoring.review.response_parser import ReviewerCallContext, ReviewerOutputError
from authoring.review.reviewer import FakeContentReviewer, ReviewerUnavailableError
from authoring.review.service import review_candidate
from tests.review_fixtures import (
    APPROVED_REFERENCES,
    CORRECTED_CANDIDATE,
    CORRECTED_QUESTION,
    CORRECTED_REVIEW_RESULT,
    INTENT,
    ORIGINAL_INACCURATE_CANDIDATE,
    ORIGINAL_INACCURATE_QUESTION,
    ORIGINAL_REVIEW_RESULT,
    SKILL,
)

FIXED_TIME = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)


def _store(tmp_path) -> AutomatedReviewReportStore:
    return AutomatedReviewReportStore(tmp_path / "reports.json")


def _config() -> ReviewPolicyConfig:
    return ReviewPolicyConfig(reviewer_passes=1)


def test_original_inaccurate_heuristic_candidate_never_gets_recommend_human_approval(tmp_path):
    reviewer = FakeContentReviewer({ORIGINAL_INACCURATE_QUESTION.question: ORIGINAL_REVIEW_RESULT})
    report = review_candidate(
        ORIGINAL_INACCURATE_CANDIDATE,
        SKILL,
        INTENT,
        APPROVED_REFERENCES,
        reviewer=reviewer,
        config=_config(),
        report_store=_store(tmp_path),
        clock=lambda: FIXED_TIME,
    )
    assert report.recommendation != "recommend_human_approval"
    assert report.risk_level == "critical"


def test_corrected_heuristic_revision_gets_recommend_human_approval(tmp_path):
    reviewer = FakeContentReviewer({CORRECTED_QUESTION.question: CORRECTED_REVIEW_RESULT})
    report = review_candidate(
        CORRECTED_CANDIDATE,
        SKILL,
        INTENT,
        APPROVED_REFERENCES,
        reviewer=reviewer,
        config=_config(),
        report_store=_store(tmp_path),
        clock=lambda: FIXED_TIME,
    )
    assert report.recommendation == "recommend_human_approval"
    assert report.risk_level == "low"


def test_deterministic_block_short_circuits_and_never_calls_reviewer(tmp_path):
    reviewer = FakeContentReviewer({})
    report = review_candidate(
        CORRECTED_CANDIDATE,
        SKILL,
        INTENT,
        APPROVED_REFERENCES,
        reviewer=reviewer,
        config=_config(),
        report_store=_store(tmp_path),
        existing_item_ids={CORRECTED_CANDIDATE.question_id},
        clock=lambda: FIXED_TIME,
    )
    assert report.recommendation == "reject"
    assert reviewer.calls == []
    # The exact-text-duplicate deterministic check already handles this candidate --
    # the hybrid equivalence gate never runs on the short-circuit path (see
    # authoring/review/service.py: it only runs once deterministic checks pass).
    assert report.equivalence_assessment is None


def test_equivalence_assessment_is_populated_on_a_normal_review(tmp_path):
    reviewer = FakeContentReviewer({CORRECTED_QUESTION.question: CORRECTED_REVIEW_RESULT})
    report = review_candidate(
        CORRECTED_CANDIDATE,
        SKILL,
        INTENT,
        APPROVED_REFERENCES,
        reviewer=reviewer,
        config=_config(),
        report_store=_store(tmp_path),
        clock=lambda: FIXED_TIME,
    )
    assert report.equivalence_assessment is not None
    assert report.equivalence_assessment.escalated is False
    assert report.equivalence_assessment.nli_model_repository == _config().equivalence_nli_model_repository
    assert len(report.equivalence_assessment.evidence) == 18  # 4 options -> 6 pairs x 3 detectors


def test_equivalence_gate_escalates_an_otherwise_clean_candidate_to_human_review(tmp_path):
    """A credible equivalence signal overrides what the semantic reviewer alone would
    have recommended -- CORRECTED_REVIEW_RESULT alone reaches recommend_human_approval
    (see test_corrected_heuristic_revision_gets_recommend_human_approval above); with a
    scorer configured to report two of CORRECTED_QUESTION's options as equivalent, the
    same candidate must instead require_full_human_review, never auto-approve."""
    option_a, option_b = CORRECTED_QUESTION.options[0], CORRECTED_QUESTION.options[1]
    stem = CORRECTED_QUESTION.question
    premise_a = f"Given the question: {stem} Claim: {option_a}"
    premise_b = f"Given the question: {stem} Claim: {option_b}"
    scorer = FakeNliScorer(
        {
            (premise_a, premise_b): EntailmentScores(contradiction=0.0, entailment=0.95, neutral=0.05),
            (premise_b, premise_a): EntailmentScores(contradiction=0.0, entailment=0.95, neutral=0.05),
        }
    )
    reviewer = FakeContentReviewer({CORRECTED_QUESTION.question: CORRECTED_REVIEW_RESULT})
    report = review_candidate(
        CORRECTED_CANDIDATE,
        SKILL,
        INTENT,
        APPROVED_REFERENCES,
        reviewer=reviewer,
        config=_config(),
        report_store=_store(tmp_path),
        clock=lambda: FIXED_TIME,
        equivalence_nli_scorer=scorer,
    )
    assert report.recommendation == "require_full_human_review"
    assert report.equivalence_assessment.escalated is True
    assert any(item.verdict == "equivalent" for item in report.equivalence_assessment.evidence)


def test_cached_report_is_reused_without_a_second_reviewer_call(tmp_path):
    reviewer = FakeContentReviewer({CORRECTED_QUESTION.question: CORRECTED_REVIEW_RESULT})
    store = _store(tmp_path)
    first = review_candidate(
        CORRECTED_CANDIDATE,
        SKILL,
        INTENT,
        APPROVED_REFERENCES,
        reviewer=reviewer,
        config=_config(),
        report_store=store,
        clock=lambda: FIXED_TIME,
    )
    second = review_candidate(
        CORRECTED_CANDIDATE,
        SKILL,
        INTENT,
        APPROVED_REFERENCES,
        reviewer=reviewer,
        config=_config(),
        report_store=store,
        clock=lambda: FIXED_TIME,
    )
    assert first.review_id == second.review_id
    assert reviewer.calls == [CORRECTED_QUESTION.question]


def test_reviewer_outage_propagates_for_the_caller_to_route_to_waiting_for_model(tmp_path):
    reviewer = FakeContentReviewer({CORRECTED_QUESTION.question: ReviewerUnavailableError("down")})
    with pytest.raises(ReviewerUnavailableError):
        review_candidate(
            CORRECTED_CANDIDATE,
            SKILL,
            INTENT,
            APPROVED_REFERENCES,
            reviewer=reviewer,
            config=_config(),
            report_store=_store(tmp_path),
            clock=lambda: FIXED_TIME,
        )
    # An outage must not persist a report -- the next attempt is a fresh cache miss.
    assert _store(tmp_path).latest_for_hash("anything") is None


def test_content_hash_changes_invalidate_the_cache(tmp_path):
    reviewer = FakeContentReviewer(
        {
            CORRECTED_QUESTION.question: CORRECTED_REVIEW_RESULT,
            ORIGINAL_INACCURATE_QUESTION.question: ORIGINAL_REVIEW_RESULT,
        }
    )
    store = _store(tmp_path)
    original_report = review_candidate(
        ORIGINAL_INACCURATE_CANDIDATE,
        SKILL,
        INTENT,
        APPROVED_REFERENCES,
        reviewer=reviewer,
        config=_config(),
        report_store=store,
        clock=lambda: FIXED_TIME,
    )
    corrected_report = review_candidate(
        CORRECTED_CANDIDATE,
        SKILL,
        INTENT,
        APPROVED_REFERENCES,
        reviewer=reviewer,
        config=_config(),
        report_store=store,
        clock=lambda: FIXED_TIME,
    )
    assert original_report.reviewed_content_hash != corrected_report.reviewed_content_hash
    assert original_report.recommendation != corrected_report.recommendation
    assert reviewer.calls == [ORIGINAL_INACCURATE_QUESTION.question, CORRECTED_QUESTION.question]


def test_parser_failure_preserves_real_provenance_instead_of_placeholders(tmp_path):
    """A ReviewerOutputError carrying context (real model/prompt identity and call
    metadata -- a genuine call was made, parsing just failed) must land on the final
    report verbatim, never the "n/a"/"unknown" placeholders reserved for when no
    reviewer call happened at all."""
    context = ReviewerCallContext(
        reviewer_model_id="meta-llama/Llama-3.1-8B-Instruct",
        reviewer_model_revision="abc123",
        reviewer_prompt_version="review-v2",
        reviewer_prompt_template_hash="d" * 64,
        rendered_review_request_hash="d" * 64,
        finish_reason="length",
        input_token_count=512,
        output_token_count=1000,
        configured_max_new_tokens=1000,
        request_count=2,
        raw_responses=("not json", "still not json"),
    )
    reviewer = FakeContentReviewer(
        {
            CORRECTED_QUESTION.question: ReviewerOutputError(
                "reviewer output still invalid after one repair retry", context=context
            )
        }
    )
    report = review_candidate(
        CORRECTED_CANDIDATE,
        SKILL,
        INTENT,
        APPROVED_REFERENCES,
        reviewer=reviewer,
        config=_config(),
        report_store=_store(tmp_path),
        clock=lambda: FIXED_TIME,
    )

    assert report.reviewer_model_id == "meta-llama/Llama-3.1-8B-Instruct"
    assert report.reviewer_model_revision == "abc123"
    assert report.reviewer_prompt_version == "review-v2"
    assert report.reviewer_prompt_template_hash == "d" * 64
    assert report.rendered_review_request_hash == "d" * 64
    assert report.finish_reason == "length"
    assert report.input_token_count == 512
    assert report.output_token_count == 1000
    assert report.configured_max_new_tokens == 1000
    assert report.reviewer_request_count == 2


def test_no_promotion_after_any_parse_failure(tmp_path):
    """However much provenance a failed parse preserves, the recommendation must
    never be an approval-eligible value -- a malformed/unparseable reviewer response
    is always a critical, require_full_human_review report."""
    context = ReviewerCallContext(
        reviewer_model_id="fake-model",
        reviewer_model_revision="fake-rev",
        reviewer_prompt_version="review-v2",
        reviewer_prompt_template_hash="e" * 64,
        rendered_review_request_hash="e" * 64,
        finish_reason="length",
        input_token_count=10,
        output_token_count=1000,
        configured_max_new_tokens=1000,
        request_count=2,
        raw_responses=("not json", "still not json"),
    )
    reviewer = FakeContentReviewer(
        {
            CORRECTED_QUESTION.question: ReviewerOutputError(
                "reviewer output still invalid after one repair retry", context=context
            )
        }
    )
    report = review_candidate(
        CORRECTED_CANDIDATE,
        SKILL,
        INTENT,
        APPROVED_REFERENCES,
        reviewer=reviewer,
        config=_config(),
        report_store=_store(tmp_path),
        clock=lambda: FIXED_TIME,
    )

    assert report.recommendation == "require_full_human_review"
    assert report.recommendation != "recommend_human_approval"
    assert report.risk_level == "critical"
