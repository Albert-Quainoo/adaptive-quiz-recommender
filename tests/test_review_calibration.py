"""Calibration harness tests: fakes only, deterministic, no network.

The reference reviewer's judgments are treated as ground truth by design (see
evaluation/review_calibration.py's module docstring) -- these tests verify the
harness's own dataset assembly and metric computation are correct, not that any
particular LLM is a good judge.
"""

from pathlib import Path

from evaluation.review_calibration import (
    CRITICAL_MUTATION_TYPES,
    MUTATIONS,
    build_negative_cases,
    build_positive_cases,
    build_reference_reviewer,
    run_calibration,
)


def test_positive_set_covers_the_38_item_bank_and_the_corrected_heuristic_revision():
    cases = build_positive_cases()
    assert len(cases) == 39
    assert all(case.label == "positive" for case in cases)
    assert "heuristic-corrected-revision" in {case.case_id for case in cases}


def test_negative_set_covers_the_original_heuristic_item_and_every_mutation_type():
    cases = build_negative_cases()
    assert all(case.label == "negative" for case in cases)
    mutation_types = {case.mutation_type for case in cases}
    assert "heuristic_wording_conflation" in mutation_types
    assert "remove_grounding_reference" in mutation_types
    assert set(MUTATIONS) <= mutation_types


def test_every_case_has_a_unique_question_stem(tmp_path: Path):
    cases = build_positive_cases() + build_negative_cases()
    stems = [case.question.question for case in cases]
    assert len(stems) == len(set(stems))


def test_calibration_run_achieves_zero_false_low_risk_rate_against_the_reference_reviewer(
    tmp_path: Path,
):
    cases = build_positive_cases() + build_negative_cases()
    reviewer_factory = lambda: build_reference_reviewer(cases)  # noqa: E731

    report = run_calibration(
        cases, reviewer_factory=reviewer_factory, report_store_path=tmp_path / "reports.json"
    )

    assert report.total_cases == len(cases)
    assert report.false_low_risk_rate == 0.0
    assert report.approval_precision == 1.0
    assert report.critical_error_detection_rate == 1.0
    assert report.parser_failure_rate == 0.0


def test_critical_mutation_types_are_a_subset_of_defined_mutations():
    assert CRITICAL_MUTATION_TYPES <= (set(MUTATIONS) | {"remove_grounding_reference", "heuristic_wording_conflation"})


def test_calibration_never_reaches_a_real_network_or_model(tmp_path: Path):
    """Regression guard: the default reviewer_factory here must never be capable of
    a live call -- the no_live_requests autouse fixture in tests/conftest.py would
    catch an httpx call, but FakeContentReviewer never even attempts one."""
    from authoring.review.reviewer import FakeContentReviewer

    cases = build_positive_cases() + build_negative_cases()
    reviewer_factory = lambda: build_reference_reviewer(cases)  # noqa: E731
    assert isinstance(reviewer_factory(), FakeContentReviewer)

    run_calibration(cases, reviewer_factory=reviewer_factory, report_store_path=tmp_path / "reports.json")
