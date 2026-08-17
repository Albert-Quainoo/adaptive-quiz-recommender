import pytest

from authoring.replenishment.budget import CycleBudgetConfig, CycleBudgetTracker


def test_config_rejects_non_positive_caps():
    with pytest.raises(ValueError):
        CycleBudgetConfig(max_new_candidates=0)
    with pytest.raises(ValueError):
        CycleBudgetConfig(max_generation_calls=0)
    with pytest.raises(ValueError):
        CycleBudgetConfig(max_cost_usd=0)
    with pytest.raises(ValueError):
        CycleBudgetConfig(max_ticks=0)
    with pytest.raises(ValueError):
        CycleBudgetConfig(cost_per_generation_call_usd=-0.01)


def test_new_candidate_cap_blocks_a_fourth_never_started_job():
    tracker = CycleBudgetTracker(config=CycleBudgetConfig(max_new_candidates=3))
    for job_id in ("a", "b", "c"):
        assert tracker.would_start_new_candidate_beyond_cap(job_id, is_new=True) is False
        tracker.record_tick(job_type="retrieve_references", job_id=job_id, is_new=True)

    assert tracker.would_start_new_candidate_beyond_cap("d", is_new=True) is True
    # A job already counted this run may keep advancing past the cap.
    assert tracker.would_start_new_candidate_beyond_cap("a", is_new=False) is False


def test_resumed_job_never_counts_against_the_new_candidate_cap():
    tracker = CycleBudgetTracker(config=CycleBudgetConfig(max_new_candidates=1))
    tracker.record_tick(job_type="retrieve_references", job_id="a", is_new=True)
    assert tracker.would_start_new_candidate_beyond_cap("a", is_new=False) is False
    assert tracker.would_start_new_candidate_beyond_cap("old-waiting-job", is_new=False) is False


def test_exhausted_on_generation_call_cap():
    tracker = CycleBudgetTracker(
        config=CycleBudgetConfig(max_generation_calls=2, max_cost_usd=1000, max_ticks=1000)
    )
    tracker.record_tick(job_type="generate_questions", job_id="a", is_new=True)
    assert tracker.exhausted() is None
    tracker.record_tick(job_type="automated_review", job_id="a", is_new=False)
    assert tracker.exhausted() is not None
    assert "call cap" in tracker.exhausted()


def test_exhausted_on_cost_cap():
    tracker = CycleBudgetTracker(
        config=CycleBudgetConfig(
            max_generation_calls=1000,
            max_cost_usd=0.05,
            cost_per_generation_call_usd=0.05,
            max_ticks=1000,
        )
    )
    tracker.record_tick(job_type="generate_questions", job_id="a", is_new=True)
    assert tracker.exhausted() is not None
    assert "cost cap" in tracker.exhausted()


def test_exhausted_on_tick_safety_limit():
    tracker = CycleBudgetTracker(
        config=CycleBudgetConfig(max_ticks=1, max_generation_calls=1000, max_cost_usd=1000)
    )
    tracker.record_tick(job_type="retrieve_references", job_id="a", is_new=True)
    assert tracker.exhausted() is not None
    assert "tick safety limit" in tracker.exhausted()


def test_search_calls_are_free_by_default():
    tracker = CycleBudgetTracker(config=CycleBudgetConfig())
    tracker.record_tick(job_type="retrieve_references", job_id="a", is_new=True)
    assert tracker.estimated_cost_usd == 0.0
    assert tracker.search_calls == 1


def test_to_dict_reports_counts_and_estimated_cost():
    tracker = CycleBudgetTracker(
        config=CycleBudgetConfig(cost_per_generation_call_usd=0.05, cost_per_review_call_usd=0.02)
    )
    tracker.record_tick(job_type="generate_questions", job_id="a", is_new=True)
    tracker.record_tick(job_type="automated_review", job_id="a", is_new=False)
    payload = tracker.to_dict()
    assert payload["generation_calls"] == 1
    assert payload["review_calls"] == 1
    assert payload["estimated_cost_usd"] == pytest.approx(0.07)
    assert payload["new_candidates_started"] == 1
