"""Backpressure cap tests: a review backlog is a normal waiting state, so these only
ever downgrade should_enqueue=True decisions, never touch job status."""

from authoring.replenishment.policy import ReplenishmentDecision
from authoring.review.backpressure import apply_review_backlog_caps
from authoring.review.config import ReviewPolicyConfig

CONFIG = ReviewPolicyConfig(max_pending_per_skill=4, max_backlog=25)


def _decision(skill_id: str, should_enqueue: bool = True) -> ReplenishmentDecision:
    return ReplenishmentDecision(
        course_id="ai",
        skill_id=skill_id,
        should_enqueue=should_enqueue,
        requested_count=3,
        reason="supply below threshold",
    )


def test_untouched_decisions_pass_through_below_caps():
    decisions = [_decision("AI-SRC-08")]
    result = apply_review_backlog_caps(
        decisions, pending_per_skill={"AI-SRC-08": 1}, total_pending_backlog=5, config=CONFIG
    )
    assert result == decisions


def test_per_skill_pending_cap_pauses_only_that_skill():
    decisions = [_decision("AI-SRC-08"), _decision("AI-FND-01")]
    result = apply_review_backlog_caps(
        decisions,
        pending_per_skill={"AI-SRC-08": 4, "AI-FND-01": 0},
        total_pending_backlog=5,
        config=CONFIG,
    )
    by_skill = {decision.skill_id: decision for decision in result}
    assert by_skill["AI-SRC-08"].should_enqueue is False
    assert by_skill["AI-SRC-08"].requested_count == 0
    assert by_skill["AI-FND-01"].should_enqueue is True


def test_global_backlog_cap_pauses_every_skill():
    decisions = [_decision("AI-SRC-08"), _decision("AI-FND-01")]
    result = apply_review_backlog_caps(
        decisions, pending_per_skill={}, total_pending_backlog=25, config=CONFIG
    )
    assert all(decision.should_enqueue is False for decision in result)


def test_already_false_decisions_are_left_alone():
    decisions = [_decision("AI-SRC-08", should_enqueue=False)]
    result = apply_review_backlog_caps(
        decisions, pending_per_skill={"AI-SRC-08": 10}, total_pending_backlog=25, config=CONFIG
    )
    assert result[0].reason == "supply below threshold"
