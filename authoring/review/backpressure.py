"""Review-backlog-aware backpressure, layered on top of (not inside)
authoring.replenishment.policy.decide_replenishment.

A review backlog is a normal waiting state, not a failure: this module never touches job
status, it only prevents *new* generation from being enqueued while a skill's pending
questions or the course's total review backlog are already at capacity. Both counts are
supplied by the caller (authoring/replenishment/cli.py's `scan`), which already has them
from authoring.replenishment.inventory.compute_course_inventory and
authoring.review.reports.count_pending_review_items -- this function is a pure
transformation of decide_replenishment's output, not a change to it.
"""

from dataclasses import replace

from authoring.replenishment.policy import ReplenishmentDecision
from authoring.review.config import ReviewPolicyConfig


def apply_review_backlog_caps(
    decisions: list[ReplenishmentDecision],
    *,
    pending_per_skill: dict[str, int],
    total_pending_backlog: int,
    config: ReviewPolicyConfig,
) -> list[ReplenishmentDecision]:
    """Downgrade any should_enqueue=True decision that would push a capped skill or the
    course's total backlog over its configured limit. Decisions this function did not
    touch are returned unchanged."""
    if total_pending_backlog >= config.max_backlog:
        return [
            replace(
                decision,
                should_enqueue=False,
                requested_count=0,
                reason=(
                    f"review backlog {total_pending_backlog} at or above cap "
                    f"{config.max_backlog}; pausing generation"
                ),
            )
            if decision.should_enqueue
            else decision
            for decision in decisions
        ]

    capped = []
    for decision in decisions:
        if not decision.should_enqueue:
            capped.append(decision)
            continue
        pending = pending_per_skill.get(decision.skill_id, 0)
        if pending >= config.max_pending_per_skill:
            capped.append(
                replace(
                    decision,
                    should_enqueue=False,
                    requested_count=0,
                    reason=(
                        f"pending questions {pending} at or above per-skill cap "
                        f"{config.max_pending_per_skill}; pausing generation"
                    ),
                )
            )
        else:
            capped.append(decision)
    return capped
