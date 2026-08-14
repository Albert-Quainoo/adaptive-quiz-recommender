import pytest

from authoring.replenishment.inventory import SkillInventory
from authoring.replenishment.policy import ReplenishmentPolicyConfig, decide_replenishment


def row(skill_id, readiness, *, generation_strategy="generated", total=0, unseen=None):
    return SkillInventory(
        course_id="ai",
        skill_id=skill_id,
        generation_strategy=generation_strategy,
        total_approved_items=total,
        unseen_approved_items=unseen,
        pending_reference_candidates=0,
        approved_reference_candidates=0,
        pending_generated_questions=0,
        template_status="not_applicable",
        latest_job=None,
        readiness=readiness,
    )


DEFAULT = ReplenishmentPolicyConfig(low_supply_threshold=3, target_supply=6)


def test_enqueues_when_supply_is_below_threshold():
    inventory = {"AI-SRC-01": row("AI-SRC-01", "taxonomy_only", total=0)}
    [decision] = decide_replenishment(inventory, DEFAULT)
    assert decision.should_enqueue is True
    assert decision.requested_count == 6


def test_no_job_when_supply_meets_threshold():
    inventory = {"AI-SRC-01": row("AI-SRC-01", "ready", total=3)}
    [decision] = decide_replenishment(inventory, DEFAULT)
    assert decision.should_enqueue is False


def test_requested_count_sizes_toward_target_supply():
    inventory = {"AI-SRC-01": row("AI-SRC-01", "content_exhausted", total=1)}
    [decision] = decide_replenishment(inventory, DEFAULT)
    assert decision.requested_count == 5


def test_does_not_enqueue_when_an_active_job_already_exists():
    for readiness in (
        "retrieval_pending",
        "reference_review",
        "generation_pending",
        "question_review",
    ):
        inventory = {"AI-SRC-01": row("AI-SRC-01", readiness, total=0)}
        [decision] = decide_replenishment(inventory, DEFAULT)
        assert decision.should_enqueue is False, readiness


def test_does_not_re_enqueue_after_permanent_failure():
    inventory = {"AI-SRC-01": row("AI-SRC-01", "replenishment_failed", total=0)}
    [decision] = decide_replenishment(inventory, DEFAULT)
    assert decision.should_enqueue is False


def test_templated_skills_are_never_enqueued():
    inventory = {
        "AI-SRC-02": row("AI-SRC-02", "template_ready", generation_strategy="templated"),
        "AI-SRC-03": row(
            "AI-SRC-03", "content_exhausted", generation_strategy="templated"
        ),
    }
    decisions = decide_replenishment(inventory, DEFAULT)
    assert all(decision.should_enqueue is False for decision in decisions)


def test_monitors_every_active_skill_not_only_the_first():
    inventory = {
        skill_id: row(skill_id, "taxonomy_only", total=0)
        for skill_id in ("AI-FND-01", "AI-AGT-01", "AI-SRC-01", "AI-ML-01")
    }
    decisions = decide_replenishment(inventory, DEFAULT)
    assert {decision.skill_id for decision in decisions if decision.should_enqueue} == set(
        inventory
    )


def test_skill_specific_override_threshold_is_respected():
    inventory = {"AI-SRC-01": row("AI-SRC-01", "ready", total=4)}
    strict_override = {"AI-SRC-01": ReplenishmentPolicyConfig(low_supply_threshold=5, target_supply=8)}
    [decision] = decide_replenishment(inventory, DEFAULT, skill_overrides=strict_override)
    assert decision.should_enqueue is True
    assert decision.requested_count == 4


def test_config_rejects_target_below_threshold():
    with pytest.raises(ValueError):
        ReplenishmentPolicyConfig(low_supply_threshold=6, target_supply=3)
