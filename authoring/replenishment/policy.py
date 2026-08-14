"""Pure decision of which skills need a replenishment job.

Takes the inventory computed elsewhere and returns decisions; it never reads
a file or touches the job repository itself, which is what makes it directly
unit-testable without any fakes.
"""

from collections.abc import Mapping
from dataclasses import dataclass

from authoring.replenishment.inventory import SkillInventory

# Readiness states where a job is already in flight, permanently blocked, or
# where the pipeline has nothing to offer (templated skills are served by
# templates/registry.py, never by retrieval+generation).
_BLOCKED_READINESS = frozenset(
    {
        "retrieval_pending",
        "reference_review",
        "generation_pending",
        "question_review",
        "replenishment_failed",
        "template_ready",
    }
)


@dataclass(frozen=True)
class ReplenishmentPolicyConfig:
    low_supply_threshold: int
    target_supply: int

    def __post_init__(self) -> None:
        if self.low_supply_threshold <= 0 or self.target_supply <= 0:
            raise ValueError("thresholds must be positive")
        if self.target_supply < self.low_supply_threshold:
            raise ValueError("target_supply must be at least low_supply_threshold")


@dataclass(frozen=True)
class ReplenishmentDecision:
    course_id: str
    skill_id: str
    should_enqueue: bool
    requested_count: int
    reason: str


def decide_replenishment(
    inventory: Mapping[str, SkillInventory],
    default_config: ReplenishmentPolicyConfig,
    *,
    skill_overrides: Mapping[str, ReplenishmentPolicyConfig] | None = None,
) -> list[ReplenishmentDecision]:
    """Decide, for every active skill, whether it needs a replenishment job."""
    overrides = skill_overrides or {}
    decisions: list[ReplenishmentDecision] = []

    for skill_id, row in inventory.items():
        config = overrides.get(skill_id, default_config)

        if row.generation_strategy == "templated":
            decisions.append(
                ReplenishmentDecision(
                    course_id=row.course_id,
                    skill_id=skill_id,
                    should_enqueue=False,
                    requested_count=0,
                    reason="templated skills are served by templates/registry.py",
                )
            )
            continue

        if row.readiness in _BLOCKED_READINESS:
            decisions.append(
                ReplenishmentDecision(
                    course_id=row.course_id,
                    skill_id=skill_id,
                    should_enqueue=False,
                    requested_count=0,
                    reason=f"not eligible: readiness={row.readiness}",
                )
            )
            continue

        supply = (
            row.unseen_approved_items
            if row.unseen_approved_items is not None
            else row.total_approved_items
        )

        if supply >= config.low_supply_threshold:
            decisions.append(
                ReplenishmentDecision(
                    course_id=row.course_id,
                    skill_id=skill_id,
                    should_enqueue=False,
                    requested_count=0,
                    reason=f"supply {supply} meets threshold {config.low_supply_threshold}",
                )
            )
            continue

        requested_count = max(config.target_supply - supply, 1)
        decisions.append(
            ReplenishmentDecision(
                course_id=row.course_id,
                skill_id=skill_id,
                should_enqueue=True,
                requested_count=requested_count,
                reason=f"supply {supply} below threshold {config.low_supply_threshold}",
            )
        )

    return decisions
