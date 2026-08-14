"""Slot-level replenishment demand: which specific (skill, intent, difficulty) slots
a blueprint declares are already fulfilled by an approved bank item, versus still
deficient.

Distinct from authoring/replenishment/policy.py's decide_replenishment(), which
answers a coarser question ("does this skill's aggregate approved count justify
enqueuing a job at all") from the skill-level bank count and a low_supply_threshold.
This module answers the precise question the worker needs right before it would
spend a network call: for a specific blueprint's intent pool, exactly which slots
still need generating. A slot's identity is the same deterministic
authoring.grounded_batch.question_id(batch_id, skill_id, question_index) the
automated-review layer's own no_duplicate_item_id check already compares against the
approved bank -- this module runs that same comparison proactively, before
generation, instead of only discovering it after a wasted model call.

Nothing here touches a file or the network -- it is a pure function of a blueprint
object and a set of already-approved bank item ids, both of which the caller already
had to load for other reasons.
"""

from pydantic import BaseModel

from authoring.grounded_batch import question_id
from authoring.question_intents import PilotBlueprint, intents_by_skill


class SlotDemand(BaseModel):
    skill_id: str
    intent_id: str
    difficulty: str | None
    question_index: int
    question_id: str
    satisfied: bool


def compute_blueprint_slot_demand(
    blueprint: PilotBlueprint, skill_id: str, approved_item_ids: set[str]
) -> list[SlotDemand]:
    """One SlotDemand per intent in this skill's pool, in blueprint order.

    questions_per_intent is fixed at 1 (PilotBlueprint.questions_per_intent), so each
    intent is exactly one slot -- there is no separate "question-type allocation" to
    track beyond the intent itself.
    """
    pool = intents_by_skill(blueprint).get(skill_id, [])
    demand: list[SlotDemand] = []
    for index, intent in enumerate(pool):
        slot_id = question_id(blueprint.batch_id, skill_id, index)
        # A slot approved as written keeps this exact id in the bank. A slot approved
        # via a human revision (authoring/grounded_review.py's propose_revision)
        # instead gets `f"{slot_id}-rev-{suffix}"` as its bank item_id -- a real
        # candidate for the same slot, just not string-equal to it. Without the
        # startswith fallback, an already-approved revision looks permanently
        # deficient here forever, and a caller (e.g. the worker, before generating)
        # would spend a redundant model call generating a duplicate for a slot a
        # human already filled.
        satisfied = slot_id in approved_item_ids or any(
            item_id.startswith(f"{slot_id}-rev-") for item_id in approved_item_ids
        )
        demand.append(
            SlotDemand(
                skill_id=skill_id,
                intent_id=intent.intent_id,
                difficulty=intent.difficulty,
                question_index=index,
                question_id=slot_id,
                satisfied=satisfied,
            )
        )
    return demand


def deficient_slots(demand: list[SlotDemand]) -> list[SlotDemand]:
    return [slot for slot in demand if not slot.satisfied]
