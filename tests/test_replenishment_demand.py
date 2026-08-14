from authoring.grounded_batch import question_id
from authoring.question_intents import PilotBlueprint, QuestionIntent
from authoring.replenishment.demand import compute_blueprint_slot_demand, deficient_slots

BATCH_ID = "demand-test-batch"
SKILL_ID = "AI-TEST-01"


def _intent(index: int) -> QuestionIntent:
    return QuestionIntent(
        intent_id=f"{SKILL_ID}-INT-{index:02d}",
        skill_id=SKILL_ID,
        assessment_focus="focus",
        question_archetype="archetype",
        preferred_reference_ids=["REF-01"],
        required_concepts=["concept"],
        prohibited_conflations=["conflation"],
        difficulty="introductory",
    )


def _blueprint(count: int) -> PilotBlueprint:
    return PilotBlueprint(
        batch_id=BATCH_ID,
        prompt_version="v1",
        review_status="reviewed",
        intents=[_intent(index) for index in range(count)],
    )


def test_slot_approved_as_written_is_satisfied():
    blueprint = _blueprint(1)
    slot_id = question_id(BATCH_ID, SKILL_ID, 0)
    demand = compute_blueprint_slot_demand(blueprint, SKILL_ID, {slot_id})
    assert demand[0].satisfied is True
    assert deficient_slots(demand) == []


def test_slot_approved_via_revision_is_satisfied():
    """A human-revised, approved candidate gets a fresh
    f"{original_question_id}-rev-{suffix}" bank item_id (see
    authoring/grounded_review.py's propose_revision) -- never string-equal to the
    slot's own deterministic id. Before this fix, such a slot looked permanently
    deficient even after human approval."""
    blueprint = _blueprint(1)
    slot_id = question_id(BATCH_ID, SKILL_ID, 0)
    revision_item_id = f"{slot_id}-rev-088615b50aa2"
    demand = compute_blueprint_slot_demand(blueprint, SKILL_ID, {revision_item_id})
    assert demand[0].satisfied is True
    assert deficient_slots(demand) == []


def test_unrelated_bank_items_never_falsely_satisfy_a_slot():
    blueprint = _blueprint(1)
    demand = compute_blueprint_slot_demand(
        blueprint, SKILL_ID, {"AI-OTHER-01-deadbeefcafefeed"}
    )
    assert demand[0].satisfied is False
    assert len(deficient_slots(demand)) == 1


def test_only_the_matching_slots_revision_satisfies_it_not_a_different_slot():
    blueprint = _blueprint(2)
    slot_0 = question_id(BATCH_ID, SKILL_ID, 0)
    slot_1 = question_id(BATCH_ID, SKILL_ID, 1)
    demand = compute_blueprint_slot_demand(
        blueprint, SKILL_ID, {f"{slot_0}-rev-abc123"}
    )
    by_index = {slot.question_index: slot for slot in demand}
    assert by_index[0].satisfied is True
    assert by_index[1].satisfied is False
    assert by_index[1].question_id == slot_1
