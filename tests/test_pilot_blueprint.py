from collections import Counter
from types import SimpleNamespace

from authoring.grounded_batch import slot_count
from authoring.pilot_blueprint import (
    EXPECTED_INVENTORY,
    PILOT_BATCH_ID,
    PILOT_SKILL_IDS,
    validate_pilot_blueprint,
)
from authoring.question_intents import load_blueprint_for_batch
from scripts.generate_grounded_batch import build_parser
from scripts.validate_pilot_blueprint import generation_command, proposed_seeds
from taxonomy.loader import (
    course_paths,
    course_provenance_path,
    load_reference_provenance,
    load_skills,
)


def inputs():
    blueprint = load_blueprint_for_batch(PILOT_BATCH_ID)
    skills = load_skills(*course_paths("ai")).skills
    provenance = load_reference_provenance(course_provenance_path("ai"))
    return blueprint, skills, provenance


def test_six_skill_blueprint_passes_every_required_validation():
    blueprint, skills, provenance = inputs()
    summary = validate_pilot_blueprint(blueprint, skills, provenance)

    assert summary["intent_count"] == 24
    assert blueprint.review_status == "blueprint-approved"
    assert blueprint.reviewer_id == "albert"
    assert blueprint.reviewed_at is not None
    assert summary["questions_per_intent"] == 1
    assert summary["difficulty_counts"] == {
        "intermediate": 9,
        "introductory": 15,
    }
    assert all(summary["checks"].values())
    assert all(item["ready"] for item in summary["reference_readiness"].values())


def test_inventory_ids_ordering_and_designs_are_deterministic_and_distinct():
    blueprint, _, _ = inputs()
    ids = [intent.intent_id for intent in blueprint.intents]
    inventory = Counter(
        (intent.skill_id, intent.difficulty) for intent in blueprint.intents
    )

    assert len(ids) == len(set(ids)) == 24
    assert dict(inventory) == EXPECTED_INVENTORY
    assert [intent.skill_id for intent in blueprint.intents] == sorted(
        (intent.skill_id for intent in blueprint.intents),
        key=PILOT_SKILL_IDS.index,
    )
    for skill_id in PILOT_SKILL_IDS:
        mine = [intent for intent in blueprint.intents if intent.skill_id == skill_id]
        assert len({intent.assessment_focus for intent in mine}) == len(mine)
        assert len(
            {(intent.question_archetype, intent.cognitive_demand) for intent in mine}
        ) == len(mine)


def test_new_src_01_intents_do_not_reuse_legacy_validator_ids():
    blueprint, _, _ = inputs()
    current = {
        intent.intent_id
        for intent in blueprint.intents
        if intent.skill_id == "AI-SRC-01"
    }

    assert current == {
        "AI-SRC-01-INT-11",
        "AI-SRC-01-INT-12",
        "AI-SRC-01-INT-13",
    }


def test_every_mapping_is_approved_same_skill_provenance():
    blueprint, _, provenance = inputs()
    by_id = {record.reference_id: record for record in provenance}

    for intent in blueprint.intents:
        assert intent.preferred_reference_ids
        assert all(
            reference_id in by_id
            and by_id[reference_id].skill_id == intent.skill_id
            for reference_id in intent.preferred_reference_ids
        )


def test_ai_src_03_convention_covers_every_required_lifecycle_decision():
    blueprint, _, _ = inputs()
    convention = " ".join(blueprint.skill_conventions["AI-SRC-03"]).casefold()

    assert "frontier contains" in convention
    assert "admitted to the frontier" in convention
    assert "expansion" in convention
    assert "repeated state" in convention
    assert "lower path cost" in convention
    assert "does not reopen" in convention
    assert "never request a bfs, dfs, ucs, greedy, or a-star expansion trace" in convention


def test_seed_proposal_and_kaggle_command_are_reproducible():
    blueprint, _, _ = inputs()
    first = proposed_seeds(blueprint)

    assert first == proposed_seeds(blueprint)
    assert len(first) == 24
    assert len({item["seed"] for item in first}) == 24
    command = generation_command(blueprint)
    assert 'export MODEL_REPOSITORY="meta-llama/Llama-3.1-8B-Instruct"' in command
    assert "--all-blueprint-intents" in command
    assert "--difficulty mixed" in command
    assert f"--base-seed {blueprint.base_seed}" in command
    assert all(f"--skill-id {skill_id}" in command for skill_id in PILOT_SKILL_IDS)


def test_cli_and_slot_count_support_the_exact_blueprint_inventory():
    arguments = build_parser().parse_args(
        [
            "--batch-id",
            PILOT_BATCH_ID,
            "--skill-id",
            "AI-AGT-01",
            "--all-blueprint-intents",
            "--base-seed",
            "20260811",
            "--output",
            "outputs/not-run",
            "--model-id",
            "model/repository",
            "--prompt-version",
            "v3.3",
            "--difficulty",
            "mixed",
        ]
    )
    blueprint, _, _ = inputs()
    agent_intents = [
        intent for intent in blueprint.intents if intent.skill_id == "AI-AGT-01"
    ]

    assert arguments.questions_per_skill is None
    assert arguments.all_blueprint_intents is True
    assert arguments.difficulty == "mixed"
    assert slot_count(SimpleNamespace(questions_per_skill=None), agent_intents) == 6
