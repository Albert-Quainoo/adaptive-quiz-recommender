"""Validation contract for the six-skill, 24-question grounded pilot."""

from collections import Counter, defaultdict
from collections.abc import Sequence

from authoring.question_intents import PilotBlueprint
from taxonomy.schemas import ReferenceProvenance, SkillDefinition


PILOT_BATCH_ID = "grounded-pilot-20260811-v3"
PILOT_SKILL_IDS = (
    "AI-FND-01",
    "AI-AGT-01",
    "AI-SRC-01",
    "AI-SRC-02",
    "AI-SRC-03",
    "AI-SRC-08",
)
EXPECTED_INVENTORY = {
    ("AI-FND-01", "intermediate"): 3,
    ("AI-AGT-01", "introductory"): 3,
    ("AI-AGT-01", "intermediate"): 3,
    ("AI-SRC-01", "introductory"): 3,
    ("AI-SRC-02", "introductory"): 3,
    ("AI-SRC-03", "introductory"): 3,
    ("AI-SRC-03", "intermediate"): 3,
    ("AI-SRC-08", "introductory"): 3,
}
MINIMUM_APPROVED_REFERENCES = {
    "AI-FND-01": 4,
    "AI-AGT-01": 2,
    "AI-SRC-01": 3,
    "AI-SRC-02": 2,
    "AI-SRC-03": 3,
    "AI-SRC-08": 2,
}


def approved_reference_inventory(
    provenance: Sequence[ReferenceProvenance],
) -> dict[str, dict[str, object]]:
    by_skill: dict[str, list[ReferenceProvenance]] = defaultdict(list)
    for record in provenance:
        if record.skill_id in PILOT_SKILL_IDS:
            by_skill[record.skill_id].append(record)

    return {
        skill_id: {
            "approved_count": len(by_skill[skill_id]),
            "domains": sorted(
                {record.source_domain for record in by_skill[skill_id]}
            ),
            "reference_ids": sorted(
                record.reference_id for record in by_skill[skill_id]
            ),
            "ready": len(by_skill[skill_id])
            >= MINIMUM_APPROVED_REFERENCES[skill_id],
        }
        for skill_id in PILOT_SKILL_IDS
    }


def validate_pilot_blueprint(
    blueprint: PilotBlueprint,
    skills: Sequence[SkillDefinition],
    provenance: Sequence[ReferenceProvenance],
) -> dict[str, object]:
    """Validate inventory, grounding, ordering, and intent diversity."""
    if blueprint.batch_id != PILOT_BATCH_ID:
        raise ValueError(f"blueprint batch_id must be {PILOT_BATCH_ID}")
    if blueprint.questions_per_intent != 1:
        raise ValueError("the pilot must generate exactly one question per intent")
    if blueprint.base_seed is None:
        raise ValueError("the pilot needs an explicit deterministic base seed")
    if len(blueprint.intents) != 24:
        raise ValueError("the pilot blueprint must contain exactly 24 intents")

    by_skill = {skill.skill_id: skill for skill in skills}
    references = {record.reference_id: record for record in provenance}
    inventory = Counter(
        (intent.skill_id, intent.difficulty) for intent in blueprint.intents
    )
    if dict(inventory) != EXPECTED_INVENTORY:
        raise ValueError("the pilot skill/difficulty inventory is incorrect")

    if {intent.skill_id for intent in blueprint.intents} != set(PILOT_SKILL_IDS):
        raise ValueError("the pilot must cover all six selected skills")

    difficulty_counts = Counter(intent.difficulty for intent in blueprint.intents)
    if difficulty_counts != Counter({"introductory": 15, "intermediate": 9}):
        raise ValueError("the pilot must contain 15 introductory and 9 intermediate intents")

    if any(
        intent.difficulty not in {"introductory", "intermediate"}
        for intent in blueprint.intents
    ):
        raise ValueError("the pilot contains an unsupported difficulty")

    skill_rank = {skill_id: index for index, skill_id in enumerate(PILOT_SKILL_IDS)}
    difficulty_rank = {"introductory": 0, "intermediate": 1}
    expected_order = sorted(
        blueprint.intents,
        key=lambda intent: (
            skill_rank[intent.skill_id],
            difficulty_rank[intent.difficulty],
            intent.intent_id,
        ),
    )
    if blueprint.intents != expected_order:
        raise ValueError("pilot intents are not in deterministic order")

    focus_by_skill: dict[str, set[str]] = defaultdict(set)
    design_by_skill: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for intent in blueprint.intents:
        skill = by_skill[intent.skill_id]
        if intent.learning_objective != skill.learning_objective:
            raise ValueError(
                f"{intent.intent_id} does not use the exact canonical learning objective"
            )
        if not intent.cognitive_demand:
            raise ValueError(f"{intent.intent_id} has no cognitive demand")
        if not intent.required_question_characteristics:
            raise ValueError(
                f"{intent.intent_id} has no required question characteristics"
            )
        if not intent.prohibited_ambiguity_patterns:
            raise ValueError(
                f"{intent.intent_id} has no prohibited ambiguity patterns"
            )
        if not intent.expected_misconception_or_distractor_strategy:
            raise ValueError(
                f"{intent.intent_id} has no misconception or distractor strategy"
            )

        missing = [
            reference_id
            for reference_id in intent.preferred_reference_ids
            if reference_id not in references
            or references[reference_id].skill_id != intent.skill_id
        ]
        if missing:
            raise ValueError(
                f"{intent.intent_id} has unapproved reference mappings: "
                + ", ".join(missing)
            )

        focus = " ".join(intent.assessment_focus.casefold().split())
        if focus in focus_by_skill[intent.skill_id]:
            raise ValueError(f"{intent.skill_id} repeats an assessment focus")
        focus_by_skill[intent.skill_id].add(focus)

        design = (
            " ".join(intent.question_archetype.casefold().split()),
            " ".join(intent.cognitive_demand.casefold().split()),
        )
        if design in design_by_skill[intent.skill_id]:
            raise ValueError(f"{intent.skill_id} repeats an intent design")
        design_by_skill[intent.skill_id].add(design)

        if intent.difficulty == "introductory":
            text = " ".join(
                (
                    intent.assessment_focus,
                    *intent.required_question_characteristics,
                    *intent.generation_constraints,
                )
            ).casefold()
            if any(term in text for term in ("list all", "complete list", "entire list")):
                raise ValueError(
                    f"{intent.intent_id} requires unsupported list recall"
                )

    conventions = " ".join(
        blueprint.skill_conventions.get("AI-SRC-03", [])
    ).casefold()
    required_convention_terms = (
        "frontier contains",
        "admitted to the frontier",
        "expansion",
        "repeated state",
        "lower path cost",
        "does not reopen",
    )
    if not all(term in conventions for term in required_convention_terms):
        raise ValueError("AI-SRC-03 solver conventions are incomplete")

    reference_inventory = approved_reference_inventory(provenance)
    if not all(item["ready"] for item in reference_inventory.values()):
        raise ValueError("one or more pilot skills lack sufficient approved grounding")

    return {
        "batch_id": blueprint.batch_id,
        "intent_count": len(blueprint.intents),
        "questions_per_intent": blueprint.questions_per_intent,
        "difficulty_counts": dict(sorted(difficulty_counts.items())),
        "skill_difficulty_counts": {
            f"{skill_id}:{difficulty}": inventory[(skill_id, difficulty)]
            for skill_id, difficulty in EXPECTED_INVENTORY
        },
        "reference_readiness": reference_inventory,
        "checks": {
            "exactly_24_intents": True,
            "correct_skill_and_difficulty_counts": True,
            "unique_intent_ids": True,
            "approved_reference_mappings": True,
            "deterministic_ordering": True,
            "all_six_skills_covered": True,
            "supported_difficulties_only": True,
            "sufficient_intent_diversity": True,
            "ai_src_03_conventions_explicit": True,
        },
    }
