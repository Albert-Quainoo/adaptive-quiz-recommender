"""Course-neutral validation for a reviewed question-intent blueprint.

authoring/pilot_blueprint.py's validate_pilot_blueprint() is hardcoded to one
specific six-skill, 24-intent batch (fixed skill list, fixed inventory, fixed
AI-SRC-03 wording). It cannot validate any other blueprint file under
authoring/blueprints/ -- including the course-scale files (grounded-dsa-v1.json
and friends) that the replenishment worker already resolves and generates
against.

This module checks the invariants that generalize across every blueprint,
independent of which course or skills it covers:

- an approved blueprint carries a reviewer and a timestamp
- every intent's declared skill_id is a real taxonomy skill
- every intent's learning_objective, when set, is that skill's exact
  canonical wording (never a paraphrase)
- every intent has the review-required narrative fields (cognitive demand,
  required characteristics, prohibited ambiguity patterns, misconception
  strategy) -- an empty one is a silent content-quality gap, not something
  automated generation should paper over
- every preferred_reference_id resolves to an approved, same-skill reference
  in taxonomy/data/<course>/reference_provenance.csv (the same grounding
  check the worker's generation step depends on implicitly)
- every skill's intents declare one consistent, explicit difficulty --
  worker.py's _blueprint_generation_difficulty fails a job closed on this at
  generation time; this catches the same defect before a job is ever queued
- within each skill, intents appear in intent_id order -- deterministic
  scheduling depends on blueprint order being stable and inspectable
"""

from collections import defaultdict
from collections.abc import Sequence

from authoring.question_intents import PilotBlueprint
from taxonomy.schemas import ReferenceProvenance, SkillDefinition


def validate_replenishment_blueprint(
    blueprint: PilotBlueprint,
    skills: Sequence[SkillDefinition],
    provenance: Sequence[ReferenceProvenance],
) -> dict[str, object]:
    """Validate a blueprint against invariants that hold for any course."""
    if blueprint.review_status == "blueprint-approved":
        if not blueprint.reviewer_id or blueprint.reviewed_at is None:
            raise ValueError("an approved blueprint needs a reviewer and timestamp")
    if blueprint.questions_per_intent != 1:
        raise ValueError("a blueprint must generate exactly one question per intent")

    by_skill_definition = {skill.skill_id: skill for skill in skills}
    references = {record.reference_id: record for record in provenance}

    grouped: dict[str, list] = defaultdict(list)
    for intent in blueprint.intents:
        grouped[intent.skill_id].append(intent)

    for skill_id, intents in grouped.items():
        if skill_id not in by_skill_definition:
            raise ValueError(f"{skill_id} is not a defined taxonomy skill")

        ids_in_order = [intent.intent_id for intent in intents]
        if ids_in_order != sorted(ids_in_order):
            raise ValueError(f"{skill_id} intents are not in deterministic order")

        declared_difficulties = {intent.difficulty for intent in intents}
        if len(declared_difficulties) != 1 or None in declared_difficulties:
            offending = ", ".join(f"{i.intent_id}={i.difficulty}" for i in intents)
            raise ValueError(
                f"{skill_id} blueprint intents do not declare one consistent, "
                f"explicit difficulty: {offending}"
            )

    for intent in blueprint.intents:
        skill = by_skill_definition.get(intent.skill_id)
        if (
            skill is not None
            and intent.learning_objective is not None
            and intent.learning_objective != skill.learning_objective
        ):
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

    return {
        "batch_id": blueprint.batch_id,
        "intent_count": len(blueprint.intents),
        "skill_ids": sorted(grouped),
        "difficulty_counts": {
            skill_id: next(iter({i.difficulty for i in intents}))
            for skill_id, intents in grouped.items()
        },
        "checks": {
            "reviewer_and_timestamp_present": True,
            "one_question_per_intent": True,
            "every_skill_id_is_defined": True,
            "deterministic_ordering_per_skill": True,
            "one_consistent_difficulty_per_skill": True,
            "canonical_learning_objectives": True,
            "narrative_fields_present": True,
            "approved_reference_mappings": True,
        },
    }
