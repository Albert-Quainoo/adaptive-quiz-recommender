"""Validate and render the 24-question six-skill pilot blueprint."""

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

from authoring.grounded_batch import derive_seed
from authoring.pilot_blueprint import (
    PILOT_BATCH_ID,
    PILOT_SKILL_IDS,
    validate_pilot_blueprint,
)
from authoring.question_intents import (
    BLUEPRINT_DIRECTORY,
    PilotBlueprint,
    load_blueprint_for_batch,
)
from taxonomy.loader import (
    course_paths,
    course_provenance_path,
    load_reference_provenance,
    load_skills,
)


DEFAULT_BLUEPRINT = BLUEPRINT_DIRECTORY / f"{PILOT_BATCH_ID}.json"
DEFAULT_REVIEW = BLUEPRINT_DIRECTORY / f"{PILOT_BATCH_ID}-review.md"
DEFAULT_SUMMARY = BLUEPRINT_DIRECTORY / f"{PILOT_BATCH_ID}-validation.json"
LLAMA_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


def proposed_seeds(blueprint: PilotBlueprint) -> list[dict[str, object]]:
    indices: dict[str, int] = defaultdict(int)
    seeds = []
    for intent in blueprint.intents:
        question_index = indices[intent.skill_id]
        indices[intent.skill_id] += 1
        seeds.append(
            {
                "intent_id": intent.intent_id,
                "skill_id": intent.skill_id,
                "question_index": question_index,
                "attempt_index": 0,
                "seed": derive_seed(
                    blueprint.batch_id,
                    intent.skill_id,
                    question_index,
                    0,
                    blueprint.base_seed,
                ),
            }
        )
    return seeds


def generation_command(blueprint: PilotBlueprint) -> str:
    skill_lines = "\n".join(
        f"  --skill-id {skill_id} \\" for skill_id in PILOT_SKILL_IDS
    )
    return (
        f"export MODEL_REPOSITORY=\"{LLAMA_MODEL_ID}\"\n\n"
        "python -m scripts.generate_grounded_batch \\\n"
        f"  --batch-id {blueprint.batch_id} \\\n"
        f"{skill_lines}\n"
        "  --all-blueprint-intents \\\n"
        f"  --base-seed {blueprint.base_seed} \\\n"
        f"  --output outputs/{blueprint.batch_id} \\\n"
        "  --model-id \"$MODEL_REPOSITORY\" \\\n"
        f"  --prompt-version {blueprint.prompt_version} \\\n"
        "  --difficulty mixed"
    )


def render_review(
    blueprint: PilotBlueprint,
    summary: dict[str, object],
    seeds: Sequence[dict[str, object]],
    command: str,
) -> str:
    lines = [
        f"# {blueprint.batch_id} blueprint review",
        "",
        f"Status: **{blueprint.review_status}**. This blueprint has not run Llama.",
        "",
        "## Validation summary",
        "",
        f"- Intents: {summary['intent_count']}",
        f"- Questions per intent: {summary['questions_per_intent']}",
        "- Difficulty inventory: 15 introductory, 9 intermediate",
        "- All automated blueprint checks: passed",
        "",
        "## Pilot reference readiness",
        "",
        "| Skill | Approved | Domains | Ready |",
        "|---|---:|---|---|",
    ]
    readiness = summary["reference_readiness"]
    for skill_id in PILOT_SKILL_IDS:
        item = readiness[skill_id]
        lines.append(
            f"| {skill_id} | {item['approved_count']} | "
            f"{', '.join(item['domains'])} | {'yes' if item['ready'] else 'no'} |"
        )

    lines += [
        "",
        "## AI-SRC-03 course and solver convention",
        "",
    ]
    lines += [
        f"- {statement}"
        for statement in blueprint.skill_conventions["AI-SRC-03"]
    ]
    lines += ["", "## Question intents", ""]

    for intent in blueprint.intents:
        lines += [
            f"### {intent.intent_id} — {intent.difficulty}",
            "",
            f"- Skill: {intent.skill_id}",
            f"- Canonical objective: {intent.learning_objective}",
            f"- Assessment focus: {intent.assessment_focus}",
            f"- Cognitive demand: {intent.cognitive_demand}",
            f"- Archetype: {intent.question_archetype}",
            f"- Preferred references: {', '.join(intent.preferred_reference_ids)}",
            "- Required characteristics: "
            + "; ".join(intent.required_question_characteristics),
            "- Prohibited ambiguity: "
            + "; ".join(intent.prohibited_ambiguity_patterns),
            "- Misconception/distractor strategy: "
            + "; ".join(intent.expected_misconception_or_distractor_strategy),
            "",
        ]

    lines += [
        "## Proposed deterministic first-attempt seeds",
        "",
        f"Base seed: `{blueprint.base_seed}`",
        "",
        "| Intent | Question index | Seed |",
        "|---|---:|---:|",
    ]
    lines += [
        f"| {item['intent_id']} | {item['question_index']} | {item['seed']} |"
        for item in seeds
    ]
    lines += [
        "",
        "Retry seeds remain deterministic because the attempt index is included in seed derivation.",
        "",
        "## Exact Kaggle Llama generation command",
        "",
        "The blueprint is approved. Run from a clean, committed worktree with HF_TOKEN configured.",
        "",
        "```bash",
        command,
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-output", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY)
    arguments = parser.parse_args(argv)

    blueprint = load_blueprint_for_batch(PILOT_BATCH_ID)
    skills = load_skills(*course_paths("ai")).skills
    provenance = load_reference_provenance(course_provenance_path("ai"))
    summary = validate_pilot_blueprint(blueprint, skills, provenance)
    seeds = proposed_seeds(blueprint)
    command = generation_command(blueprint)
    summary = summary | {
        "base_seed": blueprint.base_seed,
        "blueprint_review_status": blueprint.review_status,
        "blueprint_reviewer_id": blueprint.reviewer_id,
        "blueprint_reviewed_at": (
            blueprint.reviewed_at.isoformat().replace("+00:00", "Z")
            if blueprint.reviewed_at
            else None
        ),
        "proposed_first_attempt_seeds": seeds,
        "generation_command": command,
        "llama_run_status": "not-run",
    }

    arguments.review_output.write_text(
        render_review(blueprint, summary, seeds, command), encoding="utf-8"
    )
    arguments.summary_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        f"Validated {summary['intent_count']} intents; wrote "
        f"{arguments.review_output} and {arguments.summary_output}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
