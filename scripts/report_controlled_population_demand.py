"""Read-only demand analysis for the controlled bank-population phase.

Composes existing application code -- never reimplements it:
  - authoring.replenishment.manifest.load_preparation_eligible_manifests() for
    which courses are in scope,
  - taxonomy.loader.load_skills() for the skill catalogue (generation_strategy),
  - authoring.replenishment.demand.load_approved_item_ids() /
    compute_blueprint_slot_demand() for the exact same slot-satisfaction check
    the worker itself uses before spending a model call,
  - authoring.replenishment.worker.blueprints_covering_skill() for blueprint
    resolution, identical to what the worker does per job,
  - authoring.deterministic_templates.DETERMINISTIC_TEMPLATES for which
    deficient intents already have a registered template fallback,
  - authoring.replenishment.jobs.open_repository(read_only=True) for the
    latest job per skill, used only as a signal for Group C classification
    (a skill with a prior permanent_failure is flagged, never silently
    retried).

Writes nothing, calls no network, opens the job repository read-only. Output
is the exact set of still-unfilled approved-bank slots, grouped A/B/C per the
controlled-population plan.

    python -m scripts.report_controlled_population_demand [--json out.json]
"""

import argparse
import json
import sys
from pathlib import Path

from app.bootstrap import BootstrapError, load_approved_bank
from authoring.deterministic_templates import DETERMINISTIC_TEMPLATES
from authoring.replenishment.cli import DEFAULT_DATABASE_PATH
from authoring.replenishment.demand import (
    compute_blueprint_slot_demand,
    deficient_slots,
    load_approved_item_ids,
)
from authoring.replenishment.jobs import open_repository
from authoring.replenishment.manifest import (
    CourseManifest,
    active_bank_path,
    load_preparation_eligible_manifests,
)
from authoring.replenishment.worker import blueprints_covering_skill
from taxonomy.loader import load_skills


def _route_for_intent(intent_id: str) -> str:
    return "deterministic_template_fallback" if intent_id in DETERMINISTIC_TEMPLATES else "normal_model_generation"


def analyze_course(manifest: CourseManifest, job_repository) -> list[dict]:
    catalogue = load_skills(manifest.skills_path(), manifest.references_path())
    approved_item_ids = load_approved_item_ids(manifest)
    try:
        bank_items = load_approved_bank(active_bank_path(manifest))
    except (BootstrapError, FileNotFoundError):
        bank_items = []

    rows: list[dict] = []
    for skill in catalogue.skills:
        if skill.generation_strategy == "templated":
            continue

        latest_job = job_repository.latest_for_skill(manifest.course_id, skill.skill_id)
        prior_permanent_failure = latest_job is not None and latest_job.status == "permanent_failure"

        try:
            covering = blueprints_covering_skill(skill.skill_id)
        except Exception as exc:  # defensive: a malformed blueprint file must not abort the scan
            rows.append(
                {
                    "course_id": manifest.course_id,
                    "skill_id": skill.skill_id,
                    "intent_id": None,
                    "difficulty": None,
                    "current_approved_supply": None,
                    "remaining_demand": None,
                    "generation_route": None,
                    "group": "C",
                    "reason": f"blueprint_resolution_error: {exc}",
                }
            )
            continue

        if not covering:
            continue  # no reviewed blueprint yet -- out of scope for this phase, not a deficiency to report

        ambiguous = len(covering) > 1
        current_supply = sum(1 for item in bank_items if item.skill_id == skill.skill_id)

        for blueprint in covering:
            demand = compute_blueprint_slot_demand(blueprint, skill.skill_id, approved_item_ids)
            for slot in deficient_slots(demand):
                route = _route_for_intent(slot.intent_id)
                if ambiguous:
                    group, reason = "C", "ambiguous_blueprint_coverage"
                elif prior_permanent_failure:
                    group, reason = "C", f"prior_permanent_failure: {latest_job.error_code}"
                elif route == "deterministic_template_fallback":
                    group, reason = "B", "registered_deterministic_template"
                else:
                    group, reason = "A", "normal_generation_ready"

                rows.append(
                    {
                        "course_id": manifest.course_id,
                        "skill_id": skill.skill_id,
                        "intent_id": slot.intent_id,
                        "difficulty": slot.difficulty,
                        "current_approved_supply": current_supply,
                        "remaining_demand": 1,
                        "generation_route": route,
                        "batch_id": blueprint.batch_id,
                        "group": group,
                        "reason": reason,
                    }
                )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=str, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--json", type=Path, default=None, help="also write the full row set as JSON")
    args = parser.parse_args(argv)

    repository = open_repository(args.database, read_only=True)

    all_rows: list[dict] = []
    for manifest in load_preparation_eligible_manifests():
        all_rows.extend(analyze_course(manifest, repository))

    by_group: dict[str, list[dict]] = {"A": [], "B": [], "C": []}
    for row in all_rows:
        by_group[row["group"]].append(row)

    print(f"Total unfilled approved-bank slots: {len(all_rows)}")
    for group, label in (("A", "normal-generation ready"), ("B", "deterministic-template capable"), ("C", "unsupported / high-risk")):
        rows = by_group[group]
        print(f"\n== Group {group}: {label} ({len(rows)}) ==")
        for row in sorted(rows, key=lambda r: (r["course_id"], r["skill_id"], r["intent_id"] or "")):
            print(
                f"  {row['course_id']:<16} {row['skill_id']:<14} {row['intent_id']:<20} "
                f"diff={row['difficulty']!s:<8} supply={row['current_approved_supply']} "
                f"route={row['generation_route']:<28} reason={row['reason']}"
            )

    if args.json:
        args.json.write_text(json.dumps(all_rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"\nWrote {len(all_rows)} rows to {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
