"""Validate an approved bank and write a deterministic readiness summary."""

import argparse
import json
from collections import Counter
from pathlib import Path

from app.bootstrap import load_approved_bank
from taxonomy.loader import course_paths, load_skills


def validate_bank(
    bank_path: Path,
    *,
    course: str,
    expected_count: int,
    required_skill_ids: list[str],
    skills_path: Path | None = None,
    references_path: Path | None = None,
) -> dict:
    """skills_path/references_path let a caller whose taxonomy directory
    doesn't share its manifest's course_id (e.g. authoring/course_catalog's
    readiness checks) point at the real files directly, instead of relying
    on course_paths(course)'s taxonomy/data/{course}/ convention."""
    items = load_approved_bank(bank_path)
    item_ids = [item.item_id for item in items]
    stems = [" ".join(item.question.question.casefold().split()) for item in items]
    if skills_path is None or references_path is None:
        skills_path, references_path = course_paths(course)
    known_skills = {
        skill.skill_id for skill in load_skills(skills_path, references_path).skills
    }
    item_skills = {item.skill_id for item in items}
    unknown_skills = sorted(item_skills - known_skills)
    missing_skills = sorted(set(required_skill_ids) - item_skills)

    if len(items) != expected_count:
        raise ValueError(f"expected {expected_count} items, found {len(items)}")
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("approved bank contains duplicate item IDs")
    if len(stems) != len(set(stems)):
        raise ValueError("approved bank contains duplicate normalized stems")
    if unknown_skills:
        raise ValueError("approved bank contains unknown skills: " + ", ".join(unknown_skills))
    if missing_skills:
        raise ValueError("approved bank is missing required skills: " + ", ".join(missing_skills))

    return {
        "bank_path": str(bank_path),
        "status": "ready",
        "item_count": len(items),
        "unique_item_ids": len(set(item_ids)),
        "unique_normalized_stems": len(set(stems)),
        "required_skill_ids": required_skill_ids,
        "skill_counts": dict(sorted(Counter(item.skill_id for item in items).items())),
        "difficulty_counts": dict(
            sorted(Counter(item.question.difficulty for item in items).items())
        ),
        "unknown_skill_ids": [],
        "missing_required_skill_ids": [],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--course", default="ai")
    parser.add_argument("--expected-count", type=int, required=True)
    parser.add_argument("--required-skill-id", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(argv)
    summary = validate_bank(
        arguments.bank,
        course=arguments.course,
        expected_count=arguments.expected_count,
        required_skill_ids=arguments.required_skill_id,
    )
    if arguments.output.exists():
        raise ValueError(f"validation summary already exists: {arguments.output}")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
