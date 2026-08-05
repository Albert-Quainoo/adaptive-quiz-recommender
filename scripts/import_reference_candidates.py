"""Import approved retrieval candidates into the canonical taxonomy."""

import argparse
import csv
from collections.abc import Sequence
from pathlib import Path

from authoring.retrieval.store import CandidateStore
from taxonomy.loader import (
    REFERENCE_COLUMNS,
    REFERENCE_PROVENANCE_COLUMNS,
    TaxonomyError,
    course_paths,
    course_provenance_path,
    load_reference_provenance,
    load_skills,
)
from taxonomy.schemas import ReferenceProvenance


def write_rows(path: Path, columns: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")

    with temporary.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    temporary.replace(path)


def provenance_for(candidate) -> ReferenceProvenance:
    return ReferenceProvenance(
        reference_id=candidate.candidate_id,
        skill_id=candidate.skill_id,
        reference_material=candidate.passage,
        title=candidate.title,
        source_url=candidate.source_url,
        source_domain=candidate.source_domain,
        content_hash=candidate.content_hash,
        retrieved_at=candidate.retrieved_at,
        reviewer_id=candidate.reviewer_id,
        reviewed_at=candidate.reviewed_at,
        review_note=candidate.review_note,
    )


def import_candidates(
    store_path: Path,
    skills_path: Path,
    references_path: Path,
    provenance_path: Path,
) -> list[ReferenceProvenance]:
    """Import approved candidates, returning only records added this run."""
    catalogue = load_skills(skills_path, references_path)
    known_skill_ids = {skill.skill_id for skill in catalogue.skills}
    existing = (
        load_reference_provenance(provenance_path, known_skill_ids)
        if provenance_path.exists()
        else []
    )
    approved = [
        candidate
        for candidate in CandidateStore(store_path).load()
        if candidate.review_status == "approved"
    ]

    unknown = sorted(
        {candidate.skill_id for candidate in approved} - known_skill_ids
    )
    if unknown:
        raise TaxonomyError(
            "approved candidates reference unknown skill id " + ", ".join(unknown)
        )

    by_id = {record.reference_id: record for record in existing}
    content_keys = {(record.skill_id, record.content_hash) for record in existing}
    imported: list[ReferenceProvenance] = []

    for candidate in sorted(approved, key=lambda item: item.candidate_id):
        record = provenance_for(candidate)
        held = by_id.get(record.reference_id)

        if held is not None:
            if held != record:
                raise TaxonomyError(
                    f"reference id {record.reference_id} conflicts with canonical provenance"
                )
            continue

        content_key = (record.skill_id, record.content_hash)
        if content_key in content_keys:
            continue

        by_id[record.reference_id] = record
        content_keys.add(content_key)
        imported.append(record)

    records = sorted(by_id.values(), key=lambda item: (item.skill_id, item.reference_id))
    references = {
        skill.skill_id: list(skill.reference_material) for skill in catalogue.skills
    }

    for record in records:
        if record.reference_material not in references[record.skill_id]:
            references[record.skill_id].append(record.reference_material)

    reference_rows = [
        {"skill_id": skill.skill_id, "reference_material": passage}
        for skill in catalogue.skills
        for passage in references[skill.skill_id]
    ]
    provenance_rows = [
        record.model_dump(mode="json") for record in records
    ]

    write_rows(references_path, REFERENCE_COLUMNS, reference_rows)
    write_rows(provenance_path, REFERENCE_PROVENANCE_COLUMNS, provenance_rows)

    return imported


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Import approved reference candidates with provenance."
    )
    parser.add_argument("store", type=Path, help="reviewed candidate JSON store")
    parser.add_argument("--course", default="ai")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    skills_path, references_path = course_paths(arguments.course)
    imported = import_candidates(
        arguments.store,
        skills_path,
        references_path,
        course_provenance_path(arguments.course),
    )
    print(f"Imported {len(imported)} approved reference(s).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
