import csv
from datetime import datetime, timezone

import pytest

from authoring.retrieval.models import approve, new_candidate, reject
from authoring.retrieval.store import CandidateStore
from scripts.import_reference_candidates import import_candidates
from taxonomy.loader import (
    REFERENCE_PROVENANCE_COLUMNS,
    TaxonomyError,
    course_paths,
    load_reference_provenance,
    load_skills,
)

SKILLS_PATH, _ = course_paths("ai")
REVIEWED_AT = datetime(2026, 8, 5, 15, 0, tzinfo=timezone.utc)
RETRIEVED_AT = datetime(2026, 8, 5, 14, 0, tzinfo=timezone.utc)


def candidate(skill_id="AI-SRC-08", passage="A heuristic estimates remaining cost."):
    return new_candidate(
        skill_id=skill_id,
        title="Heuristic functions",
        source_url="https://inst.eecs.berkeley.edu/~cs188/textbook/search/informed.html",
        source_domain="inst.eecs.berkeley.edu",
        passage=passage,
        retrieved_at=RETRIEVED_AT,
    )


def paths(tmp_path):
    store_path = tmp_path / "candidates.json"
    references_path = tmp_path / "references.csv"
    provenance_path = tmp_path / "reference_provenance.csv"
    references_path.write_text("skill_id,reference_material\n", encoding="utf-8")

    return store_path, references_path, provenance_path


def approved(item, note="Matches the objective."):
    return approve(
        item,
        "albert",
        note=note,
        reviewed_at=REVIEWED_AT,
    )


def test_only_approved_candidates_are_imported(tmp_path):
    store_path, references_path, provenance_path = paths(tmp_path)
    accepted = approved(candidate())
    refused = reject(
        candidate(passage="A heuristic is merely a guess."),
        "albert",
        reviewed_at=REVIEWED_AT,
    )
    pending = candidate(passage="A heuristic ranks states.")
    CandidateStore(store_path).save([accepted, refused, pending])

    imported = import_candidates(
        store_path, SKILLS_PATH, references_path, provenance_path
    )

    assert [record.reference_id for record in imported] == [accepted.candidate_id]
    skill = next(
        skill
        for skill in load_skills(SKILLS_PATH, references_path).skills
        if skill.skill_id == accepted.skill_id
    )
    assert skill.reference_material == [accepted.passage]


def test_every_provenance_field_survives_the_import(tmp_path):
    store_path, references_path, provenance_path = paths(tmp_path)
    accepted = approved(candidate(), note="Canonical explanation.")
    CandidateStore(store_path).save([accepted])

    import_candidates(store_path, SKILLS_PATH, references_path, provenance_path)
    record = load_reference_provenance(
        provenance_path,
        {skill.skill_id for skill in load_skills(SKILLS_PATH, references_path).skills},
    )[0]

    assert record.reference_id == accepted.candidate_id
    assert record.skill_id == accepted.skill_id
    assert record.reference_material == accepted.passage
    assert record.title == accepted.title
    assert record.source_url == accepted.source_url
    assert record.source_domain == accepted.source_domain
    assert record.content_hash == accepted.content_hash
    assert record.retrieved_at == accepted.retrieved_at
    assert record.reviewer_id == accepted.reviewer_id
    assert record.reviewed_at == accepted.reviewed_at
    assert record.review_note == accepted.review_note


def test_repeated_import_and_duplicate_content_create_no_duplicates(tmp_path):
    store_path, references_path, provenance_path = paths(tmp_path)
    accepted = approved(candidate())
    same_content = accepted.model_copy(
        update={"candidate_id": "AI-SRC-08-duplicate001"}
    )
    CandidateStore(store_path).save([accepted, same_content])

    import_candidates(store_path, SKILLS_PATH, references_path, provenance_path)
    import_candidates(store_path, SKILLS_PATH, references_path, provenance_path)

    records = load_reference_provenance(provenance_path)
    skill = next(
        skill
        for skill in load_skills(SKILLS_PATH, references_path).skills
        if skill.skill_id == "AI-SRC-08"
    )
    assert len(records) == 1
    assert skill.reference_material == [accepted.passage]


def test_reference_loader_still_supplies_passage_text_to_generation(tmp_path):
    store_path, references_path, provenance_path = paths(tmp_path)
    accepted = approved(candidate())
    CandidateStore(store_path).save([accepted])

    import_candidates(store_path, SKILLS_PATH, references_path, provenance_path)

    loaded = load_skills(SKILLS_PATH, references_path)
    skill = next(skill for skill in loaded.skills if skill.skill_id == accepted.skill_id)
    references = {
        skill.skill_id: skill.reference_material
        for skill in loaded.skills
        if skill.reference_material
    }
    assert references == {"AI-SRC-08": [accepted.passage]}


def test_an_approved_candidate_for_an_unknown_skill_aborts_the_import(tmp_path):
    store_path, references_path, provenance_path = paths(tmp_path)
    CandidateStore(store_path).save([approved(candidate(skill_id="AI-SRC-99"))])

    with pytest.raises(TaxonomyError, match="unknown skill id AI-SRC-99"):
        import_candidates(store_path, SKILLS_PATH, references_path, provenance_path)

    assert references_path.read_text(encoding="utf-8") == (
        "skill_id,reference_material\n"
    )
    assert not provenance_path.exists()


def test_provenance_file_has_the_complete_committed_schema(tmp_path):
    store_path, references_path, provenance_path = paths(tmp_path)
    CandidateStore(store_path).save([approved(candidate())])

    import_candidates(store_path, SKILLS_PATH, references_path, provenance_path)

    with provenance_path.open(newline="", encoding="utf-8") as csv_file:
        assert tuple(csv.DictReader(csv_file).fieldnames or ()) == (
            REFERENCE_PROVENANCE_COLUMNS
        )
