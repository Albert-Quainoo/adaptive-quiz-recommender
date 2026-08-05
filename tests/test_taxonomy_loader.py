import pytest

from taxonomy.loader import TaxonomyError, load_skills

SKILL_HEADER = (
    "skill_id,topic,subtopic,name,learning_objective,"
    "cognitive_process,generation_strategy,template_id,prerequisite_skill_ids"
)

REFERENCE_HEADER = "skill_id,reference_material"

# Transcribed from TAXONOMY.docx, Topic 3: Search and Problem Solving.
HEURISTIC_ROW = (
    "AI-SRC-08,Search and Problem Solving,Informed search,Heuristic function,"
    "Explain how a heuristic estimates the remaining cost from a state to the goal.,"
    "understand,generated,,"
)

GREEDY_ROW = (
    "AI-SRC-09,Search and Problem Solving,Informed search,Greedy Best-First Search,"
    "Trace Greedy Best-First Search using (f(n)=h(n)).,"
    "apply,templated,search.greedy_trace,AI-SRC-08"
)

A_STAR_ROW = (
    "AI-SRC-10,Search and Problem Solving,Informed search,A-star Search,"
    "Trace A-star Search using (f(n)=g(n)+h(n)).,"
    "apply,templated,search.astar_trace,"
    '"AI-SRC-08; ai-src-09 "'
)


def write_taxonomy(
    tmp_path,
    skill_rows=(HEURISTIC_ROW,),
    reference_rows=(),
    skill_header=SKILL_HEADER,
    reference_header=REFERENCE_HEADER,
):
    skills_path = tmp_path / "skills.csv"
    references_path = tmp_path / "references.csv"

    skills_path.write_text(
        "\n".join((skill_header, *skill_rows)) + "\n", encoding="utf-8"
    )
    references_path.write_text(
        "\n".join((reference_header, *reference_rows)) + "\n", encoding="utf-8"
    )

    return skills_path, references_path


def test_loads_a_valid_skill(tmp_path):
    catalogue = load_skills(*write_taxonomy(tmp_path))

    skill = catalogue.skills[0]
    assert skill.skill_id == "AI-SRC-08"
    assert skill.name == "Heuristic function"
    assert skill.reference_material == []
    assert skill.prerequisite_skill_ids == []


def test_references_are_attached_to_their_skill(tmp_path):
    paths = write_taxonomy(
        tmp_path,
        skill_rows=(HEURISTIC_ROW, GREEDY_ROW),
        reference_rows=(
            "AI-SRC-08,A heuristic estimates the remaining cost from a state to a goal.",
            "AI-SRC-09,Greedy best-first search expands the node with the lowest h(n).",
            "AI-SRC-08,An admissible heuristic never overestimates the true cost.",
        ),
    )

    catalogue = load_skills(*paths)

    assert catalogue.skills[0].reference_material == [
        "A heuristic estimates the remaining cost from a state to a goal.",
        "An admissible heuristic never overestimates the true cost.",
    ]
    assert catalogue.skills[1].reference_material == [
        "Greedy best-first search expands the node with the lowest h(n).",
    ]


def test_reference_punctuation_survives_the_load(tmp_path):
    reference = (
        "A-star expands the frontier node with the lowest f(n); "
        "it is optimal when the heuristic is admissible, and complete on finite graphs."
    )

    paths = write_taxonomy(
        tmp_path,
        reference_rows=(f'AI-SRC-08,"{reference}"',),
    )

    assert load_skills(*paths).skills[0].reference_material == [reference]


def test_template_id_is_loaded_for_templated_skills(tmp_path):
    catalogue = load_skills(
        *write_taxonomy(tmp_path, skill_rows=(HEURISTIC_ROW, GREEDY_ROW))
    )

    assert catalogue.skills[0].template_id is None
    assert catalogue.skills[1].template_id == "search.greedy_trace"


def test_templated_row_without_a_template_id_is_rejected(tmp_path):
    row = GREEDY_ROW.replace(",search.greedy_trace,", ",,")

    with pytest.raises(TaxonomyError, match="templated skill needs a template_id"):
        load_skills(*write_taxonomy(tmp_path, skill_rows=(HEURISTIC_ROW, row)))


def test_generated_row_with_a_template_id_is_rejected(tmp_path):
    row = HEURISTIC_ROW.replace(",generated,,", ",generated,search.astar_trace,")

    with pytest.raises(TaxonomyError, match="cannot have a template_id"):
        load_skills(*write_taxonomy(tmp_path, skill_rows=(row,)))


def test_prerequisite_cells_are_split_on_semicolons(tmp_path):
    paths = write_taxonomy(
        tmp_path,
        skill_rows=(HEURISTIC_ROW, GREEDY_ROW, A_STAR_ROW),
    )

    catalogue = load_skills(*paths)

    assert catalogue.skills[2].prerequisite_skill_ids == ["AI-SRC-08", "AI-SRC-09"]


def test_spellings_and_padding_are_normalised_on_load(tmp_path):
    padded_row = (
        " ai-src-11 ,Search and Problem Solving,  Search evaluation  ,"
        "Completeness and optimality,"
        "  Compare search algorithms based on completeness and optimality.  ,"
        " Analyze ,Hand authored,,"
    )

    skill = load_skills(*write_taxonomy(tmp_path, skill_rows=(padded_row,))).skills[0]

    assert skill.skill_id == "AI-SRC-11"
    assert skill.subtopic == "Search evaluation"
    assert skill.cognitive_process == "analyse"
    assert skill.generation_strategy == "hand_authored"


def test_references_match_a_padded_skill_id(tmp_path):
    paths = write_taxonomy(
        tmp_path,
        reference_rows=(" ai-src-08 ,A heuristic estimates the remaining cost.",),
    )

    assert load_skills(*paths).skills[0].reference_material == [
        "A heuristic estimates the remaining cost.",
    ]


def test_every_invalid_row_is_reported_together(tmp_path):
    bad_process = HEURISTIC_ROW.replace(",understand,", ",create,")
    bad_id = GREEDY_ROW.replace("AI-SRC-09,", "SRC-12,")

    with pytest.raises(TaxonomyError) as failure:
        load_skills(*write_taxonomy(tmp_path, skill_rows=(bad_process, bad_id)))

    message = str(failure.value)
    assert "skills.csv line 2: cognitive_process" in message
    assert "skills.csv line 3: skill_id" in message


def test_reference_for_an_unknown_skill_is_reported(tmp_path):
    paths = write_taxonomy(
        tmp_path,
        reference_rows=("AI-SRC-99,A reference for a skill that does not exist.",),
    )

    with pytest.raises(TaxonomyError, match="unknown skill id AI-SRC-99"):
        load_skills(*paths)


def test_duplicate_reference_rows_are_reported(tmp_path):
    reference = "AI-SRC-08,A heuristic estimates the remaining cost."

    with pytest.raises(TaxonomyError, match="duplicate reference for AI-SRC-08"):
        load_skills(*write_taxonomy(tmp_path, reference_rows=(reference, reference)))


def test_empty_reference_text_is_reported(tmp_path):
    with pytest.raises(TaxonomyError, match="reference_material is empty"):
        load_skills(*write_taxonomy(tmp_path, reference_rows=("AI-SRC-08,",)))


def test_skill_row_with_an_extra_cell_is_rejected(tmp_path):
    with pytest.raises(TaxonomyError, match="unexpected extra value"):
        load_skills(*write_taxonomy(tmp_path, skill_rows=(HEURISTIC_ROW + ",extra",)))


def test_skill_row_with_a_missing_cell_is_rejected(tmp_path):
    short_row = HEURISTIC_ROW.rstrip(",")

    with pytest.raises(TaxonomyError, match="missing cells for prerequisite_skill_ids"):
        load_skills(*write_taxonomy(tmp_path, skill_rows=(short_row,)))


def test_header_only_skills_file_is_rejected(tmp_path):
    with pytest.raises(TaxonomyError, match="skills"):
        load_skills(*write_taxonomy(tmp_path, skill_rows=()))


def test_header_only_references_file_is_accepted(tmp_path):
    catalogue = load_skills(*write_taxonomy(tmp_path, reference_rows=()))

    assert catalogue.skills[0].reference_material == []


def test_empty_skills_file_is_rejected(tmp_path):
    skills_path, references_path = write_taxonomy(tmp_path)
    skills_path.write_text("", encoding="utf-8")

    with pytest.raises(TaxonomyError, match="skills.csv: the file has no header row"):
        load_skills(skills_path, references_path)


def test_empty_references_file_is_rejected(tmp_path):
    skills_path, references_path = write_taxonomy(tmp_path)
    references_path.write_text("", encoding="utf-8")

    with pytest.raises(
        TaxonomyError, match="references.csv: the file has no header row"
    ):
        load_skills(skills_path, references_path)


def test_missing_skill_column_is_reported(tmp_path):
    header = SKILL_HEADER.replace("skill_id,", "", 1)

    with pytest.raises(TaxonomyError, match="missing columns: skill_id"):
        load_skills(*write_taxonomy(tmp_path, skill_header=header))


def test_misspelled_column_is_reported_as_missing_and_unexpected(tmp_path):
    header = SKILL_HEADER.replace("learning_objective", "learning_objectve")

    with pytest.raises(TaxonomyError) as failure:
        load_skills(*write_taxonomy(tmp_path, skill_header=header))

    message = str(failure.value)
    assert "missing columns: learning_objective" in message
    assert "unexpected columns: learning_objectve" in message


def test_reference_material_column_in_the_skills_file_is_rejected(tmp_path):
    header = SKILL_HEADER + ",reference_material"

    with pytest.raises(TaxonomyError, match="unexpected columns: reference_material"):
        load_skills(*write_taxonomy(tmp_path, skill_header=header))


def test_duplicate_column_is_reported(tmp_path):
    header = SKILL_HEADER + ",topic"

    with pytest.raises(TaxonomyError, match="duplicate columns: topic"):
        load_skills(*write_taxonomy(tmp_path, skill_header=header))


def test_padded_header_is_accepted(tmp_path):
    header = SKILL_HEADER.replace("skill_id,", " skill_id ,", 1)

    catalogue = load_skills(*write_taxonomy(tmp_path, skill_header=header))

    assert catalogue.skills[0].skill_id == "AI-SRC-08"


def test_dangling_prerequisite_is_reported(tmp_path):
    row = HEURISTIC_ROW + "AI-SRC-99"

    with pytest.raises(TaxonomyError, match="AI-SRC-08 -> AI-SRC-99"):
        load_skills(*write_taxonomy(tmp_path, skill_rows=(row,)))


def test_skills_from_two_courses_in_one_file_are_reported(tmp_path):
    other_course_row = HEURISTIC_ROW.replace("AI-SRC-08,", "DB-SQL-01,", 1)

    with pytest.raises(TaxonomyError, match="one file must hold one course, found AI, DB"):
        load_skills(
            *write_taxonomy(tmp_path, skill_rows=(HEURISTIC_ROW, other_course_row))
        )


def test_duplicate_skill_ids_are_reported(tmp_path):
    with pytest.raises(TaxonomyError, match="Duplicate skill ids: AI-SRC-08"):
        load_skills(*write_taxonomy(tmp_path, skill_rows=(HEURISTIC_ROW, HEURISTIC_ROW)))
