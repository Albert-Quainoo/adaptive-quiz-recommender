import pytest

from taxonomy.loader import TaxonomyError, load_skills

HEADER = (
    "skill_id,topic,subtopic,name,learning_objective,"
    "cognitive_process,generation_strategy,"
    "reference_material,prerequisite_skill_ids"
)

VALID_ROW = (
    "AI-SRC-08,Search and Problem Solving,Informed search,Heuristic function,"
    "Explain how a heuristic estimates the remaining cost from a state to the goal.,"
    "understand,generated,"
    '"A heuristic estimates the remaining cost from a state to a goal.",'
)


PREREQUISITE_ROW = (
    "AI-SRC-07,Search and Problem Solving,Informed search,Greedy best-first search,"
    "Describe how greedy search expands the lowest heuristic node.,"
    "understand,generated,,"
)


def write_csv(tmp_path, *rows):
    path = tmp_path / "skills.csv"
    path.write_text("\n".join((HEADER,) + rows) + "\n", encoding="utf-8")
    return path


def test_loads_a_valid_row(tmp_path):
    catalogue = load_skills(write_csv(tmp_path, VALID_ROW))

    skill = catalogue.skills[0]
    assert skill.skill_id == "AI-SRC-08"
    assert skill.learning_objective.startswith("Explain how a heuristic")
    assert skill.reference_material == [
        "A heuristic estimates the remaining cost from a state to a goal.",
    ]
    assert skill.prerequisite_skill_ids == []


def test_list_cells_are_split_on_semicolons(tmp_path):
    row = (
        "AI-SRC-09,Search and Problem Solving,Informed search,A* search,"
        "Apply A* search to a weighted graph.,"
        "apply,generated,"
        '"A* expands the lowest f-cost node.;f(n) = g(n) + h(n)",'
        '"AI-SRC-08; ai-src-07 "'
    )

    catalogue = load_skills(write_csv(tmp_path, PREREQUISITE_ROW, VALID_ROW, row))

    skill = catalogue.skills[2]
    assert skill.reference_material == [
        "A* expands the lowest f-cost node.",
        "f(n) = g(n) + h(n)",
    ]
    assert skill.prerequisite_skill_ids == ["AI-SRC-08", "AI-SRC-07"]


def test_prose_commas_survive_the_load(tmp_path):
    row = (
        "AI-SRC-10,Search and Problem Solving,Informed search,Admissibility,"
        "Explain why an admissible heuristic never overestimates.,"
        "understand,generated,"
        '"A heuristic is admissible if it never overestimates the true cost, '
        'so A* stays optimal.",'
    )

    catalogue = load_skills(write_csv(tmp_path, row))

    assert catalogue.skills[0].reference_material == [
        "A heuristic is admissible if it never overestimates the true cost, "
        "so A* stays optimal.",
    ]


def test_spellings_and_padding_are_normalised_on_load(tmp_path):
    row = (
        " ai-src-11 ,Search and Problem Solving,  Informed search  ,Comparison,"
        "  Analyse the tradeoffs between greedy search and A*.  ,"
        " Analyze ,Hand authored,,"
    )

    skill = load_skills(write_csv(tmp_path, row)).skills[0]

    assert skill.skill_id == "AI-SRC-11"
    assert skill.subtopic == "Informed search"
    assert skill.cognitive_process == "analyse"
    assert skill.generation_strategy == "hand_authored"


def test_every_invalid_row_is_reported_together(tmp_path):
    bad_process = VALID_ROW.replace(",understand,", ",create,")
    bad_id = VALID_ROW.replace("AI-SRC-08,", "SRC-12,")

    with pytest.raises(TaxonomyError) as failure:
        load_skills(write_csv(tmp_path, bad_process, bad_id))

    message = str(failure.value)
    assert "line 2: cognitive_process" in message
    assert "line 3: skill_id" in message


def test_dangling_prerequisite_is_reported(tmp_path):
    row = VALID_ROW + "AI-SRC-99"

    with pytest.raises(TaxonomyError, match="AI-SRC-08 -> AI-SRC-99"):
        load_skills(write_csv(tmp_path, row))


def test_duplicate_skill_ids_are_reported(tmp_path):
    with pytest.raises(TaxonomyError, match="Duplicate skill ids: AI-SRC-08"):
        load_skills(write_csv(tmp_path, VALID_ROW, VALID_ROW))
