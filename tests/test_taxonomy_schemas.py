import pytest
from pydantic import ValidationError

from taxonomy.schemas import (
    SkillCatalogue,
    SkillDefinition,
    find_skills_missing_reference_material,
)

SKILLS = {
    "AI-SRC-07": (
        "Uninformed search",
        "DLS and IDDFS",
        "Explain how iterative deepening combines the strengths of DFS and BFS.",
    ),
    "AI-SRC-08": (
        "Informed search",
        "Heuristic function",
        "Explain how a heuristic estimates the remaining cost from a state to the goal.",
    ),
    "AI-SRC-09": (
        "Informed search",
        "Greedy Best-First Search",
        "Describe how greedy best-first search expands the lowest heuristic node.",
    ),
    "AI-SRC-10": (
        "Informed search",
        "A-star Search",
        "Trace A-star search using f(n) = g(n) + h(n).",
    ),
}


def valid_skill(**overrides) -> SkillDefinition:
    skill_id = str(overrides.get("skill_id", "AI-SRC-08")).strip().upper()
    subtopic, name, learning_objective = SKILLS.get(skill_id, SKILLS["AI-SRC-08"])

    fields = {
        "skill_id": "AI-SRC-08",
        "topic": "Search and Problem Solving",
        "subtopic": subtopic,
        "name": name,
        "learning_objective": learning_objective,
        "cognitive_process": "understand",
        "generation_strategy": "generated",
    }
    fields.update(overrides)
    return SkillDefinition(**fields)


def test_valid_skill_is_accepted():
    skill = valid_skill(
        reference_material=[
            "A heuristic estimates the remaining cost from a state to a goal.",
        ],
    )

    assert skill.skill_id == "AI-SRC-08"
    assert skill.cognitive_process == "understand"
    assert skill.prerequisite_skill_ids == []


@pytest.mark.parametrize(
    "skill_id",
    ["SRC-08", "AI-S-08", "AI-SEARCHING-08", "AI-SRC-8", "AI SRC 08", ""],
)
def test_malformed_skill_id_is_rejected(skill_id):
    with pytest.raises(ValidationError, match="skill_id"):
        valid_skill(skill_id=skill_id)


@pytest.mark.parametrize(
    "skill_id, expected",
    [
        (" AI-SRC-08 ", "AI-SRC-08"),
        ("ai-src-08", "AI-SRC-08"),
        ("AI-SRC-08a", "AI-SRC-08A"),
    ],
)
def test_skill_id_case_and_padding_are_normalised(skill_id, expected):
    assert valid_skill(skill_id=skill_id).skill_id == expected


def test_unsupported_cognitive_process_is_rejected():
    with pytest.raises(ValidationError, match="cognitive_process"):
        valid_skill(cognitive_process="create")


@pytest.mark.parametrize(
    "value", ["analyze", "Analyze", "Analyse", " analyse ", "ANALYZE"]
)
def test_cognitive_process_spelling_is_normalised(value):
    assert valid_skill(cognitive_process=value).cognitive_process == "analyse"


def test_unsupported_generation_strategy_is_rejected():
    with pytest.raises(ValidationError, match="generation_strategy"):
        valid_skill(generation_strategy="scraped")


@pytest.mark.parametrize(
    "value, expected",
    [
        (" Generated ", "generated"),
        ("TEMPLATED", "templated"),
        ("Hand authored", "hand_authored"),
        ("hand-authored", "hand_authored"),
    ],
)
def test_generation_strategy_is_normalised(value, expected):
    assert valid_skill(generation_strategy=value).generation_strategy == expected


@pytest.mark.parametrize(
    "field", ["topic", "subtopic", "name", "learning_objective"]
)
@pytest.mark.parametrize("value", ["", "   "])
def test_empty_text_fields_are_rejected(field, value):
    with pytest.raises(ValidationError, match=field):
        valid_skill(**{field: value})


@pytest.mark.parametrize(
    "field", ["topic", "subtopic", "name", "learning_objective"]
)
def test_text_fields_are_stripped(field):
    skill = valid_skill(**{field: "  Informed search  "})

    assert getattr(skill, field) == "Informed search"


def test_skill_cannot_be_its_own_prerequisite():
    with pytest.raises(ValidationError, match="AI-SRC-08"):
        valid_skill(prerequisite_skill_ids=["AI-SRC-07", "AI-SRC-08"])


def test_self_prerequisite_is_caught_after_normalisation():
    with pytest.raises(ValidationError, match="AI-SRC-08"):
        valid_skill(prerequisite_skill_ids=[" ai-src-08 "])


@pytest.mark.parametrize("prerequisite_id", ["SRC-07", "AI-SRC-7", "not-an-id", ""])
def test_malformed_prerequisite_skill_id_is_rejected(prerequisite_id):
    with pytest.raises(ValidationError, match="prerequisite_skill_ids"):
        valid_skill(prerequisite_skill_ids=[prerequisite_id])


def test_prerequisite_skill_ids_are_normalised():
    skill = valid_skill(prerequisite_skill_ids=[" ai-src-07 ", "AI-SRC-09"])

    assert skill.prerequisite_skill_ids == ["AI-SRC-07", "AI-SRC-09"]


def test_duplicate_prerequisites_within_a_skill_are_rejected():
    with pytest.raises(ValidationError, match="AI-SRC-07"):
        valid_skill(prerequisite_skill_ids=["AI-SRC-07", " ai-src-07 "])


def test_catalogue_accepts_a_resolvable_graph():
    catalogue = SkillCatalogue(
        skills=[
            valid_skill(skill_id="AI-SRC-08"),
            valid_skill(skill_id="AI-SRC-09", prerequisite_skill_ids=["AI-SRC-08"]),
            valid_skill(skill_id="AI-SRC-10", prerequisite_skill_ids=["AI-SRC-09"]),
        ]
    )

    assert len(catalogue.skills) == 3


def test_catalogue_rejects_an_empty_skill_list():
    with pytest.raises(ValidationError, match="skills"):
        SkillCatalogue(skills=[])


def test_catalogue_rejects_duplicate_skill_ids():
    with pytest.raises(ValidationError, match="Duplicate skill ids: AI-SRC-08"):
        SkillCatalogue(
            skills=[valid_skill(), valid_skill(name="Heuristics again")]
        )


def test_catalogue_rejects_dangling_prerequisites():
    with pytest.raises(ValidationError, match="AI-SRC-08 -> AI-SRC-99"):
        SkillCatalogue(
            skills=[valid_skill(prerequisite_skill_ids=["AI-SRC-99"])]
        )


def test_catalogue_rejects_a_prerequisite_cycle():
    with pytest.raises(ValidationError, match="Prerequisite cycle"):
        SkillCatalogue(
            skills=[
                valid_skill(skill_id="AI-SRC-08", prerequisite_skill_ids=["AI-SRC-09"]),
                valid_skill(skill_id="AI-SRC-09", prerequisite_skill_ids=["AI-SRC-10"]),
                valid_skill(skill_id="AI-SRC-10", prerequisite_skill_ids=["AI-SRC-08"]),
            ]
        )


def test_catalogue_accepts_a_diamond_dependency():
    catalogue = SkillCatalogue(
        skills=[
            valid_skill(skill_id="AI-SRC-07"),
            valid_skill(skill_id="AI-SRC-08", prerequisite_skill_ids=["AI-SRC-07"]),
            valid_skill(skill_id="AI-SRC-09", prerequisite_skill_ids=["AI-SRC-07"]),
            valid_skill(
                skill_id="AI-SRC-10",
                prerequisite_skill_ids=["AI-SRC-08", "AI-SRC-09"],
            ),
        ]
    )

    assert len(catalogue.skills) == 4


def test_generated_skills_without_reference_material_are_flagged():
    catalogue = SkillCatalogue(
        skills=[
            valid_skill(
                skill_id="AI-SRC-08",
                reference_material=["A heuristic estimates the remaining cost."],
            ),
            valid_skill(skill_id="AI-SRC-09"),
            valid_skill(skill_id="AI-SRC-10", generation_strategy="hand_authored"),
        ]
    )

    assert find_skills_missing_reference_material(catalogue) == ["AI-SRC-09"]


def test_list_defaults_are_not_shared_between_skills():
    first = valid_skill()
    second = valid_skill(skill_id="AI-SRC-09")

    first.reference_material.append("A heuristic estimates remaining cost.")
    first.prerequisite_skill_ids.append("AI-SRC-07")

    assert second.reference_material == []
    assert second.prerequisite_skill_ids == []
