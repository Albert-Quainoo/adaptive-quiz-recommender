from taxonomy.loader import course_paths, load_skills
from taxonomy.schemas import find_skills_missing_reference_material

AI_COURSE_PATHS = course_paths("ai")

# Skills whose reference material has not been transcribed yet. Every entry here
# is a generated skill that cannot be used for generation until it is grounded,
# so this list should shrink to nothing as references.csv is filled in.
AWAITING_REFERENCE_MATERIAL = [
    "AI-SRC-01",
    "AI-SRC-02",
    "AI-SRC-03",
    "AI-SRC-07",
    "AI-SRC-08",
    "AI-SRC-11",
]


def test_search_taxonomy_loads():
    catalogue = load_skills(*AI_COURSE_PATHS)

    assert len(catalogue.skills) == 11
    assert {skill.skill_id for skill in catalogue.skills} == {
        f"AI-SRC-{number:02d}"
        for number in range(1, 12)
    }


def test_search_taxonomy_topic_is_consistent():
    catalogue = load_skills(*AI_COURSE_PATHS)

    assert {skill.topic for skill in catalogue.skills} == {"Search and Problem Solving"}


def test_trace_skills_are_templated():
    catalogue = load_skills(*AI_COURSE_PATHS)

    tracing = [
        skill for skill in catalogue.skills
        if skill.learning_objective.startswith("Trace")
    ]

    assert len(tracing) == 5
    assert all(skill.generation_strategy == "templated" for skill in tracing)


def test_generation_readiness_gap_is_known():
    catalogue = load_skills(*AI_COURSE_PATHS)

    assert find_skills_missing_reference_material(catalogue) == (
        AWAITING_REFERENCE_MATERIAL
    )
