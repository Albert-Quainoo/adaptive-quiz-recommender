from collections import Counter

from taxonomy.loader import (
    course_paths,
    course_provenance_path,
    load_reference_provenance,
    load_skills,
)
from taxonomy.schemas import (
    find_prerequisite_cycle,
    find_skills_missing_reference_material,
)
from templates.registry import (
    implemented_template_ids,
    unimplemented_template_ids,
)

AI_COURSE_PATHS = course_paths("ai")

CANONICAL_SKILL_IDS = (
    tuple(f"AI-FND-{number:02d}" for number in range(1, 5))
    + tuple(f"AI-AGT-{number:02d}" for number in range(1, 6))
    + tuple(f"AI-SRC-{number:02d}" for number in range(1, 12))
    + tuple(f"AI-ML-{number:02d}" for number in range(1, 10))
    + tuple(f"AI-NN-{number:02d}" for number in range(1, 10))
    + tuple(f"AI-ETH-{number:02d}" for number in range(1, 8))
)

SKILLS_PER_TOPIC = {
    "Foundations of Artificial Intelligence": 4,
    "Intelligent Agents": 5,
    "Search and Problem Solving": 11,
    "Machine Learning": 9,
    "Neural Networks and Deep Learning": 9,
    "Responsible and Ethical AI": 7,
}

# Generated skills whose grounding text has not been transcribed from the course
# material yet. Templated skills are absent because their questions come from
# code, and hand-authored ones because a person writes them. This list should
# shrink to nothing as references.csv is filled in.
AWAITING_REFERENCE_MATERIAL = [
    "AI-FND-01", "AI-FND-02", "AI-FND-03", "AI-FND-04",
    "AI-AGT-01", "AI-AGT-02", "AI-AGT-03", "AI-AGT-04", "AI-AGT-05",
    "AI-SRC-03", "AI-SRC-07", "AI-SRC-11",
    "AI-ML-01", "AI-ML-02", "AI-ML-03", "AI-ML-04", "AI-ML-05",
    "AI-ML-06", "AI-ML-07", "AI-ML-08", "AI-ML-09",
    "AI-NN-01", "AI-NN-02", "AI-NN-03", "AI-NN-04",
    "AI-NN-06", "AI-NN-07", "AI-NN-08", "AI-NN-09",
    "AI-ETH-02", "AI-ETH-03", "AI-ETH-04", "AI-ETH-05", "AI-ETH-06",
]

IMPLEMENTED_TEMPLATES = [
    "nn.forward_trace",
    "search.astar_trace",
    "search.bfs_trace",
    "search.dfs_trace",
    "search.greedy_trace",
    "search.ucs_trace",
]

DECLARED_BUT_UNIMPLEMENTED_TEMPLATES = []

PILOT_REFERENCE_IDS = {
    "AI-SRC-01-4024dce75930",
    "AI-SRC-01-8ef4e1416152",
    "AI-SRC-01-9ba6548d4450",
    "AI-SRC-02-a506d362b314",
    "AI-SRC-02-aa97b7fb3bd9",
    "AI-SRC-08-a366da363e17",
    "AI-SRC-08-cbd77b22bcb9",
}


def catalogue():
    return load_skills(*AI_COURSE_PATHS)


# --- taxonomy-valid ---------------------------------------------------------

def test_every_canonical_skill_loads():
    skills = catalogue().skills

    assert len(skills) == 45
    assert tuple(skill.skill_id for skill in skills) == CANONICAL_SKILL_IDS


def test_skill_ids_are_unique():
    skill_ids = [skill.skill_id for skill in catalogue().skills]

    assert len(set(skill_ids)) == len(skill_ids)


def test_every_prerequisite_exists():
    skills = catalogue().skills
    known = {skill.skill_id for skill in skills}

    for skill in skills:
        assert set(skill.prerequisite_skill_ids) <= known


def test_the_prerequisite_graph_is_acyclic():
    prerequisites = {
        skill.skill_id: skill.prerequisite_skill_ids for skill in catalogue().skills
    }

    assert find_prerequisite_cycle(prerequisites) is None


def test_each_topic_holds_the_expected_number_of_skills():
    counts = Counter(skill.topic for skill in catalogue().skills)

    assert dict(counts) == SKILLS_PER_TOPIC


def test_stored_taxonomy_uses_the_a_star_spelling_from_the_prompt():
    skills = catalogue().skills

    assert not any("A*" in skill.name for skill in skills)
    assert not any("A*" in skill.learning_objective for skill in skills)


# --- generation-ready -------------------------------------------------------

def test_the_missing_reference_report_is_complete_and_explicit():
    assert find_skills_missing_reference_material(catalogue()) == (
        AWAITING_REFERENCE_MATERIAL
    )


def test_only_generated_skills_can_be_missing_references():
    strategies = {
        skill.skill_id: skill.generation_strategy for skill in catalogue().skills
    }

    assert all(
        strategies[skill_id] == "generated"
        for skill_id in AWAITING_REFERENCE_MATERIAL
    )


def test_the_approved_pilot_skills_are_generation_ready():
    by_id = {skill.skill_id: skill for skill in catalogue().skills}

    assert {
        skill_id: len(by_id[skill_id].reference_material)
        for skill_id in ("AI-SRC-01", "AI-SRC-02", "AI-SRC-08")
    } == {
        "AI-SRC-01": 3,
        "AI-SRC-02": 2,
        "AI-SRC-08": 2,
    }


def test_canonical_passages_and_provenance_stay_in_sync():
    loaded = catalogue()
    records = load_reference_provenance(
        course_provenance_path("ai"),
        {skill.skill_id for skill in loaded.skills},
    )
    passages = {
        (skill.skill_id, passage)
        for skill in loaded.skills
        for passage in skill.reference_material
    }

    assert {record.reference_id for record in records} == PILOT_REFERENCE_IDS
    assert {(record.skill_id, record.reference_material) for record in records} == (
        passages
    )


# --- template-implemented ---------------------------------------------------

def test_every_templated_skill_names_a_template():
    templated = [
        skill for skill in catalogue().skills
        if skill.generation_strategy == "templated"
    ]

    assert len(templated) == 6
    assert all(skill.template_id for skill in templated)


def test_untemplated_skills_never_name_a_template():
    assert all(
        skill.template_id is None
        for skill in catalogue().skills
        if skill.generation_strategy != "templated"
    )


def test_implemented_and_unimplemented_templates_are_reported_separately():
    loaded = catalogue()

    assert implemented_template_ids(loaded) == IMPLEMENTED_TEMPLATES
    assert unimplemented_template_ids(loaded) == DECLARED_BUT_UNIMPLEMENTED_TEMPLATES
    assert not set(IMPLEMENTED_TEMPLATES) & set(DECLARED_BUT_UNIMPLEMENTED_TEMPLATES)


def test_trace_skills_are_templated():
    tracing = [
        skill for skill in catalogue().skills
        if skill.learning_objective.startswith("Trace")
    ]

    assert len(tracing) == 6
    assert all(skill.generation_strategy == "templated" for skill in tracing)
