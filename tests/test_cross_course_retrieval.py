"""Cross-course retrieval-context regression coverage.

score_relevance used to gate every skill's passage on AI_CONTEXT_ANCHOR/
CONTEXT_TERMS - vocabulary hand-curated for the AI course alone. A non-AI
skill's passage could never contain that vocabulary, so
`bool(self.passage_context)` in RelevanceScore.is_relevant() failed for every
one of them regardless of how good the page actually was.

The fix (authoring/retrieval/relevance.py's course_context_vocabulary) keeps
CONTEXT_TERMS for the AI course exactly as before, and derives an equivalent
vocabulary for every other course from its own taxonomy (course title plus
every skill's topic and subtopic - fields already read for concept/objective
scoring, just pooled across the whole course here).

These tests prove, for one representative skill per course: a genuinely
on-topic passage is accepted, and a genuinely off-topic passage is still
rejected. They also prove the AI course's own behaviour is untouched.
"""

from taxonomy.loader import course_paths, load_skills
from taxonomy.schemas import SkillDefinition

from authoring.retrieval.relevance import (
    CONTEXT_TERMS,
    MAX_OBJECTIVE_TERMS_REQUIRED,
    OBJECTIVE_COVERAGE_DENOMINATOR,
    OBJECTIVE_COVERAGE_NUMERATOR,
    course_context_vocabulary,
    objective_terms,
    score_relevance,
)
from authoring.retrieval.search import build_search_queries

IRRELEVANT_PASSAGE = (
    "Photosynthesis converts sunlight, water, and carbon dioxide into glucose "
    "and oxygen inside the chloroplasts of a plant cell, driven by the "
    "light-dependent and light-independent reactions."
)


def load_skill(course: str, skill_id: str) -> SkillDefinition:
    catalogue = load_skills(*course_paths(course))
    return next(skill for skill in catalogue.skills if skill.skill_id == skill_id)


def relevance_for(course: str, skill_id: str, passage: str, course_title: str):
    catalogue = load_skills(*course_paths(course))
    skill = next(item for item in catalogue.skills if item.skill_id == skill_id)
    vocabulary = course_context_vocabulary(course_title, catalogue.skills)

    return score_relevance(
        skill, "https://example.edu/x", "", "", passage, context_vocabulary=vocabulary
    )


def test_a_globally_ai_specific_vocabulary_used_to_block_every_other_course():
    """The defect this fix closes, pinned down: CONTEXT_TERMS alone (the old
    behaviour, before it was scoped to the AI course) rejects a genuinely
    on-topic passage for every other course, regardless of quality."""
    dsa_skill = load_skill("dsa", "DSA-HSH-01")
    scored = score_relevance(
        dsa_skill,
        "https://example.edu/x",
        "",
        "",
        "A hash function maps keys to positions in a hash table. Collisions "
        "are detected and resolved using open addressing or chaining.",
        context_vocabulary=CONTEXT_TERMS,
    )

    assert scored.passage_context == ()
    assert not scored.is_relevant()


def test_dsa_accepts_a_representative_on_topic_hashing_passage():
    scored = relevance_for(
        "dsa",
        "DSA-HSH-01",
        (
            "A hash function maps keys to positions in a hash table. When two "
            "keys map to the same position, a collision has occurred; "
            "collisions are detected during insertion and resolved using open "
            "addressing or chaining, so every key still occupies a table "
            "position."
        ),
        "Data Structures & Algorithms",
    )

    assert scored.passage_coverage_passed
    assert scored.passage_context
    assert scored.is_relevant()


def test_dsa_still_rejects_clearly_irrelevant_material():
    scored = relevance_for(
        "dsa", "DSA-HSH-01", IRRELEVANT_PASSAGE, "Data Structures & Algorithms"
    )

    assert not scored.is_relevant()


def test_linear_algebra_accepts_a_representative_on_topic_span_passage():
    scored = relevance_for(
        "linear-algebra",
        "LA-VSP-01",
        (
            "To determine whether a set of vectors spans a space, check "
            "whether every vector in that space is a combination of them. "
            "The set is linearly independent when no vector equals a "
            "combination of the others, and together the two properties mean "
            "it forms a basis. To test whether a subset is a subspace, check "
            "whether it is closed under addition and scalar multiplication."
        ),
        "Linear Algebra",
    )

    assert scored.passage_coverage_passed
    assert scored.passage_context
    assert scored.is_relevant()


def test_linear_algebra_still_rejects_clearly_irrelevant_material():
    scored = relevance_for(
        "linear-algebra", "LA-VSP-01", IRRELEVANT_PASSAGE, "Linear Algebra"
    )

    assert not scored.is_relevant()


def test_database_systems_accepts_a_representative_on_topic_normalization_passage():
    scored = relevance_for(
        "database-systems",
        "DB-NRM-01",
        (
            "To normalize a schema in a relational database through third "
            "normal form, first identify the functional dependencies present "
            "in the relation, since removing partial and transitive "
            "functional dependencies is what pushes a relation from first "
            "normal form toward third normal form."
        ),
        "Database Systems",
    )

    assert scored.passage_coverage_passed
    assert scored.passage_context
    assert scored.is_relevant()


def test_database_systems_still_rejects_clearly_irrelevant_material():
    scored = relevance_for(
        "database-systems", "DB-NRM-01", IRRELEVANT_PASSAGE, "Database Systems"
    )

    assert not scored.is_relevant()


def test_intro_to_ai_still_accepts_its_own_representative_heuristic_passage():
    """Unchanged behaviour: the AI course keeps CONTEXT_TERMS, not a course-
    derived vocabulary, so this is exactly the pre-fix path."""
    scored = relevance_for(
        "ai",
        "AI-SRC-08",
        (
            "A heuristic estimates the remaining cost from a state to the "
            "goal, allowing informed search to prioritize its frontier."
        ),
        "Introduction to Artificial Intelligence",
    )

    assert scored.is_relevant()
    assert course_context_vocabulary(
        "Introduction to Artificial Intelligence", [load_skill("ai", "AI-SRC-08")]
    ) is CONTEXT_TERMS


def test_intro_to_ai_still_rejects_clearly_irrelevant_material():
    scored = relevance_for("ai", "AI-FND-01", IRRELEVANT_PASSAGE, "Introduction to Artificial Intelligence")

    assert not scored.is_relevant()


def test_search_queries_carry_each_courses_own_anchor_not_ai():
    """build_search_queries used to append AI_CONTEXT_ANCHOR to every query
    regardless of course. Every non-AI query must no longer carry it, and
    must instead carry the course's own title."""
    dsa_skill = load_skill("dsa", "DSA-HSH-01")
    queries = build_search_queries(dsa_skill, course_anchor="Data Structures & Algorithms")

    assert queries
    assert all("artificial intelligence" not in query for query in queries)
    assert all("data structures & algorithms" in query for query in queries)


def test_ai_search_queries_still_carry_the_ai_anchor_by_default():
    ai_skill = load_skill("ai", "AI-SRC-08")
    queries = build_search_queries(ai_skill)

    assert queries
    assert all("introduction to artificial intelligence" in query for query in queries)


# --- objective-coverage cap -------------------------------------------------
#
# The generic passage_covers_skill fallback requires a passage to literally
# contain roughly two thirds of a skill's own learning-objective vocabulary.
# That ratio was tuned against AI's own (short) objectives; DSA/LA/DB write
# longer, multi-clause objectives, so the same ratio can demand as many as 10
# exact word matches (LA-SLE-01) - a bar real, on-topic prose does not clear.
# MAX_OBJECTIVE_TERMS_REQUIRED bounds the requirement without changing the
# ratio for any objective already under the cap.


def raw_required(skill_id: str, course: str) -> int:
    """The uncapped two-thirds requirement, computed the same way
    score_relevance does internally, so tests can assert against a skill's
    real taxonomy wording rather than a hand-picked count."""
    skill = load_skill(course, skill_id)
    vocabulary = objective_terms(skill)
    return max(
        1,
        (
            OBJECTIVE_COVERAGE_NUMERATOR * len(vocabulary)
            + OBJECTIVE_COVERAGE_DENOMINATOR
            - 1
        )
        // OBJECTIVE_COVERAGE_DENOMINATOR,
    )


def test_short_ai_objectives_are_unaffected_by_the_cap():
    """AI-FND-01's objective is short enough that its raw requirement already
    sits under the cap - the cap must be a no-op here, not a new floor."""
    assert raw_required("AI-FND-01", "ai") < MAX_OBJECTIVE_TERMS_REQUIRED

    scored = score_relevance(
        load_skill("ai", "AI-FND-01"),
        "https://example.edu/x",
        "",
        "",
        "",
        context_vocabulary=course_context_vocabulary("x", []),
    )

    assert scored.objective_terms_required == raw_required("AI-FND-01", "ai")


def test_a_long_objective_is_capped_rather_than_scaling_without_bound():
    """LA-SLE-01's 15-word objective would otherwise demand 10 literal
    matches; the cap must hold it at MAX_OBJECTIVE_TERMS_REQUIRED, not let it
    grow with objective length."""
    assert raw_required("LA-SLE-01", "linear-algebra") > MAX_OBJECTIVE_TERMS_REQUIRED

    scored = score_relevance(
        load_skill("linear-algebra", "LA-SLE-01"),
        "https://example.edu/x",
        "",
        "",
        "",
        context_vocabulary=course_context_vocabulary("x", []),
    )

    assert scored.objective_terms_required == MAX_OBJECTIVE_TERMS_REQUIRED


def test_dsa_hsh_01_previously_failing_live_passage_now_passes():
    """The real page a live retrieval run found for DSA-HSH-01
    (opendsa-server.cs.vt.edu, "10.1. Introduction to Hashing") matched 5 of
    its 9 objective terms - one short of the uncapped 6-term requirement, so
    it was rejected before this fix. With the cap it is accepted."""
    scored = relevance_for(
        "dsa",
        "DSA-HSH-01",
        (
            "It is impractical in this situation to use a hash table with "
            "65,536 slots, because then the vast majority of the slots would "
            "be left empty. Instead, we must devise a hash function that "
            "allows us to store and retrieve any given key/value pair in "
            "constant time. Say we wish to store a collection of records "
            "with keys such as employee ID numbers. If a collision occurs "
            "because a key already maps to that table position, we need a "
            "way to resolve it."
        ),
        "Data Structures & Algorithms",
    )

    assert scored.objective_terms_required == MAX_OBJECTIVE_TERMS_REQUIRED
    assert len(scored.objective) >= MAX_OBJECTIVE_TERMS_REQUIRED
    assert scored.passage_coverage_passed
    assert scored.is_relevant()


def test_ai_src_01_still_uses_its_own_component_gate_not_the_capped_ratio():
    """AI-SRC-01's raw requirement (8) sits well above the cap, but it never
    reaches the generic fallback at all - passage_covers_skill special-cases
    it to a fixed 5-component check. Proves the cap did not touch that path."""
    from authoring.retrieval.pilot import scopes_for

    assert raw_required("AI-SRC-01", "ai") > MAX_OBJECTIVE_TERMS_REQUIRED

    scored = score_relevance(
        load_skill("ai", "AI-SRC-01"),
        "https://inst.eecs.berkeley.edu/~cs188/textbook/search/state.html",
        "Search",
        "",
        (
            "An artificial intelligence search problem begins at an initial "
            "state. Its actions use a transition model to produce new states."
        ),
        scopes=scopes_for("AI-SRC-01"),
    )

    assert scored.is_relevant()
