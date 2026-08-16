from authoring.course_catalog.registry import resolve_course
from authoring.replenishment.manifest import CourseManifest


def _manifest(course_id, status, aliases, title=None):
    return CourseManifest(
        course_id=course_id,
        title=title or course_id.upper(),
        version="1",
        taxonomy_path=f"taxonomy/data/{course_id}",
        approved_bank_path=f"outputs/{course_id}-bank.jsonl",
        bkt_model_path=f"outputs/{course_id}-model.pkl",
        candidate_store_path=f"outputs/{course_id}-candidates.json",
        review_store_path=f"outputs/{course_id}-reviews",
        allowed_domains=("example.edu",),
        low_supply_threshold=3,
        target_supply=6,
        default_bkt_model_version="v1",
        status=status,
        aliases=aliases,
    )


MANIFESTS = [
    _manifest("intro-ai", "active", ("AI", "Introduction to AI")),
    _manifest("dsa", "approved_for_preparation", ("DSA", "Data Structures")),
    _manifest("linear-algebra", "proposed", ("LA", "Linear Algebra")),
]


def test_resolves_by_exact_course_id_case_insensitively():
    assert resolve_course("INTRO-AI", MANIFESTS).course_id == "intro-ai"


def test_resolves_by_title():
    assert resolve_course("introduction to ai", MANIFESTS).course_id == "intro-ai"


def test_resolves_by_alias_case_insensitively_with_whitespace():
    assert resolve_course("  dsa  ", MANIFESTS).course_id == "dsa"


def test_resolves_a_not_yet_active_course_so_the_being_prepared_message_can_show():
    resolved = resolve_course("Linear Algebra", MANIFESTS)
    assert resolved is not None
    assert resolved.status == "proposed"


def test_unrecognized_query_returns_none():
    assert resolve_course("Astrophysics", MANIFESTS) is None


def test_empty_query_returns_none():
    assert resolve_course("   ", MANIFESTS) is None
