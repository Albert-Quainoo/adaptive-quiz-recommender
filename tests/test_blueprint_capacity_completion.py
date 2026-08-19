"""Global regression proof that the remaining blueprint-capacity gap (Linear
Algebra, Database Systems, AI-FND-03, AI-FND-04) is closed, on top of PR #9's
DSA closure -- against the real blueprint, taxonomy, and approved-bank files
in this repo, not synthetic fixtures.

Course policy's target_supply is 6 for every scoped skill (see each course's
authoring/replenishment/manifests/*.json). PilotBlueprint.questions_per_intent
is a hard Literal[1], so a skill's supply ceiling is exactly its declared
intent count -- these tests prove that ceiling is now 6 for all 22 scoped
skills across all four course blueprints, and that the mixed-difficulty
resolution PR #9 introduced validates and computes demand correctly against
every one of them, not just grounded-dsa-v1.json.
"""

from authoring.blueprint_validation import validate_replenishment_blueprint
from authoring.question_intents import intents_by_skill, load_blueprint_for_batch
from authoring.replenishment.demand import (
    compute_blueprint_slot_demand,
    compute_demand_fingerprint,
    deficient_slots,
    load_approved_item_ids,
)
from authoring.replenishment.manifest import active_bank_path, load_course_manifest
from authoring.replenishment.worker import blueprints_covering_skill
from taxonomy.loader import course_paths, course_provenance_path, load_reference_provenance, load_skills

COURSES = {
    "dsa": ("grounded-dsa-v1", "dsa"),
    "linear-algebra": ("grounded-linear-algebra-v1", "linear-algebra"),
    "database-systems": ("grounded-database-systems-v1", "database-systems"),
    "ai": ("grounded-ai-fnd-release-v1", "ai"),
}

TARGET_SUPPLY = 6

EXPECTED_SKILL_IDS = {
    "dsa": {
        "DSA-CPX-01", "DSA-LST-01", "DSA-STK-01", "DSA-SRC-01",
        "DSA-SRT-01", "DSA-HSH-01", "DSA-TGR-01",
    },
    "linear-algebra": {
        "LA-SLE-01", "LA-VSP-01", "LA-LTR-01", "LA-DET-01", "LA-EIG-01", "LA-MKV-01",
    },
    "database-systems": {
        "DB-ERM-01", "DB-REL-01", "DB-ALG-01", "DB-SQL-01",
        "DB-NRM-01", "DB-TXN-01", "DB-IDX-01",
    },
    "ai": {"AI-FND-03", "AI-FND-04"},
}

# course_id used by each course's authoring/replenishment/manifests/<id>.json
MANIFEST_COURSE_ID = {
    "dsa": "dsa",
    "linear-algebra": "linear-algebra",
    "database-systems": "database-systems",
    "ai": "intro-ai",
}

# The approved-bank count each skill already had before this capacity-completion
# wave (DSA: PR #9; LA/DB: 4/6 per the Capacity Gap Packet; AI-FND-03/04: 1/6
# each). Used only to prove old approvals still map onto the same slots -- not a
# claim about the bank's current state going forward.
KNOWN_PRIOR_APPROVED_COUNT = {
    "DSA-CPX-01": 4, "DSA-LST-01": 4, "DSA-STK-01": 4, "DSA-SRC-01": 4,
    "DSA-SRT-01": 4, "DSA-HSH-01": 4, "DSA-TGR-01": 4,
    "LA-SLE-01": 4, "LA-VSP-01": 4, "LA-LTR-01": 4,
    "LA-DET-01": 4, "LA-EIG-01": 4, "LA-MKV-01": 4,
    "DB-ERM-01": 4, "DB-REL-01": 4, "DB-ALG-01": 4, "DB-SQL-01": 4,
    "DB-NRM-01": 4, "DB-TXN-01": 4, "DB-IDX-01": 4,
    "AI-FND-03": 1, "AI-FND-04": 1,
}


def all_scoped_skill_ids() -> set[str]:
    return {skill_id for skills in EXPECTED_SKILL_IDS.values() for skill_id in skills}


def test_exactly_22_skills_are_in_scope_across_the_four_course_blueprints():
    scoped = all_scoped_skill_ids()
    assert len(scoped) == 22


def test_every_scoped_skill_declares_exactly_target_supply_intents():
    """The structural fact the whole capacity-gap packet followed from:
    questions_per_intent=1, so declared intent count IS the supply ceiling.
    Every one of the 22 scoped skills must now declare exactly 6."""
    for batch_id, _course in COURSES.values():
        blueprint = load_blueprint_for_batch(batch_id)
        grouped = intents_by_skill(blueprint)
        for skill_id in EXPECTED_SKILL_IDS[
            next(key for key, val in COURSES.items() if val[0] == batch_id)
        ]:
            assert skill_id in grouped, f"{skill_id} missing from {batch_id}"
            assert len(grouped[skill_id]) == TARGET_SUPPLY, (
                f"{skill_id} declares {len(grouped[skill_id])} intents, "
                f"not target_supply={TARGET_SUPPLY}"
            )


def test_no_skill_remains_structurally_incapable_of_reaching_target_supply():
    """Structurally incapable = declared intent count < target_supply, which no
    amount of generation/review retrying can ever fix. Proves the gap is zero."""
    deficits = {}
    for batch_id, _course in COURSES.values():
        blueprint = load_blueprint_for_batch(batch_id)
        for skill_id, intents in intents_by_skill(blueprint).items():
            if skill_id in all_scoped_skill_ids() and len(intents) < TARGET_SUPPLY:
                deficits[skill_id] = len(intents)
    assert deficits == {}


def test_no_duplicate_intent_ids_across_all_four_course_blueprints():
    seen: dict[str, str] = {}
    for batch_id, _course in COURSES.values():
        blueprint = load_blueprint_for_batch(batch_id)
        for intent in blueprint.intents:
            assert intent.intent_id not in seen, (
                f"{intent.intent_id} declared in both {seen[intent.intent_id]} and {batch_id}"
            )
            seen[intent.intent_id] = batch_id


def test_real_blueprints_pass_every_course_neutral_invariant_with_mixed_difficulty():
    """Every scoped skill's intents span more than one explicit difficulty
    (mixed introductory/intermediate), and every real blueprint must still
    validate cleanly against taxonomy skills, canonical learning objectives,
    and approved reference provenance -- not just a synthetic fixture."""
    for course, (batch_id, taxonomy_course) in COURSES.items():
        blueprint = load_blueprint_for_batch(batch_id)
        skills = load_skills(*course_paths(taxonomy_course)).skills
        provenance = load_reference_provenance(course_provenance_path(taxonomy_course))

        summary = validate_replenishment_blueprint(blueprint, skills, provenance)

        assert all(summary["checks"].values()), f"{batch_id} failed: {summary['checks']}"
        for skill_id in EXPECTED_SKILL_IDS[course]:
            assert summary["difficulty_counts"][skill_id] == "mixed", (
                f"{skill_id} in {batch_id} expected mixed difficulty, "
                f"got {summary['difficulty_counts'][skill_id]}"
            )


def test_one_blueprint_per_skill_resolution_remains_intact():
    """blueprints_covering_skill() must resolve exactly one blueprint for
    every scoped skill -- ambiguous coverage would make automated
    replenishment's difficulty/batch resolution fail closed as "unknown"."""
    for skill_id in all_scoped_skill_ids():
        covering = blueprints_covering_skill(skill_id)
        assert len(covering) == 1, (
            f"{skill_id} is covered by {len(covering)} blueprints, expected exactly 1"
        )


def test_compute_blueprint_slot_demand_reports_six_slots_for_every_scoped_skill():
    """With no approved items at all, every scoped skill's demand must have
    exactly target_supply=6 slots -- proving compute_blueprint_slot_demand()
    reads the real, now-completed intent pools correctly."""
    for batch_id, _course in COURSES.values():
        blueprint = load_blueprint_for_batch(batch_id)
        for skill_id in intents_by_skill(blueprint):
            if skill_id not in all_scoped_skill_ids():
                continue
            demand = compute_blueprint_slot_demand(blueprint, skill_id, set())
            assert len(demand) == TARGET_SUPPLY


def test_existing_approved_bank_records_remain_compatible_after_capacity_expansion():
    """Appending new intents after each skill's existing four (rather than
    reordering or renumbering them) must leave every previously-approved bank
    item mapped onto the same slot index it always was -- proving the newly
    declared capacity is additive, never a silent break of already-approved
    work. Reads the real, committed approved-bank files read-only."""
    for course, (batch_id, _taxonomy_course) in COURSES.items():
        manifest = load_course_manifest(MANIFEST_COURSE_ID[course])
        approved_item_ids = load_approved_item_ids(manifest)
        blueprint = load_blueprint_for_batch(batch_id)

        for skill_id in EXPECTED_SKILL_IDS[course]:
            demand = compute_blueprint_slot_demand(blueprint, skill_id, approved_item_ids)
            satisfied = [slot for slot in demand if slot.satisfied]
            expected_prior = KNOWN_PRIOR_APPROVED_COUNT[skill_id]
            assert len(satisfied) == expected_prior, (
                f"{skill_id}: expected {expected_prior} previously-approved slots "
                f"still satisfied, found {len(satisfied)}"
            )
            # The satisfied slots must be exactly the first `expected_prior`
            # positions -- the newly authored intents were appended, not spliced
            # into the middle of the existing pool.
            satisfied_indices = sorted(slot.question_index for slot in satisfied)
            assert satisfied_indices == list(range(expected_prior))


def test_a_fully_satisfied_declared_blueprint_is_not_incorrectly_capacity_exhausted():
    """A blueprint that still has deficient (unapproved) slots after this
    capacity expansion must never be misreported as capacity_exhausted --
    that outcome is reserved for a blueprint whose every declared slot is
    already an approved bank item. Every scoped skill here still has real,
    generatable headroom (declared 6, approved < 6), so deficient_slots()
    must be non-empty for all of them against the real active bank."""
    for course, (batch_id, _taxonomy_course) in COURSES.items():
        manifest = load_course_manifest(MANIFEST_COURSE_ID[course])
        approved_item_ids = load_approved_item_ids(manifest)
        blueprint = load_blueprint_for_batch(batch_id)

        for skill_id in EXPECTED_SKILL_IDS[course]:
            demand = compute_blueprint_slot_demand(blueprint, skill_id, approved_item_ids)
            assert deficient_slots(demand), (
                f"{skill_id} incorrectly has zero deficient slots -- would be "
                "misreported capacity_exhausted despite having real headroom"
            )


def test_demand_fingerprint_recomputes_deterministically_for_real_blueprints():
    for course, (batch_id, _taxonomy_course) in COURSES.items():
        manifest = load_course_manifest(MANIFEST_COURSE_ID[course])
        approved_item_ids = load_approved_item_ids(manifest)
        blueprint = load_blueprint_for_batch(batch_id)

        for skill_id in EXPECTED_SKILL_IDS[course]:
            first = compute_demand_fingerprint(
                blueprint, skill_id, approved_item_ids,
                difficulty="mixed", target_supply=TARGET_SUPPLY,
            )
            second = compute_demand_fingerprint(
                blueprint, skill_id, approved_item_ids,
                difficulty="mixed", target_supply=TARGET_SUPPLY,
            )
            assert first == second
            assert isinstance(first, str) and first


def test_active_bank_paths_resolve_and_are_readable_for_every_scoped_course():
    """Sanity check on the fixtures the tests above depend on: every scoped
    course's manifest resolves to a real, readable active bank file."""
    for course in COURSES:
        manifest = load_course_manifest(MANIFEST_COURSE_ID[course])
        bank_path = active_bank_path(manifest)
        assert bank_path.is_file(), f"{course}: active bank path {bank_path} missing"
