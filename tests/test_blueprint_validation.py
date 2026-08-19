from datetime import UTC, datetime

import pytest

from authoring.blueprint_validation import validate_replenishment_blueprint
from authoring.question_intents import PilotBlueprint, QuestionIntent, load_blueprint_for_batch
from taxonomy.loader import course_paths, course_provenance_path, load_reference_provenance, load_skills
from taxonomy.schemas import ReferenceProvenance, SkillDefinition


def real_ai_inputs():
    skills = load_skills(*course_paths("ai")).skills
    provenance = load_reference_provenance(course_provenance_path("ai"))
    return skills, provenance


def skill(skill_id="AI-FND-03", learning_objective="Identify appropriate applications of AI in different fields."):
    return SkillDefinition(
        skill_id=skill_id,
        topic="Foundations of Artificial Intelligence",
        subtopic="AI applications",
        name="Real-world AI systems",
        learning_objective=learning_objective,
        cognitive_process="apply",
        generation_strategy="generated",
    )


def reference(reference_id="AI-FND-03-aaaaaaaaaaaa", skill_id="AI-FND-03"):
    return ReferenceProvenance(
        reference_id=reference_id,
        skill_id=skill_id,
        reference_material="Some approved grounding passage.",
        title="Some title",
        source_url="https://cs50.harvard.edu/ai/2024/",
        source_domain="cs50.harvard.edu",
        content_hash="a" * 64,
        retrieved_at=datetime(2026, 8, 17, tzinfo=UTC),
        reviewer_id="albert",
        reviewed_at=datetime(2026, 8, 17, tzinfo=UTC),
    )


def intent(**overrides):
    fields = {
        "intent_id": "AI-FND-03-INT-01",
        "skill_id": "AI-FND-03",
        "assessment_focus": "Recognize which AI subfield a scenario exemplifies.",
        "question_archetype": "AI subfield recognition",
        "preferred_reference_ids": ["AI-FND-03-aaaaaaaaaaaa"],
        "required_concepts": ["AI subfields"],
        "prohibited_conflations": ["conflating ML with AI in general"],
        "difficulty": "introductory",
        "learning_objective": "Identify appropriate applications of AI in different fields.",
        "cognitive_demand": "Recognize the subfield.",
        "required_question_characteristics": ["State a scenario."],
        "prohibited_ambiguity_patterns": ["Do not allow two equally valid subfields."],
        "expected_misconception_or_distractor_strategy": ["Confuse ML with search."],
    }
    fields.update(overrides)
    return QuestionIntent(**fields)


def blueprint(**overrides):
    fields = {
        "batch_id": "test-batch",
        "prompt_version": "v3.3",
        "review_status": "blueprint-approved",
        "reviewer_id": "albert",
        "reviewed_at": datetime(2026, 8, 17, tzinfo=UTC),
        "base_seed": 1,
        "questions_per_intent": 1,
        "intents": [intent()],
    }
    fields.update(overrides)
    return PilotBlueprint(**fields)


def test_the_real_bounded_ai_fnd_blueprint_passes_every_invariant_with_mixed_difficulty():
    """grounded-ai-fnd-release-v1.json: AI-FND-03 and AI-FND-04 each now declare 6
    intents (1 original + 5 added to close the target-supply gap), spanning both
    introductory and intermediate tiers -- the same mixed-difficulty shape as DSA/
    Linear Algebra/Database Systems, and the reason this blueprint must resolve
    "mixed" rather than being rejected."""
    blueprint = load_blueprint_for_batch("grounded-ai-fnd-release-v1")
    skills, provenance = real_ai_inputs()

    summary = validate_replenishment_blueprint(blueprint, skills, provenance)

    assert summary["intent_count"] == 12
    assert summary["skill_ids"] == ["AI-FND-03", "AI-FND-04"]
    assert all(difficulty == "mixed" for difficulty in summary["difficulty_counts"].values())
    assert all(summary["checks"].values())


def test_the_real_dsa_blueprint_passes_every_invariant_with_mixed_difficulty():
    """grounded-dsa-v1.json: every one of its 7 skills declares 2 introductory + 4
    intermediate intents (2 original intermediate + 2 added to close the
    target-supply gap) -- the real shape this repo's capacity-gap packet
    identified, and the reason _blueprint_generation_difficulty/this module's
    difficulty check were changed to accept "mixed" instead of rejecting it."""
    blueprint = load_blueprint_for_batch("grounded-dsa-v1")
    skills = load_skills(*course_paths("dsa")).skills
    provenance = load_reference_provenance(course_provenance_path("dsa"))

    summary = validate_replenishment_blueprint(blueprint, skills, provenance)

    assert summary["intent_count"] == 42
    assert len(summary["skill_ids"]) == 7
    assert all(difficulty == "mixed" for difficulty in summary["difficulty_counts"].values())
    assert all(summary["checks"].values())


def test_approved_blueprint_without_reviewer_fails():
    bp = blueprint(reviewer_id=None, reviewed_at=None)
    with pytest.raises(ValueError, match="reviewer and timestamp"):
        validate_replenishment_blueprint(bp, [skill()], [reference()])


def test_skill_id_not_in_taxonomy_fails():
    bp = blueprint(intents=[intent(skill_id="AI-NOT-REAL", intent_id="AI-NOT-REAL-INT-01")])
    with pytest.raises(ValueError, match="not a defined taxonomy skill"):
        validate_replenishment_blueprint(bp, [skill()], [reference()])


def test_learning_objective_must_match_canonical_wording_exactly():
    bp = blueprint(intents=[intent(learning_objective="A paraphrased objective.")])
    with pytest.raises(ValueError, match="canonical learning objective"):
        validate_replenishment_blueprint(bp, [skill()], [reference()])


def test_unapproved_reference_mapping_fails():
    bp = blueprint(intents=[intent(preferred_reference_ids=["AI-FND-03-doesnotexist"])])
    with pytest.raises(ValueError, match="unapproved reference mappings"):
        validate_replenishment_blueprint(bp, [skill()], [reference()])


def test_reference_belonging_to_a_different_skill_fails():
    bp = blueprint(intents=[intent(preferred_reference_ids=["AI-FND-04-bbbbbbbbbbbb"])])
    other_skill_reference = reference(reference_id="AI-FND-04-bbbbbbbbbbbb", skill_id="AI-FND-04")
    with pytest.raises(ValueError, match="unapproved reference mappings"):
        validate_replenishment_blueprint(bp, [skill()], [other_skill_reference])


def test_mixed_difficulty_within_one_skill_reports_mixed_and_passes():
    """Intentional behavior change (mirrors _blueprint_generation_difficulty in
    worker.py): a skill whose intents span more than one explicit difficulty is no
    longer a validation failure -- generate_batch's own mixed-mode support (see
    grounded_batch.py) resolves each question from its own intent.difficulty. Only a
    missing explicit difficulty remains a real defect (see the test below)."""
    bp = blueprint(
        intents=[
            intent(intent_id="AI-FND-03-INT-01", difficulty="introductory"),
            intent(intent_id="AI-FND-03-INT-02", difficulty="intermediate"),
        ]
    )
    summary = validate_replenishment_blueprint(
        bp, [skill()], [reference(), reference("AI-FND-03-bbbbbbbbbbbb")]
    )
    assert summary["difficulty_counts"]["AI-FND-03"] == "mixed"
    assert all(summary["checks"].values())


def test_missing_declared_difficulty_still_fails():
    bp = blueprint(
        intents=[
            intent(intent_id="AI-FND-03-INT-01", difficulty="introductory"),
            intent(intent_id="AI-FND-03-INT-02", difficulty=None),
        ]
    )
    with pytest.raises(ValueError, match="no explicit difficulty"):
        validate_replenishment_blueprint(bp, [skill()], [reference(), reference("AI-FND-03-bbbbbbbbbbbb")])


def test_missing_narrative_field_fails():
    bp = blueprint(intents=[intent(expected_misconception_or_distractor_strategy=[])])
    with pytest.raises(ValueError, match="misconception or distractor strategy"):
        validate_replenishment_blueprint(bp, [skill()], [reference()])
