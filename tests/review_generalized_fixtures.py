"""Generalized (non-AI-FND-04-specific) calibration fixtures for the reviewer's
per-option-assessment methodology (authoring/review/prompt.py's review-v7 "ANSWERING
METHODOLOGY" section).

The AI-FND-04-b4cd5c51a8cab3c4 semantic-overlap case
(tests/review_fnd_fixtures.py, tests/test_review_ai_fnd_04_semantic_overlap_regression.py)
is one real, concrete instance of "multiple options express the same claim in
different words." These four fixtures generalize the failure/success modes across
unrelated domains, so the fix is proven to generalize rather than being tuned to one
candidate:

- GEN_PARAPHRASE: three of four options are paraphrases of the same claim (ambiguous
  -- must be blocked).
- GEN_MATH_EQUIVALENT: two of four options are the same numeric value in different
  notation (ambiguous -- must be blocked).
- GEN_CLOSE_DISTRACTOR: four options that are topically/lexically close (all
  atmospheric gases) but only one is actually correct (non-ambiguous control -- must
  NOT be blocked).
- GEN_NEGATION: options built from a "yes"/"no" negation pattern where only one is
  actually true (non-ambiguous control -- a negation is a structural resemblance, not
  a semantic equivalence, and must NOT be blocked).

Every skill/intent/reference here is a deliberately synthetic calibration fixture
(skill_id prefixed GEN-), not real course content -- mirrors the synthetic-skill
convention already used by scripts/run_offline_calibration_batch.py's
CalibrationBatchCase.
"""

from datetime import datetime, timezone

from authoring.grounded_batch import IntentQuestion, PendingQuestion
from authoring.question_intents import QuestionIntent
from taxonomy.schemas import ReferenceProvenance, SkillDefinition

FIXED_TIME = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)


def _skill(skill_id: str, name: str, learning_objective: str) -> SkillDefinition:
    return SkillDefinition(
        skill_id=skill_id,
        topic="Generalized calibration fixture",
        subtopic="Generalized calibration fixture",
        name=name,
        learning_objective=learning_objective,
        cognitive_process="understand",
        generation_strategy="generated",
    )


def _intent(skill: SkillDefinition, assessment_focus: str) -> QuestionIntent:
    return QuestionIntent(
        intent_id=f"{skill.skill_id}-INT-01",
        skill_id=skill.skill_id,
        learning_objective=skill.learning_objective,
        difficulty="introductory",
        assessment_focus=assessment_focus,
        cognitive_demand="understand",
        question_archetype="generalized calibration fixture",
        preferred_reference_ids=[f"{skill.skill_id}-REF-01"],
        required_question_characteristics=["State one clear scenario or claim.", "Ask for the correct value or claim."],
        prohibited_ambiguity_patterns=["Do not present a scenario that plausibly fits more than one answer equally well."],
        expected_misconception_or_distractor_strategy=["Distractors should be plausible but genuinely incorrect."],
        required_concepts=[assessment_focus],
        prohibited_conflations=["conflating a superficially similar but incorrect claim with the correct one"],
    )


def _candidate(skill_id: str, question_id: str, intent_id: str, reference_id: str, question: IntentQuestion) -> PendingQuestion:
    return PendingQuestion(
        batch_id="generalized-calibration-fixture",
        question_id=question_id,
        skill_id=skill_id,
        question_index=0,
        intent_id=intent_id,
        seed=1,
        reference_ids=[reference_id],
        prompt_version="v3.3",
        prompt_hash="d" * 64,
        model_id="test-model",
        model_revision="test-rev-1",
        generation_parameters={},
        generated_at=FIXED_TIME,
        git_commit="0" * 40,
        raw_response="{}",
        question=question,
    )


# --- GEN-PARAPHRASE: multiple paraphrased correct answers (ambiguous) ---------------

GEN_PARAPHRASE_SKILL = _skill(
    "GEN-PAR-01", "Network security fundamentals",
    "Explain the primary purpose of a firewall in network security.",
)
GEN_PARAPHRASE_INTENT = _intent(
    GEN_PARAPHRASE_SKILL, "Identify what a firewall is primarily used for."
)
GEN_PARAPHRASE_REFERENCE = ReferenceProvenance(
    reference_id="GEN-PAR-01-REF-01",
    skill_id=GEN_PARAPHRASE_SKILL.skill_id,
    reference_material=(
        "A firewall is a network security device that monitors incoming and outgoing "
        "network traffic and blocks or allows traffic based on a defined set of "
        "security rules, in order to establish a barrier between a trusted network "
        "and untrusted networks such as the internet."
    ),
    title="Firewall fundamentals (calibration fixture)",
    source_url="https://example.invalid/firewalls",
    source_domain="example.invalid",
    content_hash="e" * 64,
    retrieved_at=FIXED_TIME,
    reviewer_id="calibration-fixture",
    reviewed_at=FIXED_TIME,
)
GEN_PARAPHRASE_QUESTION = IntentQuestion(
    intent_id=GEN_PARAPHRASE_INTENT.intent_id,
    question="What is the primary purpose of a firewall in network security?",
    options=[
        "To block unauthorized access to or from a private network",
        "To prevent unauthorized network traffic from entering or leaving a protected network",
        "To filter network traffic so unapproved connections cannot reach or leave a private network",
        "To encrypt data transmitted between two networked devices",
    ],
    correct_answer="To block unauthorized access to or from a private network",
    explanation=(
        "A firewall establishes a barrier between a trusted network and untrusted "
        "networks by allowing or blocking traffic according to security rules."
    ),
    concept="firewall purpose",
    difficulty="introductory",
)
GEN_PARAPHRASE_CANDIDATE = _candidate(
    GEN_PARAPHRASE_SKILL.skill_id,
    "GEN-PAR-01-fixture-candidate",
    GEN_PARAPHRASE_INTENT.intent_id,
    GEN_PARAPHRASE_REFERENCE.reference_id,
    GEN_PARAPHRASE_QUESTION,
)


# --- GEN-MATH-EQUIVALENT: two mathematically equivalent answers (ambiguous) --------

GEN_MATH_SKILL = _skill(
    "GEN-MEQ-01", "Decimal and fraction conversion",
    "Convert a fractional quantity to its decimal equivalent.",
)
GEN_MATH_INTENT = _intent(GEN_MATH_SKILL, "Convert 3/4 cup to a decimal quantity.")
GEN_MATH_REFERENCE = ReferenceProvenance(
    reference_id="GEN-MEQ-01-REF-01",
    skill_id=GEN_MATH_SKILL.skill_id,
    reference_material=(
        "To convert a fraction to a decimal, divide the numerator by the "
        "denominator. Three quarters (3/4) is equal to 0.75, since 3 divided by 4 "
        "equals 0.75."
    ),
    title="Fraction-to-decimal conversion (calibration fixture)",
    source_url="https://example.invalid/fractions",
    source_domain="example.invalid",
    content_hash="f" * 64,
    retrieved_at=FIXED_TIME,
    reviewer_id="calibration-fixture",
    reviewed_at=FIXED_TIME,
)
GEN_MATH_QUESTION = IntentQuestion(
    intent_id=GEN_MATH_INTENT.intent_id,
    question="A recipe requires 3/4 cup of sugar. Expressed as a decimal, how much sugar is required?",
    options=[
        "0.75 cups",
        "75/100 cups",
        "0.57 cups",
        "1.75 cups",
    ],
    correct_answer="0.75 cups",
    explanation="3/4 is equal to 0.75, since 3 divided by 4 equals 0.75.",
    concept="fraction-to-decimal conversion",
    difficulty="introductory",
)
GEN_MATH_CANDIDATE = _candidate(
    GEN_MATH_SKILL.skill_id,
    "GEN-MEQ-01-fixture-candidate",
    GEN_MATH_INTENT.intent_id,
    GEN_MATH_REFERENCE.reference_id,
    GEN_MATH_QUESTION,
)


# --- GEN-CLOSE-DISTRACTOR: closely related but genuinely wrong distractors ---------
# (non-ambiguous control -- must NOT be flagged multiple_defensible_answers)

GEN_CLOSE_SKILL = _skill(
    "GEN-CLS-01", "Photosynthesis gas exchange",
    "Identify which atmospheric gas plants absorb during photosynthesis.",
)
GEN_CLOSE_INTENT = _intent(
    GEN_CLOSE_SKILL, "Identify the gas plants primarily absorb during photosynthesis."
)
GEN_CLOSE_REFERENCE = ReferenceProvenance(
    reference_id="GEN-CLS-01-REF-01",
    skill_id=GEN_CLOSE_SKILL.skill_id,
    reference_material=(
        "During photosynthesis, plants absorb carbon dioxide from the atmosphere and "
        "release oxygen as a byproduct, using light energy to convert carbon dioxide "
        "and water into glucose."
    ),
    title="Photosynthesis gas exchange (calibration fixture)",
    source_url="https://example.invalid/photosynthesis",
    source_domain="example.invalid",
    content_hash="1" * 64,
    retrieved_at=FIXED_TIME,
    reviewer_id="calibration-fixture",
    reviewed_at=FIXED_TIME,
)
GEN_CLOSE_QUESTION = IntentQuestion(
    intent_id=GEN_CLOSE_INTENT.intent_id,
    question="Which gas do plants primarily absorb from the atmosphere during photosynthesis?",
    options=[
        "Carbon dioxide",
        "Oxygen",
        "Nitrogen",
        "Carbon monoxide",
    ],
    correct_answer="Carbon dioxide",
    explanation=(
        "Plants absorb carbon dioxide during photosynthesis and release oxygen as a "
        "byproduct; oxygen, nitrogen, and carbon monoxide are not what photosynthesis "
        "consumes from the atmosphere."
    ),
    concept="photosynthesis gas exchange",
    difficulty="introductory",
)
GEN_CLOSE_CANDIDATE = _candidate(
    GEN_CLOSE_SKILL.skill_id,
    "GEN-CLS-01-fixture-candidate",
    GEN_CLOSE_INTENT.intent_id,
    GEN_CLOSE_REFERENCE.reference_id,
    GEN_CLOSE_QUESTION,
)


# --- GEN-NEGATION: negation-based distractors -------------------------------------
# (non-ambiguous control -- a negation pattern is not semantic equivalence)

GEN_NEGATION_SKILL = _skill(
    "GEN-NEG-01", "Natural selection and reproduction rate",
    "Explain how trait fitness affects reproductive success under natural selection.",
)
GEN_NEGATION_INTENT = _intent(
    GEN_NEGATION_SKILL, "State whether all individuals reproduce at an equal rate under natural selection."
)
GEN_NEGATION_REFERENCE = ReferenceProvenance(
    reference_id="GEN-NEG-01-REF-01",
    skill_id=GEN_NEGATION_SKILL.skill_id,
    reference_material=(
        "Natural selection is the process by which individuals with traits better "
        "suited to their environment tend to survive and reproduce more successfully "
        "than individuals with less favorable traits, so reproduction rate is not "
        "equal across all individuals in a population."
    ),
    title="Natural selection fundamentals (calibration fixture)",
    source_url="https://example.invalid/natural-selection",
    source_domain="example.invalid",
    content_hash="2" * 64,
    retrieved_at=FIXED_TIME,
    reviewer_id="calibration-fixture",
    reviewed_at=FIXED_TIME,
)
GEN_NEGATION_QUESTION = IntentQuestion(
    intent_id=GEN_NEGATION_INTENT.intent_id,
    question="According to the theory of evolution by natural selection, do all members of a species reproduce at an equal rate?",
    options=[
        "No, individuals with traits better suited to their environment tend to reproduce more successfully",
        "Yes, all individuals in a species reproduce at exactly the same rate regardless of traits",
        "No, but reproduction rate depends only on random chance rather than traits",
        "Yes, reproduction rate is determined solely by geographic location",
    ],
    correct_answer="No, individuals with traits better suited to their environment tend to reproduce more successfully",
    explanation=(
        "Natural selection means individuals with traits better suited to their "
        "environment tend to survive and reproduce more successfully than others -- "
        "reproduction rate is not equal, not purely random, and not determined solely "
        "by geography."
    ),
    concept="natural selection and reproduction rate",
    difficulty="introductory",
)
GEN_NEGATION_CANDIDATE = _candidate(
    GEN_NEGATION_SKILL.skill_id,
    "GEN-NEG-01-fixture-candidate",
    GEN_NEGATION_INTENT.intent_id,
    GEN_NEGATION_REFERENCE.reference_id,
    GEN_NEGATION_QUESTION,
)
