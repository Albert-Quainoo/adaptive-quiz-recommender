"""Two additional domain-neutral calibration fixtures for the reviewer-model benchmark
(scripts/run_reviewer_model_benchmark.py), extending tests/review_generalized_fixtures.py's
four cases and tests/review_fnd_fixtures.py's three real intro-ai cases to the full set
the reviewer-model-evaluation benchmark plan calls for.

Reuses review_generalized_fixtures.py's _skill/_intent/_candidate helpers rather than
duplicating them.

- GEN_UNIT_EQUIVALENT: two options are the same physical quantity in different units
  (ambiguous -- must be blocked). A different equivalence pattern than GEN_MATH_
  EQUIVALENT's decimal/fraction case, so the benchmark doesn't only exercise one
  notion of "equivalent."
- GEN_CAUSAL_DISTRACTOR: four options naming different, genuinely distinct causes of
  the same observed effect, only one of which the reference material actually
  supports (non-ambiguous control -- must NOT be blocked).
"""

from datetime import datetime, timezone

from taxonomy.schemas import ReferenceProvenance

from tests.review_generalized_fixtures import IntentQuestion, _candidate, _intent, _skill

FIXED_TIME = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)


# --- GEN-UNIT-EQUIVALENT: same quantity in different units (ambiguous) -------------

GEN_UNIT_SKILL = _skill(
    "GEN-UNT-01", "Metric distance conversion",
    "Convert a distance measured in meters to kilometers.",
)
GEN_UNIT_INTENT = _intent(GEN_UNIT_SKILL, "Convert 1500 meters to kilometers.")
GEN_UNIT_REFERENCE = ReferenceProvenance(
    reference_id="GEN-UNT-01-REF-01",
    skill_id=GEN_UNIT_SKILL.skill_id,
    reference_material=(
        "To convert meters to kilometers, divide the number of meters by 1000. "
        "1500 meters is equal to 1.5 kilometers, which is the same distance as "
        "1500000 millimeters."
    ),
    title="Metric distance conversion (calibration fixture)",
    source_url="https://example.invalid/metric-distance",
    source_domain="example.invalid",
    content_hash="3" * 64,
    retrieved_at=FIXED_TIME,
    reviewer_id="calibration-fixture",
    reviewed_at=FIXED_TIME,
)
GEN_UNIT_QUESTION = IntentQuestion(
    intent_id=GEN_UNIT_INTENT.intent_id,
    question="A hiking trail is 1500 meters long. Expressed in kilometers, how long is the trail?",
    options=[
        "1.5 kilometers",
        "1500000 millimeters",
        "0.15 kilometers",
        "15 kilometers",
    ],
    correct_answer="1.5 kilometers",
    explanation="1500 meters divided by 1000 equals 1.5 kilometers.",
    concept="metric distance conversion",
    difficulty="introductory",
)
GEN_UNIT_CANDIDATE = _candidate(
    GEN_UNIT_SKILL.skill_id,
    "GEN-UNT-01-fixture-candidate",
    GEN_UNIT_INTENT.intent_id,
    GEN_UNIT_REFERENCE.reference_id,
    GEN_UNIT_QUESTION,
)


# --- GEN-CAUSAL-DISTRACTOR: distinct candidate causes, one supported (control) -----
# (non-ambiguous control -- must NOT be flagged multiple_defensible_answers)

GEN_CAUSAL_SKILL = _skill(
    "GEN-CAU-01", "Tide formation",
    "Identify the primary cause of ocean tides on Earth.",
)
GEN_CAUSAL_INTENT = _intent(
    GEN_CAUSAL_SKILL, "Identify what primarily causes ocean tides."
)
GEN_CAUSAL_REFERENCE = ReferenceProvenance(
    reference_id="GEN-CAU-01-REF-01",
    skill_id=GEN_CAUSAL_SKILL.skill_id,
    reference_material=(
        "Ocean tides are primarily caused by the gravitational pull of the Moon on "
        "Earth's oceans, with a smaller contribution from the Sun's gravity. Wind, "
        "atmospheric pressure, and Earth's rotation on its own axis do not primarily "
        "cause tides, though they can influence local wave conditions."
    ),
    title="Tide formation (calibration fixture)",
    source_url="https://example.invalid/tides",
    source_domain="example.invalid",
    content_hash="4" * 64,
    retrieved_at=FIXED_TIME,
    reviewer_id="calibration-fixture",
    reviewed_at=FIXED_TIME,
)
GEN_CAUSAL_QUESTION = IntentQuestion(
    intent_id=GEN_CAUSAL_INTENT.intent_id,
    question="What is the primary cause of ocean tides on Earth?",
    options=[
        "The Moon's gravitational pull on Earth's oceans",
        "Wind blowing across the ocean surface",
        "Earth's rotation on its own axis",
        "Changes in atmospheric pressure over the ocean",
    ],
    correct_answer="The Moon's gravitational pull on Earth's oceans",
    explanation=(
        "Ocean tides are primarily caused by the Moon's gravitational pull, with a "
        "smaller contribution from the Sun; wind, Earth's rotation, and atmospheric "
        "pressure are not the primary cause."
    ),
    concept="tide formation",
    difficulty="introductory",
)
GEN_CAUSAL_CANDIDATE = _candidate(
    GEN_CAUSAL_SKILL.skill_id,
    "GEN-CAU-01-fixture-candidate",
    GEN_CAUSAL_INTENT.intent_id,
    GEN_CAUSAL_REFERENCE.reference_id,
    GEN_CAUSAL_QUESTION,
)
