"""Shared fixtures for the automated-review test suite, centered on the permanent
heuristic h(n)/g(n) regression case.

A live Modal run once generated an inaccurate answer for a heuristic h(n) question on
skill AI-SRC-08: "The cost of the cheapest path from the initial state to the goal
state" -- that is the definition of path cost g(n), not the remaining-cost estimate
h(n). Human review corrected it to "The remaining cost from the current state n to a
goal state." Only the corrected revision was ever promoted. This module fixes both
candidates so authoring/review/ can be regression-tested against the exact case that
motivated it, without depending on any generated batch on disk.
"""

from datetime import datetime, timezone

from api.schemas import QuizGenerationRequest
from authoring.grounded_batch import IntentQuestion, PendingQuestion
from authoring.grounding_briefs import PILOT_GROUNDING_BRIEFS
from authoring.question_intents import QuestionIntent
from authoring.review.models import (
    AnswerAssessment,
    DifficultyAssessment,
    DuplicateAssessment,
    GroundingAssessment,
    ObjectiveAssessment,
    SemanticReviewResult,
)
from taxonomy.schemas import ReferenceProvenance, SkillDefinition

FIXED_TIME = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)

SKILL = SkillDefinition(
    skill_id="AI-SRC-08",
    topic="Search and Problem Solving",
    subtopic="Informed search",
    name="Heuristic function",
    learning_objective="Explain how a heuristic estimates the remaining cost from a state to the goal.",
    cognitive_process="understand",
    generation_strategy="generated",
    reference_material=[
        "A heuristic function h(n) estimates the remaining cost from a state to the "
        "goal. It is distinct from g(n), the accumulated cost from the start."
    ],
    prerequisite_skill_ids=["AI-SRC-03"],
)

INTENT = QuestionIntent(
    intent_id="AI-SRC-08-INT-02",
    skill_id="AI-SRC-08",
    learning_objective=SKILL.learning_objective,
    difficulty="introductory",
    assessment_focus="Distinguish the forward heuristic estimate h(n) from accumulated path cost g(n).",
    cognitive_demand="understand",
    question_archetype="g-versus-h interpretation",
    preferred_reference_ids=["AI-SRC-08-REF-01"],
    required_question_characteristics=[
        "State both an accumulated start cost and an estimated remaining cost.",
        "Ask which value is the heuristic.",
    ],
    prohibited_ambiguity_patterns=["Do not require calculating f(n)."],
    expected_misconception_or_distractor_strategy=[
        "Distractors should swap g and h or add them despite the question asking only for h."
    ],
    required_concepts=["g(n)", "h(n)", "remaining cost"],
    prohibited_conflations=["g and h are interchangeable", "h includes cost already travelled"],
)

APPROVED_REFERENCES = [
    ReferenceProvenance(
        reference_id="AI-SRC-08-REF-01",
        skill_id="AI-SRC-08",
        reference_material=SKILL.reference_material[0],
        title="Introduction to the A* Algorithm",
        source_url="https://www.redblobgames.com/pathfinding/a-star/introduction.html",
        source_domain="redblobgames.com",
        content_hash="a" * 64,
        retrieved_at=FIXED_TIME,
        reviewer_id="albert",
        reviewed_at=FIXED_TIME,
    )
]


def _pending_question(question_id: str, question: IntentQuestion) -> PendingQuestion:
    return PendingQuestion(
        batch_id="grounded-pilot-regression-test",
        question_id=question_id,
        skill_id="AI-SRC-08",
        question_index=0,
        intent_id=INTENT.intent_id,
        seed=1,
        reference_ids=[reference.reference_id for reference in APPROVED_REFERENCES],
        prompt_version="v3.3",
        prompt_hash="b" * 64,
        model_id="test-model",
        model_revision="test-rev-1",
        generation_parameters={},
        generated_at=FIXED_TIME,
        git_commit="0" * 40,
        raw_response="{}",
        question=question,
    )


ORIGINAL_INACCURATE_QUESTION = IntentQuestion(
    intent_id=INTENT.intent_id,
    question=(
        "State the heuristic value h(n) from the following description: we have "
        "already travelled 3 units and our estimated remaining distance to the goal "
        "is 5 units. What is the heuristic value h(n)?"
    ),
    options=[
        "The cost of the cheapest path from the initial state to the goal state",
        "The accumulated cost from the start, which is 3 units",
        "The total cost from start to goal, which is 8 units",
        "The path cost, which is the sum of the accumulated cost and the estimated remaining cost",
    ],
    correct_answer="The cost of the cheapest path from the initial state to the goal state",
    explanation=(
        "The heuristic h(n) is the cost of the cheapest path from the initial state to "
        "the goal state."
    ),
    concept="heuristic value h(n)",
    difficulty="introductory",
)

CORRECTED_QUESTION = IntentQuestion(
    intent_id=INTENT.intent_id,
    question=ORIGINAL_INACCURATE_QUESTION.question,
    options=[
        "The remaining cost from the current state n to a goal state",
        "The accumulated cost from the start, which is 3 units",
        "The total cost from start to goal, which is 8 units",
        "The path cost, which is the sum of the accumulated cost and the estimated remaining cost",
    ],
    correct_answer="The remaining cost from the current state n to a goal state",
    explanation=(
        "The heuristic h(n) estimates the remaining cost from the current state n to a "
        "goal state; it never includes the 3 units already travelled, which is g(n)."
    ),
    concept="heuristic value h(n)",
    difficulty="introductory",
)

ORIGINAL_INACCURATE_CANDIDATE = _pending_question(
    "AI-SRC-08-regression-original", ORIGINAL_INACCURATE_QUESTION
)
CORRECTED_CANDIDATE = _pending_question("AI-SRC-08-regression-corrected", CORRECTED_QUESTION)


def _reviewer_result(
    *,
    grounded: bool,
    independently_supported_answer: bool,
    selected_option_text: str,
    matches_declared_answer: bool,
    contradictions: list[str] = (),
) -> SemanticReviewResult:
    return SemanticReviewResult(
        grounding_assessment=GroundingAssessment(
            grounded=grounded,
            independently_supported_answer=independently_supported_answer,
            consulted_reference_ids=["AI-SRC-08-REF-01"],
            supporting_reference_ids=["AI-SRC-08-REF-01"] if grounded else [],
            supported_claims=(
                ["h(n) estimates the remaining forward cost"] if grounded else []
            ),
            unsupported_claims=[],
            contradictions=list(contradictions),
            grounding_confidence=0.9 if grounded else 0.2,
        ),
        answer_assessment=AnswerAssessment(
            selected_option_text=selected_option_text,
            matches_declared_answer=matches_declared_answer,
            option_assessments={},
            multiple_defensible_answers=False,
            obviously_signalled_answer=False,
            duplicate_or_rephrased_distractors=[],
            answer_confidence=0.9,
        ),
        objective_assessment=ObjectiveAssessment(
            measures_declared_skill=True,
            satisfies_intent_blueprint=True,
            matches_objective_verb=True,
            cognitive_demand="understand",
            duplicates_another_intent=False,
        ),
        difficulty_assessment=DifficultyAssessment(
            difficulty_justified=True,
            explanation_depth_matches_difficulty=True,
            is_definition_recall_only=False,
        ),
        duplicate_assessment=DuplicateAssessment(),
        reviewer_model_id="fake-reviewer",
        reviewer_model_revision="fake-rev-1",
        reviewer_prompt_version="review-v1",
        reviewer_prompt_template_hash="c" * 64,
        rendered_review_request_hash="c" * 64,
    )


# The reviewer independently answers h(n) = "the remaining cost from n to a goal state"
# in both cases (it is reviewing the h(n)/g(n) distinction, not copying the candidate),
# so the conflated original disagrees with its own declared answer and is contradicted
# by the canonical grounding brief statement "h(n) estimates the remaining forward cost."
ORIGINAL_REVIEW_RESULT = _reviewer_result(
    grounded=False,
    independently_supported_answer=False,
    selected_option_text=CORRECTED_QUESTION.correct_answer,
    matches_declared_answer=False,
    contradictions=[
        "declared answer describes accumulated/path cost, not the heuristic's "
        "remaining-cost estimate stated in the canonical grounding brief"
    ],
)

CORRECTED_REVIEW_RESULT = _reviewer_result(
    grounded=True,
    independently_supported_answer=True,
    selected_option_text=CORRECTED_QUESTION.correct_answer,
    matches_declared_answer=True,
)

assert "h(n) estimates the remaining forward cost." in PILOT_GROUNDING_BRIEFS["AI-SRC-08"].statements
