"""Regression fixture for the AI-FND-04-b4cd5c51a8cab3c4 semantic-overlap case.

A live Modal run (batch grounded-ai-fnd-release-v1, 2026-08-17) generated a
four-option Chinese Room question where every option restated the same underlying
proposition ("behavioral success without genuine understanding") in different words.
Albert's own review caught this; the live automated semantic reviewer did not --
its real response (reconstructed verbatim below from
outputs/replenishment/ai/reviews/automated_review_reports/grounded-ai-fnd-release-v1__AI-FND-04.json)
reported multiple_defensible_answers=False, duplicate_or_rephrased_distractors=[],
and recommended human approval at "low" risk.

This is not a scoring-logic defect: authoring/review/risk.py already blocks on both
multiple_defensible_answers (test_review_risk.py::test_multiple_defensible_answers_is_blocking)
and a non-empty duplicate_or_rephrased_distractors. The gap is that the reviewer
*model* did not populate either field for this candidate on this pass -- a real,
observed LLM-as-judge limitation (see authoring/review/service.py's module
docstring), not something a narrow code change can reliably fix. Marked strict
xfail rather than silently skipped or forced green: if a future reviewer prompt/model
change ever does catch this, this test starts failing as an XPASS, which is the
signal to update or remove the xfail marker.
"""

from datetime import datetime, timezone

import pytest

from authoring.grounded_batch import IntentQuestion, PendingQuestion
from authoring.question_intents import QuestionIntent
from authoring.review.config import ReviewPolicyConfig
from authoring.review.models import (
    AnswerAssessment,
    DifficultyAssessment,
    DuplicateAssessment,
    GroundingAssessment,
    ObjectiveAssessment,
    SemanticReviewResult,
)
from authoring.review.reports import AutomatedReviewReportStore
from authoring.review.reviewer import FakeContentReviewer
from authoring.review.service import review_candidate
from taxonomy.schemas import ReferenceProvenance, SkillDefinition

FIXED_TIME = datetime(2026, 8, 17, 9, 22, 57, tzinfo=timezone.utc)

SKILL = SkillDefinition(
    skill_id="AI-FND-04",
    topic="Foundations of Artificial Intelligence",
    subtopic="Philosophy of AI",
    name="Machine intelligence",
    learning_objective=(
        "Explain how the Turing Test and Chinese Room argument relate to machine "
        "intelligence and understanding."
    ),
    cognitive_process="understand",
    generation_strategy="generated",
)

INTENT = QuestionIntent(
    intent_id="AI-FND-04-INT-02",
    skill_id="AI-FND-04",
    learning_objective=SKILL.learning_objective,
    difficulty="introductory",
    assessment_focus=(
        "Given a description of the Turing Test or the Chinese Room argument, "
        "identify what it claims to test or argue."
    ),
    cognitive_demand=(
        "Given a description of the Turing Test or the Chinese Room argument, "
        "identify what it claims to test or argue."
    ),
    question_archetype="Turing Test vs. Chinese Room claim recognition",
    preferred_reference_ids=["AI-FND-04-1d9f2a7b336c", "AI-FND-04-0a36a4f50e30"],
    required_question_characteristics=[
        "Describe either the Turing Test or the Chinese Room argument.",
        "Ask what that description claims to test or argue about machine intelligence or understanding.",
    ],
    prohibited_ambiguity_patterns=[
        "Do not require the learner to judge whether the argument succeeds.",
    ],
    expected_misconception_or_distractor_strategy=[
        "Distractors should swap the Turing Test's behavioral claim with the Chinese "
        "Room's understanding-based challenge, or vice versa."
    ],
    required_concepts=[
        "the Turing Test as a behavioral test conducted through conversation",
        "the Chinese Room argument as a challenge to whether behavior alone establishes understanding",
    ],
    prohibited_conflations=[
        "conflating what the Turing Test measures with what the Chinese Room challenges",
    ],
)

APPROVED_REFERENCES = [
    ReferenceProvenance(
        reference_id="AI-FND-04-1d9f2a7b336c",
        skill_id="AI-FND-04",
        reference_material=(
            "Suppose that we have a person, a machine, and an interrogator. The "
            "interrogator is in a room separated from the other person and the "
            "machine. The object of the game is for the interrogator to determine "
            "which of the other two is the person, and which is the machine."
        ),
        title="The Turing Test (Stanford Encyclopedia of Philosophy)",
        source_url="https://plato.stanford.edu/entries/turing-test/",
        source_domain="plato.stanford.edu",
        content_hash="1d9f2a7b336c150c53a991641d46ec6cb6e52ce0ef4cfb2a95a1a8e2dee73dac",
        retrieved_at=FIXED_TIME,
        reviewer_id="claude (intro-ai foundations release candidate, delegated reference review by Claude Code on behalf of Albert Quainoo)",
        reviewed_at=FIXED_TIME,
    ),
    ReferenceProvenance(
        reference_id="AI-FND-04-0a36a4f50e30",
        skill_id="AI-FND-04",
        reference_material=(
            "The point of the argument is this: if the man in the room does not "
            "understand Chinese on the basis of implementing the appropriate program "
            "for understanding Chinese then neither does any other digital computer "
            "solely on that basis because no computer, qua computer, has anything the "
            "man does not have."
        ),
        title="The Chinese Room Argument (Stanford Encyclopedia of Philosophy)",
        source_url="https://plato.stanford.edu/entries/chinese-room/",
        source_domain="plato.stanford.edu",
        content_hash="0a36a4f50e30752cd65982302fa8ab1e29638b8f97a9d511365b14e34990d9f4",
        retrieved_at=FIXED_TIME,
        reviewer_id="claude (intro-ai foundations release candidate, delegated reference review by Claude Code on behalf of Albert Quainoo)",
        reviewed_at=FIXED_TIME,
    ),
]

# Verbatim: the original, semantically-overlapping four-option candidate as generated
# (outputs/replenishment/ai/batches/grounded-ai-fnd-release-v1__AI-FND-04/pending_questions.jsonl,
# question_id AI-FND-04-b4cd5c51a8cab3c4). Every option restates "behavior without
# understanding" vs. "behavior proves understanding" in different words -- options 1, 3,
# and 4 are arguably the same proposition as the declared answer in different phrasing.
ORIGINAL_SEMANTIC_OVERLAP_QUESTION = IntentQuestion(
    intent_id=INTENT.intent_id,
    question="What does the Chinese Room argument claim to test or argue about machine intelligence and understanding?",
    options=[
        "Whether a machine can pass a behavioral test like the Turing Test without actually understanding the language",
        "Whether a machine can appear indistinguishable from a human in conversation without having genuine understanding",
        "Whether a machine can correctly manipulate symbols to pass a test for understanding a language without understanding the language",
        "Whether a machine can be designed to pass a behavioral test like the Turing Test without any understanding of the language",
    ],
    correct_answer="Whether a machine can correctly manipulate symbols to pass a test for understanding a language without understanding the language",
    explanation=(
        "The Chinese Room argument challenges whether passing a behavioral test like "
        "the Turing Test is sufficient to establish genuine understanding, not merely "
        "correct behavioral output."
    ),
    concept="the Chinese Room argument as a challenge to whether behavior alone establishes understanding",
    difficulty="introductory",
)

ORIGINAL_SEMANTIC_OVERLAP_CANDIDATE = PendingQuestion(
    batch_id="grounded-ai-fnd-release-v1-regression-test",
    question_id="AI-FND-04-regression-original-semantic-overlap",
    skill_id="AI-FND-04",
    question_index=0,
    intent_id=INTENT.intent_id,
    seed=1,
    reference_ids=[reference.reference_id for reference in APPROVED_REFERENCES],
    prompt_version="v3.3",
    prompt_hash="385ea632a795a55d721ba5511fbd9e81c0afd9e15a029d5b112ee5734ea60821",
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    model_revision="unknown",
    generation_parameters={},
    generated_at=FIXED_TIME,
    git_commit="815247398a63e3f91ce12144960a2e659ea68d3e",
    raw_response="{}",
    question=ORIGINAL_SEMANTIC_OVERLAP_QUESTION,
)

# Reconstructed verbatim from the real live-Modal report (not fabricated): the reviewer
# independently selected the declared answer, reported it grounded and matching, and did
# NOT flag multiple_defensible_answers or duplicate_or_rephrased_distractors -- this is
# genuinely what the reviewer returned for this exact candidate.
REAL_CAPTURED_REVIEW_RESULT = SemanticReviewResult(
    grounding_assessment=GroundingAssessment(
        grounded=True,
        independently_supported_answer=True,
        consulted_reference_ids=[
            "AI-FND-04-1d9f2a7b336c",
            "AI-FND-04-0a36a4f50e30",
        ],
        supporting_reference_ids=["AI-FND-04-0a36a4f50e30"],
        supported_claims=[],
        unsupported_claims=[],
        contradictions=[],
        grounding_confidence=1.0,
    ),
    answer_assessment=AnswerAssessment(
        selected_option_text=ORIGINAL_SEMANTIC_OVERLAP_QUESTION.correct_answer,
        matches_declared_answer=True,
        option_assessments={},
        multiple_defensible_answers=False,
        obviously_signalled_answer=False,
        duplicate_or_rephrased_distractors=[],
        answer_confidence=1.0,
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
    reviewer_model_id="meta-llama/Llama-3.1-8B-Instruct",
    reviewer_model_revision="unknown",
    reviewer_prompt_version="review-v5",
    reviewer_prompt_template_hash="real-modal-run-2026-08-17",
    rendered_review_request_hash="real-modal-run-2026-08-17",
)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known live-Modal reviewer gap, observed 2026-08-17 on "
        "AI-FND-04-b4cd5c51a8cab3c4: options that restate the same proposition in "
        "different words are not reliably flagged via multiple_defensible_answers or "
        "duplicate_or_rephrased_distractors. Not a scoring-logic bug (both fields "
        "already block correctly when populated -- see test_review_risk.py); the "
        "reviewer model itself did not populate them for this candidate. No narrow "
        "code fix exists; a real fix would mean a broader reviewer prompt change, "
        "out of scope here."
    ),
)
def test_semantically_overlapping_options_should_not_be_recommended_for_human_approval(tmp_path):
    reviewer = FakeContentReviewer(
        {ORIGINAL_SEMANTIC_OVERLAP_QUESTION.question: REAL_CAPTURED_REVIEW_RESULT}
    )
    report = review_candidate(
        ORIGINAL_SEMANTIC_OVERLAP_CANDIDATE,
        SKILL,
        INTENT,
        APPROVED_REFERENCES,
        reviewer=reviewer,
        config=ReviewPolicyConfig(reviewer_passes=1),
        report_store=AutomatedReviewReportStore(tmp_path / "reports.json"),
        clock=lambda: FIXED_TIME,
    )

    assert report.recommendation != "recommend_human_approval", (
        "a candidate whose four options restate the same proposition in different "
        "words must never be recommended for human approval as clean"
    )
