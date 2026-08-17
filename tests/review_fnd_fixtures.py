"""Shared fixtures for the real intro-ai foundations release candidates used to prove
the option_assessments hardening (authoring/review/response_parser.py) actually
discriminates between a defective candidate and its good counterparts.

Every SkillDefinition/QuestionIntent/ReferenceProvenance/question here is transcribed
verbatim from real, already-committed pipeline artifacts -- never fabricated:

- AI-FND-04's skill/intent/references/original-candidate come from
  authoring/blueprints/grounded-ai-fnd-release-v1.json,
  taxonomy/data/ai/reference_provenance.csv, and
  outputs/replenishment/ai/batches/grounded-ai-fnd-release-v1__AI-FND-04/
  pending_questions.jsonl (question_id AI-FND-04-b4cd5c51a8cab3c4) -- the same source
  tests/test_review_ai_fnd_04_semantic_overlap_regression.py already used.
- AI-FND-04's REVISED_QUESTION is Albert's real approved human revision of that same
  candidate (outputs/replenishment/ai/reviews/grounded-ai-fnd-release-v1__AI-FND-04.json,
  revision_id AI-FND-04-b4cd5c51a8cab3c4-rev-ac7c2a7defa5), now live in the active bank.
- AI-FND-03's skill/intent/references/KNOWN_GOOD_QUESTION come from
  authoring/blueprints/grounded-ai-fnd-release-v1.json,
  taxonomy/data/ai/reference_provenance.csv, and the real approved revision in
  outputs/replenishment/ai/reviews/grounded-ai-fnd-release-v1__AI-FND-03.json
  (revision_id AI-FND-03-216e939347f36d4f-rev-880bbd38fda4), also live in the active
  bank -- a genuinely good, currently-approved candidate, not a synthetic control.
"""

from datetime import datetime, timezone

from authoring.grounded_batch import IntentQuestion, PendingQuestion
from authoring.question_intents import QuestionIntent
from taxonomy.schemas import ReferenceProvenance, SkillDefinition

FIXED_TIME = datetime(2026, 8, 17, 9, 22, 57, tzinfo=timezone.utc)

# --- AI-FND-04: Chinese Room / Turing Test -------------------------------------------

FND04_SKILL = SkillDefinition(
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

FND04_INTENT = QuestionIntent(
    intent_id="AI-FND-04-INT-02",
    skill_id="AI-FND-04",
    learning_objective=FND04_SKILL.learning_objective,
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

FND04_APPROVED_REFERENCES = [
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


def _fnd04_pending_question(question_id: str, question: IntentQuestion) -> PendingQuestion:
    return PendingQuestion(
        batch_id="grounded-ai-fnd-release-v1-regression-test",
        question_id=question_id,
        skill_id="AI-FND-04",
        question_index=0,
        intent_id=FND04_INTENT.intent_id,
        seed=1,
        reference_ids=[reference.reference_id for reference in FND04_APPROVED_REFERENCES],
        prompt_version="v3.3",
        prompt_hash="385ea632a795a55d721ba5511fbd9e81c0afd9e15a029d5b112ee5734ea60821",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_revision="unknown",
        generation_parameters={},
        generated_at=FIXED_TIME,
        git_commit="815247398a63e3f91ce12144960a2e659ea68d3e",
        raw_response="{}",
        question=question,
    )


# Verbatim: the original, defective four-option candidate as generated
# (outputs/replenishment/ai/batches/grounded-ai-fnd-release-v1__AI-FND-04/
# pending_questions.jsonl, question_id AI-FND-04-b4cd5c51a8cab3c4). Every option
# restates "behavior without understanding" vs. "behavior proves understanding" in
# different words -- options 1, 3, and 4 are arguably the same proposition as the
# declared answer in different phrasing.
FND04_ORIGINAL_QUESTION = IntentQuestion(
    intent_id=FND04_INTENT.intent_id,
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

FND04_ORIGINAL_CANDIDATE = _fnd04_pending_question(
    "AI-FND-04-regression-original-semantic-overlap", FND04_ORIGINAL_QUESTION
)

# Verbatim: Albert's real approved human revision of the same candidate
# (outputs/replenishment/ai/reviews/grounded-ai-fnd-release-v1__AI-FND-04.json,
# revision_id AI-FND-04-b4cd5c51a8cab3c4-rev-ac7c2a7defa5) -- four options that are now
# textually and semantically distinct propositions, live in the active bank.
FND04_REVISED_QUESTION = IntentQuestion(
    intent_id=FND04_INTENT.intent_id,
    question="What does the Chinese Room thought experiment argue about machine intelligence?",
    options=[
        "Correct rule-based symbol manipulation can produce appropriate responses without genuine understanding.",
        "Passing the Turing Test proves that a machine genuinely understands language.",
        "Replacing symbolic programs with neural networks automatically creates genuine understanding.",
        "The speed at which a machine processes language determines whether it understands it.",
    ],
    correct_answer="Correct rule-based symbol manipulation can produce appropriate responses without genuine understanding.",
    explanation=(
        "The Chinese Room argues that manipulating symbols according to formal rules "
        "can produce convincing responses without semantic understanding. Therefore, "
        "successful language behaviour alone may not establish genuine understanding."
    ),
    concept="the Chinese Room argument as a challenge to whether behavior alone establishes understanding",
    difficulty="introductory",
)

FND04_REVISED_CANDIDATE = _fnd04_pending_question(
    "AI-FND-04-regression-approved-revision", FND04_REVISED_QUESTION
)


# --- AI-FND-03: real-world AI subfield recognition -----------------------------------

FND03_SKILL = SkillDefinition(
    skill_id="AI-FND-03",
    topic="Foundations of Artificial Intelligence",
    subtopic="AI applications",
    name="Real-world AI systems",
    learning_objective="Identify appropriate applications of AI in different fields.",
    cognitive_process="apply",
    generation_strategy="generated",
)

FND03_INTENT = QuestionIntent(
    intent_id="AI-FND-03-INT-01",
    skill_id="AI-FND-03",
    learning_objective=FND03_SKILL.learning_objective,
    difficulty="introductory",
    assessment_focus="Recognize which AI subfield or technique a given real-world scenario exemplifies.",
    cognitive_demand="Recognize which single AI subfield a described real-world scenario exemplifies.",
    question_archetype="AI subfield recognition from a real-world scenario",
    preferred_reference_ids=["AI-FND-03-1277170ce874", "AI-FND-03-81534113035f"],
    required_question_characteristics=[
        "State one clear real-world scenario.",
        "Ask which single AI subfield or technique the scenario exemplifies.",
    ],
    prohibited_ambiguity_patterns=[
        "Do not present a scenario that plausibly fits more than one subfield equally well.",
        "Do not require ranking or comparing multiple scenarios.",
    ],
    expected_misconception_or_distractor_strategy=[
        "Distractors should name a plausible but incorrect AI subfield that "
        "superficially resembles the scenario, e.g. confusing search-based "
        "pathfinding with machine learning."
    ],
    required_concepts=[
        "AI subfields (search, machine learning, natural language processing, computer vision)",
        "mapping a concrete real-world scenario to the technique it exemplifies",
    ],
    prohibited_conflations=[
        "conflating machine learning with AI in general",
        "conflating a specific technique such as search with any AI application whatsoever",
    ],
)

FND03_APPROVED_REFERENCES = [
    ReferenceProvenance(
        reference_id="AI-FND-03-1277170ce874",
        skill_id="AI-FND-03",
        reference_material=(
            "This course explores the concepts and algorithms at the foundation of "
            "modern artificial intelligence, diving into the ideas that give rise to "
            "technologies like game-playing engines, handwriting recognition, and "
            "machine translation."
        ),
        title="CS50's Introduction to Artificial Intelligence with Python",
        source_url="https://cs50.harvard.edu/ai/2024/",
        source_domain="cs50.harvard.edu",
        content_hash="1277170ce874150402521f9465c13f4ff3754106969342bf3028174af50b35c8",
        retrieved_at=FIXED_TIME,
        reviewer_id="claude (intro-ai foundations release candidate, delegated reference review by Claude Code on behalf of Albert Quainoo)",
        reviewed_at=FIXED_TIME,
    ),
    ReferenceProvenance(
        reference_id="AI-FND-03-81534113035f",
        skill_id="AI-FND-03",
        reference_material=(
            "Graph search algorithms let us find the shortest path on a map "
            "represented as a graph."
        ),
        title="Introduction to the A* Algorithm -- Red Blob Games",
        source_url="https://www.redblobgames.com/pathfinding/a-star/introduction.html",
        source_domain="redblobgames.com",
        content_hash="81534113035fc777e027d7bf5f5257c2caaee4fa334db028de81a14585bf04bc",
        retrieved_at=FIXED_TIME,
        reviewer_id="claude (intro-ai foundations release candidate, delegated reference review by Claude Code on behalf of Albert Quainoo)",
        reviewed_at=FIXED_TIME,
    ),
]

# Verbatim: Albert's real approved human revision, now live in the active bank
# (outputs/replenishment/ai/reviews/grounded-ai-fnd-release-v1__AI-FND-03.json,
# revision_id AI-FND-03-216e939347f36d4f-rev-880bbd38fda4) -- a genuinely good,
# currently-approved control candidate.
FND03_KNOWN_GOOD_QUESTION = IntentQuestion(
    intent_id=FND03_INTENT.intent_id,
    question="Which AI subfield is exemplified by a GPS navigation system that finds the shortest route between two locations?",
    options=[
        "Machine learning",
        "Natural language processing",
        "Computer vision",
        "Graph search",
    ],
    correct_answer="Graph search",
    explanation=(
        "Graph search algorithms are used to find the shortest path on a map "
        "represented as a graph, which is the basis of applications such as GPS "
        "navigation."
    ),
    concept="Graph search",
    difficulty="introductory",
)

FND03_KNOWN_GOOD_CANDIDATE = PendingQuestion(
    batch_id="grounded-ai-fnd-release-v1-regression-test",
    question_id="AI-FND-03-regression-known-good",
    skill_id="AI-FND-03",
    question_index=0,
    intent_id=FND03_INTENT.intent_id,
    seed=1,
    reference_ids=[reference.reference_id for reference in FND03_APPROVED_REFERENCES],
    prompt_version="v3.3",
    prompt_hash="5baa90699657771636f7f3459803ebc5d1f16aedec1b3d05386e836c334670c8",
    model_id="meta-llama/Llama-3.1-8B-Instruct",
    model_revision="unknown",
    generation_parameters={},
    generated_at=FIXED_TIME,
    git_commit="815247398a63e3f91ce12144960a2e659ea68d3e",
    raw_response="{}",
    question=FND03_KNOWN_GOOD_QUESTION,
)
