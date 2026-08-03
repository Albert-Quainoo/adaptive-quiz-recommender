from api.schemas import QuizGenerationRequest
import pytest
from api.prompt_builder import build_quiz_messages

def test_builds_system_and_user_messages():
    valid_request = QuizGenerationRequest(
        topic="Stacks",
        difficulty="introductory",
        learning_objective="Stack operations",
        question_count=2
    )
    
    messages = build_quiz_messages(valid_request)

    assert isinstance(messages, list)
    assert len(messages) == 2

    for msg in messages:
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "content" in msg
        assert isinstance(msg["role"], str)
        assert isinstance(msg["content"], str)

    

@pytest.mark.parametrize(
    "topic, difficulty, objective, count",
    [
        ("Asyncio Loops", "advanced", "Understand the event loop matrix", 8),
        ("HTML Basics", "introductory", "Learn structural semantic tags", 1),
        ("SQL Joins", "intermediate", "Identify inner vs left joins", 5),
    ]
)
def test_user_message_contains_request_values(topic, difficulty, objective, count):
    request = QuizGenerationRequest(
        topic=topic,
        difficulty=difficulty,
        learning_objective=objective,
        question_count=count
    )

    messages = build_quiz_messages(request)
    user_content = messages[1]["content"]

    assert f"- Topic: {topic}" in user_content
    assert f"- Difficulty: {difficulty}" in user_content
    assert f"- Learning Objective: {objective}" in user_content
    assert f"- Number of Questions: {count}" in user_content



def test_system_message_contains_output_requirements():
     request_two = QuizGenerationRequest(
        topic="Any",
        difficulty="intermediate",
        learning_objective="Any",
        question_count=3
    )

     messages_one = build_quiz_messages(request_two)
     system_content = messages_one[0]["content"]

     required_system_fragments = [
        "Return only valid JSON.",
        "Do not include Markdown code fences.",
        "Generate exactly the requested number of questions.",
        "Give every question exactly four distinct options.",
        "Make correct_answer exactly match one of the options.",
        '"questions"',
        '"options"',
        '"correct_answer"',
        '"explanation"',
    ]

     for fragment in required_system_fragments:
        assert fragment in system_content, f"Required system prompt fragment is missing: '{fragment!r}'"

@pytest.fixture
def system_content():
    request = QuizGenerationRequest(
        topic="Neural networks",
        difficulty="advanced",
        learning_objective="Analyse a neural-network forward pass.",
        question_count=3,
    )

    messages = build_quiz_messages(request)

    return messages[0]["content"]

def test_system_message_defines_difficulty_levels(system_content):
    assert "Introductory:" in system_content
    assert "Intermediate:" in system_content
    assert "Advanced:" in system_content

def test_system_message_requires_objective_alignment(system_content):
    assert "directly assess the learning objective" in system_content
    assert "adjacent topics" in system_content

def test_system_message_respects_objective_verbs(system_content):
    assert "calculate, apply, analyse, or evaluate" in system_content
    assert "definition recall" in system_content

def test_system_message_prohibits_duplicate_questions(system_content):
    assert "distinct aspect of the learning objective" in system_content
    assert "Do not restate an earlier question" in system_content

def test_system_message_requires_plausible_distractors(system_content):
    assert "plausible misconceptions" in system_content
    assert "obviously unrelated options" in system_content

def test_system_message_requires_teaching_explanations(system_content):
    assert "Explain why the correct answer is correct" in system_content
    assert "Do not merely repeat the correct answer" in system_content

def test_system_message_does_not_use_broad_a_star_replacement(system_content):
    assert 'Write "A-star" instead of "A*"' not in system_content
    assert 'Do not use "-star" as an answer-option label' in system_content