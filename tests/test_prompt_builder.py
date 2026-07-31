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
