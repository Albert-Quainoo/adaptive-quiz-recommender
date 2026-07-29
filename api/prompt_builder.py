from api.schemas import QuizGenerationRequest, QuizResponse
from pydantic import BaseModel 
from typing import Literal
import json


class ChatTurn(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


def build_quiz_messages(request: QuizGenerationRequest,) -> list[dict[str,str]]:

    response_schema = json.dumps(
        QuizResponse.model_json_schema(),
        indent=2
    )
    system_text = f"""
    You are an expert educator who generates high-quality quiz questions.

    Follow these rules:
    - Return only valid JSON.
    - Do not include Markdown code fences.
    - Do not include introductory or concluding text.
    - Generate exactly the requested number of questions.
    - Give every question exactly four distinct options.
    - Make correct_answer exactly match one of the options.
    - Use the requested difficulty.
    - Keep each explanation concise.
    - Avoid ambiguous and trick questions.
    - Use plain text in all JSON string values.
    - Do not include backslash characters in generated field values.
    - Do not escape asterisks. Write "A-star" instead of "A*".

    The schema is an instruction only. Do not repeat or reproduce it.
    Output exactly one JSON object whose top-level key is "questions".

    Your response must follow this JSON schema:

    {response_schema}
    """.strip()

    user_text = (
        f"Generate a quiz based on the following requirements:\n"
        f"- Topic: {request.topic}\n"
        f"- Difficulty: {request.difficulty}\n"
        f"- Learning Objective: {request.learning_objective}\n"
        f"- Number of Questions: {request.question_count}"
    )

    turns = [
        ChatTurn(role="system", content=system_text),
        ChatTurn(role="user",content=user_text)
    ]

    return [turn.model_dump() for turn in turns]

quiz_input = QuizGenerationRequest(
    topic="Python Pydantic V2",
    difficulty="intermediate",
    learning_objective="Understand validation",
    question_count=5
)
messages = build_quiz_messages(quiz_input)

for message in messages:
    print(f"\n--- {message['role'].upper()} ---\n")
    print(message["content"])