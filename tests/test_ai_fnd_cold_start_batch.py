import json
from datetime import datetime, timezone

from api.bank import BankItem
from api.prompt_builder import PROMPT_VERSION
from api.schemas import QuizQuestion
from authoring.grounded_batch import BatchConfig, generate_batch
from authoring.grounded_review import build_pending_review
from authoring.grounding_briefs import grounding_brief
from authoring.question_intents import (
    COLD_START_BATCH_ID,
    intents_by_skill,
    load_blueprint_for_batch,
)
from taxonomy.loader import course_paths, course_provenance_path


SKILLS_PATH, REFERENCES_PATH = course_paths("ai")
PROVENANCE_PATH = course_provenance_path("ai")
FIXED_TIME = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)
GIT_COMMIT = "b" * 40
EXPECTED_REFERENCES = {
    "AI-FND-01-INT-01": [
        "AI-FND-01-8bbbddaf2aa6",
        "AI-FND-01-b50c85fa00a5",
    ],
    "AI-FND-01-INT-02": ["AI-FND-01-d03d77e0aca2"],
    "AI-FND-01-INT-03": ["AI-FND-01-f7c5eb1ccf76"],
}


QUESTIONS = {
    "AI-FND-01-INT-01": {
        "question": "Which behavior best shows an intelligent system learning?",
        "options": [
            "Improving future decisions after observing past outcomes",
            "Printing the same stored message whenever a button is pressed",
            "Copying a file without examining its contents",
            "Displaying a fixed clock format selected by its programmer",
        ],
        "correct_answer": "Improving future decisions after observing past outcomes",
        "explanation": "Learning from experience to improve decisions is one of the capabilities associated with intelligent systems.",
        "concept": "Core capabilities of intelligent systems",
        "difficulty": "introductory",
    },
    "AI-FND-01-INT-02": {
        "question": "A photo application identifies a person's face in newly uploaded pictures. Which capability makes this an AI application?",
        "options": [
            "Recognizing a face in image data",
            "Saving every picture under a fixed filename",
            "Showing the same welcome message at startup",
            "Turning the screen off after a fixed number of minutes",
        ],
        "correct_answer": "Recognizing a face in image data",
        "explanation": "Face recognition is a concrete AI application supported by the approved reference; fixed repeated processing alone is not enough.",
        "concept": "Real-world AI applications",
        "difficulty": "introductory",
    },
    "AI-FND-01-INT-03": {
        "question": "A delivery robot detects a box blocking its path and turns left. How are the environment and the robot's action related?",
        "options": [
            "The detected box is a percept used to select the turn",
            "The turn creates the box in the environment",
            "The robot must ignore its sensors before choosing",
            "The detected box is unrelated to any action",
        ],
        "correct_answer": "The detected box is a percept used to select the turn",
        "explanation": "An intelligent agent receives percepts from its environment and selects actions based on those percepts.",
        "concept": "Percept-to-action relationship",
        "difficulty": "introductory",
    },
}


class BlueprintFakeModel:
    model_id = "fake-ai-fnd-model"
    model_revision = "fake-ai-fnd-revision-1"

    def generate(self, messages, seed, generation_parameters):
        prompt = messages[1]["content"].split("CANONICAL GROUNDING BRIEF:", 1)[0]
        intent_id = next(intent for intent in QUESTIONS if intent in prompt)
        return json.dumps({"questions": [QUESTIONS[intent_id]]}, sort_keys=True)


def config() -> BatchConfig:
    return BatchConfig(
        batch_id=COLD_START_BATCH_ID,
        skill_ids=["AI-FND-01"],
        questions_per_skill=3,
        base_seed=42,
        model_id=BlueprintFakeModel.model_id,
        prompt_version=PROMPT_VERSION,
        difficulty="introductory",
        generation_parameters={
            "max_new_tokens": 600,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    )


def run(tmp_path, directory):
    output = tmp_path / directory
    result = generate_batch(
        config(),
        BlueprintFakeModel(),
        output,
        skills_path=SKILLS_PATH,
        references_path=REFERENCES_PATH,
        provenance_path=PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
    )
    return result, output


def test_ai_fnd_grounding_brief_and_preferred_reference_mappings_exist():
    brief = grounding_brief("AI-FND-01")
    blueprint = load_blueprint_for_batch(COLD_START_BATCH_ID)
    intents = intents_by_skill(blueprint)["AI-FND-01"]

    assert brief.intent_reference_ids == EXPECTED_REFERENCES
    assert {intent.intent_id: intent.preferred_reference_ids for intent in intents} == EXPECTED_REFERENCES


def test_blueprint_has_exactly_three_introductory_ai_fnd_intents():
    blueprint = load_blueprint_for_batch(COLD_START_BATCH_ID)

    assert len(blueprint.intents) == 3
    assert {intent.intent_id for intent in blueprint.intents} == set(EXPECTED_REFERENCES)
    assert {intent.skill_id for intent in blueprint.intents} == {"AI-FND-01"}
    assert {intent.difficulty for intent in blueprint.intents} == {"introductory"}
    assert {intent.learning_objective for intent in blueprint.intents} == {
        "Define artificial intelligence and recognise tasks that require intelligent behaviour."
    }
    assert all(intent.generation_constraints for intent in blueprint.intents)


def test_batch_construction_is_deterministic_and_pending(tmp_path):
    first_result, first = run(tmp_path, "first")
    second_result, second = run(tmp_path, "second")

    assert first_result.status == second_result.status == "complete"
    for filename in ("pending_questions.jsonl", "audit.jsonl", "manifest.json", "summary.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()
    assert {item.review_status for item in first_result.questions} == {"pending"}


def test_generated_output_validates_and_retains_provenance(tmp_path):
    result, output = run(tmp_path, "batch")

    assert result.summary.accepted == 3
    for item in result.questions:
        QuizQuestion.model_validate(item.question.model_dump(exclude={"intent_id"}))
        BankItem(
            item_id=item.question_id,
            skill_id=item.skill_id,
            provenance="generated",
            question=item.question,
        )
        assert item.reference_ids == EXPECTED_REFERENCES[item.intent_id]
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["accepted_intent_ids"] == list(EXPECTED_REFERENCES)


def test_generated_batch_has_required_intent_diversity(tmp_path):
    result, _ = run(tmp_path, "batch")
    by_intent = {item.intent_id: item.question for item in result.questions}

    assert len({question.question for question in by_intent.values()}) == 3
    assert "photo application" in by_intent["AI-FND-01-INT-02"].question
    assert sum("percept" in question.explanation.casefold() for question in by_intent.values()) == 1
    assert not any(
        "list all" in question.question.casefold()
        for question in by_intent.values()
    )


def test_complete_batch_enters_pending_human_review_without_approval(tmp_path):
    _, output = run(tmp_path, "batch")
    review = build_pending_review(output)

    assert len(review.items) == 3
    assert {item.final_review_status for item in review.items} == {"pending"}
    assert {item.intent_id for item in review.items} == set(EXPECTED_REFERENCES)
