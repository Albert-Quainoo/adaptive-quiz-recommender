import json
from datetime import datetime, timezone

import pytest

from api.prompt_builder import PROMPT_VERSION
from authoring.grounded_batch import (
    BatchConfig,
    BatchGenerationError,
    derive_seed,
    generate_batch,
    validate_pilot_question,
)
from authoring.question_intents import PILOT_BATCH_ID, QuestionIntent, load_pilot_blueprint
from scripts.generate_grounded_batch import build_parser
from taxonomy.loader import course_paths, course_provenance_path

SKILLS_PATH, REFERENCES_PATH = course_paths("ai")
PROVENANCE_PATH = course_provenance_path("ai")
FIXED_TIME = datetime(2026, 8, 6, 9, 0, tzinfo=timezone.utc)
GIT_COMMIT = "a" * 40

REFERENCE_IDS = {
    "AI-SRC-01": {
        "AI-SRC-01-4024dce75930",
        "AI-SRC-01-8ef4e1416152",
        "AI-SRC-01-9ba6548d4450",
    },
    "AI-SRC-02": {
        "AI-SRC-02-a506d362b314",
        "AI-SRC-02-aa97b7fb3bd9",
    },
    "AI-SRC-08": {
        "AI-SRC-08-a366da363e17",
        "AI-SRC-08-cbd77b22bcb9",
    },
}


def raw_question(seed: int, question: str | None = None, **overrides) -> str:
    fields = {
        "question": question or f"Which grounded statement matches seed {seed}?",
        "options": [
            f"Correct {seed}",
            f"Distractor A {seed}",
            f"Distractor B {seed}",
            f"Distractor C {seed}",
        ],
        "correct_answer": f"Correct {seed}",
        "explanation": f"The approved reference supports answer {seed}.",
        "concept": "Grounded search concept",
        "difficulty": "intermediate",
    }
    fields.update(overrides)
    return json.dumps({"questions": [fields]}, sort_keys=True)


class DeterministicFakeModel:
    model_id = "fake-grounded-model"
    model_revision = "fake-revision-1"

    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.calls = []

    def generate(self, messages, seed, generation_parameters):
        self.calls.append(
            {
                "messages": messages,
                "seed": seed,
                "generation_parameters": generation_parameters,
            }
        )
        if self.responses:
            response = self.responses.pop(0)
            return response(seed) if callable(response) else response

        prompt = messages[1]["content"]
        if '"intent_id": "AI-SRC-01-INT-01"' in prompt:
            answer = "Initial state, actions, transition model, goal test, and path cost"
            return raw_question(
                seed,
                options=[answer, f"A {seed}", f"B {seed}", f"C {seed}"],
                correct_answer=answer,
            )

        return raw_question(seed)


def config(**overrides):
    fields = {
        "batch_id": PILOT_BATCH_ID,
        "skill_ids": ["AI-SRC-01", "AI-SRC-02", "AI-SRC-08"],
        "questions_per_skill": 2,
        "base_seed": 20260806,
        "model_id": DeterministicFakeModel.model_id,
        "prompt_version": PROMPT_VERSION,
        "max_attempts_per_question": 3,
        "generation_parameters": {
            "max_new_tokens": 600,
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
        },
    }
    fields.update(overrides)
    return BatchConfig(**fields)


def run(tmp_path, model=None, batch_config=None, directory="batch"):
    output = tmp_path / directory
    result = generate_batch(
        batch_config or config(),
        model or DeterministicFakeModel(),
        output,
        skills_path=SKILLS_PATH,
        references_path=REFERENCES_PATH,
        provenance_path=PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
    )
    return result, output


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_seed_derivation_is_stable_and_uses_every_coordinate():
    expected = derive_seed("batch", "AI-SRC-01", 4, 2, 99)

    assert expected == derive_seed("batch", "AI-SRC-01", 4, 2, 99)
    assert len(
        {
            expected,
            derive_seed("other", "AI-SRC-01", 4, 2, 99),
            derive_seed("batch", "AI-SRC-02", 4, 2, 99),
            derive_seed("batch", "AI-SRC-01", 5, 2, 99),
            derive_seed("batch", "AI-SRC-01", 4, 3, 99),
            derive_seed("batch", "AI-SRC-01", 4, 2, 100),
        }
    ) == 6


def test_cli_accepts_the_required_reproducibility_coordinates(tmp_path):
    arguments = build_parser().parse_args(
        [
            "--batch-id",
            "batch-1",
            "--skill-id",
            "AI-SRC-01",
            "--skill-id",
            "AI-SRC-02",
            "--questions-per-skill",
            "10",
            "--base-seed",
            "42",
            "--output",
            str(tmp_path / "batch"),
            "--model-id",
            "model/repository",
            "--prompt-version",
            PROMPT_VERSION,
        ]
    )

    assert arguments.skill_id == ["AI-SRC-01", "AI-SRC-02"]
    assert arguments.questions_per_skill == 10
    assert arguments.base_seed == 42


def test_identical_inputs_produce_identical_fake_model_batches(tmp_path):
    _, first = run(tmp_path, directory="first")
    _, second = run(tmp_path, directory="second")

    for filename in (
        "pending_questions.jsonl",
        "manifest.json",
        "audit.jsonl",
        "summary.json",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_only_approved_same_skill_references_are_used(tmp_path):
    _, output = run(tmp_path)

    for record in jsonl(output / "pending_questions.jsonl"):
        assert set(record["reference_ids"]) <= REFERENCE_IDS[record["skill_id"]]


def test_pending_question_provenance_is_complete(tmp_path):
    _, output = run(
        tmp_path,
        batch_config=config(skill_ids=["AI-SRC-08"], questions_per_skill=1),
    )
    record = jsonl(output / "pending_questions.jsonl")[0]

    assert set(record) == {
        "batch_id",
        "question_id",
        "skill_id",
        "question_index",
        "intent_id",
        "seed",
        "reference_ids",
        "prompt_version",
        "prompt_hash",
        "model_id",
        "model_revision",
        "generation_parameters",
        "validation_status",
        "review_status",
        "generated_at",
        "git_commit",
        "raw_response",
        "question",
    }
    assert record["validation_status"] == "valid"
    assert record["review_status"] == "pending"
    assert record["model_revision"] == DeterministicFakeModel.model_revision
    assert len(record["prompt_hash"]) == 64
    assert record["raw_response"]
    assert record["question"]["explanation"]
    assert record["question"]["intent_id"] == record["intent_id"]

    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert {
        reference["reference_id"] for reference in manifest["references"]
    } == REFERENCE_IDS["AI-SRC-08"]
    assert all(reference["content_hash"] for reference in manifest["references"])


def test_invalid_raw_output_is_retained_and_retried(tmp_path):
    empty_explanation = lambda seed: raw_question(seed, explanation="")
    model = DeterministicFakeModel(
        [empty_explanation, "not json", lambda seed: raw_question(seed)]
    )
    _, output = run(
        tmp_path,
        model=model,
        batch_config=config(skill_ids=["AI-SRC-08"], questions_per_skill=1),
    )
    attempts = jsonl(output / "audit.jsonl")

    assert len(attempts) == 3
    assert attempts[0]["validation_status"] == "invalid"
    assert attempts[0]["parsed_question"]
    assert attempts[0]["validation_error"]
    assert attempts[1]["raw_response"] == "not json"
    assert attempts[1]["parsed_question"] is None
    assert attempts[2]["validation_status"] == "accepted"
    assert attempts[2]["parsed_question"]


def test_retries_are_bounded_and_incomplete_artifacts_are_written(tmp_path):
    model = DeterministicFakeModel(["bad", "still bad"])
    output = tmp_path / "failed"
    batch_config = config(
        skill_ids=["AI-SRC-08"],
        questions_per_skill=1,
        max_attempts_per_question=2,
    )

    result = generate_batch(
        batch_config,
        model,
        output,
        skills_path=SKILLS_PATH,
        references_path=REFERENCES_PATH,
        provenance_path=PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
    )

    assert result.status == "incomplete"
    assert len(model.calls) == 2
    assert len(jsonl(output / "audit.jsonl")) == 2
    assert json.loads((output / "summary.json").read_text()) == {
        "requested": 1,
        "accepted": 0,
        "rejected": 2,
        "duplicated": 0,
    }


def test_duplicate_questions_are_rejected_and_bounded(tmp_path):
    first = lambda seed: raw_question(seed, question="What is repeated?")
    changed_format = lambda seed: raw_question(seed, question="  WHAT is repeated?  ")
    model = DeterministicFakeModel([first, changed_format, changed_format])
    output = tmp_path / "duplicates"
    batch_config = config(
        skill_ids=["AI-SRC-08"],
        questions_per_skill=2,
        max_attempts_per_question=2,
    )

    result = generate_batch(
        batch_config,
        model,
        output,
        skills_path=SKILLS_PATH,
        references_path=REFERENCES_PATH,
        provenance_path=PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
    )

    assert result.status == "incomplete"
    attempts = jsonl(output / "audit.jsonl")
    assert [attempt["validation_status"] for attempt in attempts] == [
        "accepted",
        "duplicate",
        "duplicate",
    ]
    assert json.loads((output / "summary.json").read_text())["duplicated"] == 2


@pytest.mark.parametrize("skill_id", ["AI-SRC-99", "AI-SRC-03"])
def test_unknown_or_not_generation_ready_skill_fails_before_model_use(
    tmp_path, skill_id
):
    model = DeterministicFakeModel()

    with pytest.raises(BatchGenerationError, match="unknown|generation-ready"):
        run(
            tmp_path,
            model=model,
            batch_config=config(skill_ids=[skill_id], questions_per_skill=1),
        )

    assert model.calls == []
