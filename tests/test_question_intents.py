import json

import pytest

from api.prompt_builder import PROMPT_VERSION
from api.schemas import QuizQuestion
from authoring.grounded_batch import (
    BatchConfig,
    BatchGenerationError,
    generate_batch,
    validate_pilot_question,
)
from authoring.question_intents import (
    PILOT_BATCH_ID,
    QuestionIntent,
    intents_by_skill,
    load_pilot_blueprint,
)
from taxonomy.loader import course_paths, course_provenance_path
from tests.test_grounded_batch import (
    FIXED_TIME,
    GIT_COMMIT,
    DeterministicFakeModel,
    config,
    jsonl,
    raw_question,
)


SKILLS_PATH, REFERENCES_PATH = course_paths("ai")
PROVENANCE_PATH = course_provenance_path("ai")


def src01_intent(index: int) -> QuestionIntent:
    return intents_by_skill(load_pilot_blueprint())["AI-SRC-01"][index]


def question(**overrides) -> QuizQuestion:
    fields = {
        "question": "Which statement is correct?",
        "options": ["Correct", "Wrong A", "Wrong B", "Wrong C"],
        "correct_answer": "Correct",
        "explanation": "Correct is supported by the formulation.",
        "concept": "problem formulation",
        "difficulty": "intermediate",
    }
    fields.update(overrides)
    return QuizQuestion(**fields)


def test_blueprint_has_ten_distinct_reviewable_intents_per_pilot_skill():
    blueprint = load_pilot_blueprint()
    grouped = intents_by_skill(blueprint)

    assert blueprint.batch_id == PILOT_BATCH_ID
    assert blueprint.prompt_version == PROMPT_VERSION == "v3.3"
    assert set(grouped) == {"AI-SRC-01", "AI-SRC-02", "AI-SRC-08"}
    assert {skill_id: len(intents) for skill_id, intents in grouped.items()} == {
        "AI-SRC-01": 10,
        "AI-SRC-02": 10,
        "AI-SRC-08": 10,
    }
    assert len({intent.intent_id for intent in blueprint.intents}) == 30
    src01_focuses = " ".join(intent.assessment_focus for intent in grouped["AI-SRC-01"])
    for phrase in (
        "complete canonical",
        "initial state",
        "available actions",
        "transition model",
        "goal test",
        "action cost",
        "missing",
        "constraint-satisfaction",
        "Degrees",
        "invalid",
    ):
        assert phrase.casefold() in src01_focuses.casefold()


@pytest.mark.parametrize(
    "invalid_question,error",
    [
        (
            question(
                options=[
                    "Initial state and start state are two components",
                    "Wrong A",
                    "Wrong B",
                    "Wrong C",
                ],
                correct_answer="Initial state and start state are two components",
            ),
            "different components",
        ),
        (
            question(
                question="A search problem has six components. Which list is right?"
            ),
            "count must be five",
        ),
        (
            question(
                options=["Initial state", "Wrong A", "Wrong B", "Wrong C"],
                correct_answer="Initial state",
                explanation="Initial state is incorrect.",
            ),
            "contradicts",
        ),
        (
            question(question="The action cost is the accumulated whole path cost."),
            "individual action",
        ),
        (
            question(question="The path cost is the cost of one action."),
            "accumulated cost",
        ),
    ],
)
def test_canonical_terminology_and_explanation_validation(invalid_question, error):
    intent_index = 5 if error in {"individual action", "accumulated cost"} else 0
    with pytest.raises(ValueError, match=error):
        validate_pilot_question(invalid_question, src01_intent(intent_index), set())


def test_retry_prompt_contains_precise_schema_failure_and_same_intent(tmp_path):
    mismatch = lambda seed: json.dumps(
        {
            "questions": [
                {
                    "question": "Which estimate is valid?",
                    "options": ["A", "B", "C", "D"],
                    "correct_answer": "option A",
                    "explanation": "A is valid.",
                    "concept": "heuristic",
                    "difficulty": "intermediate",
                }
            ]
        }
    )
    model = DeterministicFakeModel([mismatch, lambda seed: raw_question(seed)])
    generate_batch(
        config(skill_ids=["AI-SRC-08"], questions_per_skill=1),
        model,
        tmp_path / "batch",
        skills_path=SKILLS_PATH,
        references_path=REFERENCES_PATH,
        provenance_path=PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
    )

    retry_prompt = model.calls[1]["messages"][1]["content"]
    assert "Previous validation failure:" in retry_prompt
    assert "correct_answer" in retry_prompt
    assert "copying one option verbatim" in retry_prompt
    assert "AI-SRC-08-INT-01" in retry_prompt


def test_duplicate_retry_requires_new_stem_and_scenario_and_has_avoid_list(tmp_path):
    repeated = lambda seed: raw_question(seed, question="Which route is closest?")
    model = DeterministicFakeModel(
        [repeated, repeated, lambda seed: raw_question(seed, "Which grid move remains?")]
    )
    result = generate_batch(
        config(skill_ids=["AI-SRC-08"], questions_per_skill=2),
        model,
        tmp_path / "batch",
        skills_path=SKILLS_PATH,
        references_path=REFERENCES_PATH,
        provenance_path=PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
    )

    retry_prompt = model.calls[2]["messages"][1]["content"]
    assert result.status == "complete"
    assert "different stem and a different scenario" in retry_prompt
    assert "preserving the assigned intent" in retry_prompt
    assert "- Which route is closest?" in retry_prompt
    assert "AI-SRC-08-INT-02" in retry_prompt


def test_semantically_equivalent_component_lists_cannot_fill_two_slots(tmp_path):
    complete = (
        "Initial state, actions, transition model, goal test, and path cost"
    )
    first = lambda seed: raw_question(
        seed,
        question="Which option gives the complete formulation?",
        options=[complete, "A", "B", "C"],
        correct_answer=complete,
    )
    equivalent = lambda seed: raw_question(
        seed,
        question="A draft is incomplete. Which option restores every component?",
        options=[complete, "D", "E", "F"],
        correct_answer=complete,
    )
    model = DeterministicFakeModel([first, equivalent])
    result = generate_batch(
        config(
            skill_ids=["AI-SRC-01"],
            questions_per_skill=2,
            max_attempts_per_question=1,
        ),
        model,
        tmp_path / "batch",
        skills_path=SKILLS_PATH,
        references_path=REFERENCES_PATH,
        provenance_path=PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
    )

    assert result.status == "incomplete"
    assert result.summary.accepted == 1
    assert result.summary.duplicated == 1
    assert result.attempts[-1].validation_error.endswith(
        "semantically equivalent component-list question"
    )


def test_attempt_and_question_are_persisted_incrementally(tmp_path):
    output = tmp_path / "batch"

    class InspectingModel(DeterministicFakeModel):
        def generate(self, messages, seed, generation_parameters):
            if self.calls:
                assert len(jsonl(output / "pending_questions.jsonl")) == 1
                assert len(jsonl(output / "audit.jsonl")) == 1
                assert json.loads((output / "manifest.json").read_text())["status"] == "incomplete"
            return super().generate(messages, seed, generation_parameters)

    result = generate_batch(
        config(skill_ids=["AI-SRC-08"], questions_per_skill=2),
        InspectingModel(),
        output,
        skills_path=SKILLS_PATH,
        references_path=REFERENCES_PATH,
        provenance_path=PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
    )
    assert result.status == "complete"


def test_resume_keeps_accepted_question_and_continues_exhausted_slot(tmp_path):
    output = tmp_path / "batch"
    first_model = DeterministicFakeModel(
        [lambda seed: raw_question(seed, "What was accepted before interruption?"), "bad"]
    )
    first = generate_batch(
        config(
            skill_ids=["AI-SRC-08"],
            questions_per_skill=2,
            max_attempts_per_question=1,
        ),
        first_model,
        output,
        skills_path=SKILLS_PATH,
        references_path=REFERENCES_PATH,
        provenance_path=PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
    )
    accepted_before = (output / "pending_questions.jsonl").read_bytes()
    assert first.status == "incomplete"

    resumed_model = DeterministicFakeModel(
        [lambda seed: raw_question(seed, "What was accepted after resume?")]
    )
    resumed = generate_batch(
        config(
            skill_ids=["AI-SRC-08"],
            questions_per_skill=2,
            max_attempts_per_question=1,
        ),
        resumed_model,
        output,
        skills_path=SKILLS_PATH,
        references_path=REFERENCES_PATH,
        provenance_path=PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
        resume=True,
    )

    assert resumed.status == "complete"
    assert len(resumed_model.calls) == 1
    assert (output / "pending_questions.jsonl").read_bytes().startswith(
        accepted_before.rstrip(b"\n")
    )
    assert resumed.attempts[-1].attempt_index == 1


def test_intent_reuse_requires_explicit_configuration(tmp_path):
    model = DeterministicFakeModel()
    with pytest.raises(BatchGenerationError, match="distinct intents"):
        generate_batch(
            config(skill_ids=["AI-SRC-08"], questions_per_skill=11),
            model,
            tmp_path / "batch",
            skills_path=SKILLS_PATH,
            references_path=REFERENCES_PATH,
            provenance_path=PROVENANCE_PATH,
            clock=lambda: FIXED_TIME,
            git_commit=GIT_COMMIT,
        )
    assert model.calls == []


def test_manifest_records_slot_and_accepted_intent_ids(tmp_path):
    model = DeterministicFakeModel()
    output = tmp_path / "batch"
    generate_batch(
        config(skill_ids=["AI-SRC-02"], questions_per_skill=2),
        model,
        output,
        skills_path=SKILLS_PATH,
        references_path=REFERENCES_PATH,
        provenance_path=PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    questions = jsonl(output / "pending_questions.jsonl")
    audits = jsonl(output / "audit.jsonl")

    assert manifest["slot_intent_ids"]["AI-SRC-02"] == [
        "AI-SRC-02-INT-01",
        "AI-SRC-02-INT-02",
    ]
    assert manifest["accepted_intent_ids"] == [item["intent_id"] for item in questions]
    assert all(item["intent_id"] for item in audits)
