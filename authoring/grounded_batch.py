"""Reproducible, review-first generation of grounded question batches."""

import hashlib
import json
import re
import subprocess
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, field_validator

from api.bank import BankItem
from api.prompt_builder import PROMPT_VERSION, build_grounded_quiz_messages
from api.response_parser import parse_quiz_messages
from api.schemas import QuizGenerationRequest, QuizQuestion, difficulty_level
from authoring.question_intents import (
    COLD_START_BATCH_ID,
    QuestionIntent,
    intents_by_skill,
    load_blueprint_for_batch,
)
from authoring.grounding_briefs import grounding_brief
from authoring.question_quality import generic_quality_issues
from taxonomy.loader import load_reference_provenance, load_skills
from taxonomy.schemas import ReferenceProvenance, SkillDefinition

GenerationValue = bool | int | float | str
AttemptStatus = Literal["accepted", "invalid", "duplicate"]
BatchDifficulty = difficulty_level | Literal["mixed"]


class BatchGenerationError(RuntimeError):
    pass


class BatchModel(Protocol):
    model_id: str
    model_revision: str

    def generate(
        self,
        messages: list[dict[str, str]],
        seed: int,
        generation_parameters: dict[str, GenerationValue],
    ) -> str: ...


class BatchConfig(BaseModel):
    batch_id: str = Field(min_length=1)
    skill_ids: list[str] = Field(min_length=1)
    questions_per_skill: int | None = Field(default=None, ge=1)
    base_seed: int = Field(ge=0)
    model_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    difficulty: BatchDifficulty = "intermediate"
    max_attempts_per_question: int = Field(default=3, ge=1, le=20)
    allow_intent_reuse: bool = False
    generation_parameters: dict[str, GenerationValue] = Field(default_factory=dict)

    @field_validator("batch_id", "model_id", "prompt_version", mode="before")
    @classmethod
    def strip_identifier(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value

    @field_validator("skill_ids", mode="before")
    @classmethod
    def normalise_skill_ids(cls, value: list) -> list:
        if not isinstance(value, list):
            return value

        return [item.strip().upper() if isinstance(item, str) else item for item in value]

    @field_validator("skill_ids")
    @classmethod
    def reject_duplicate_skill_ids(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("skill_ids must be distinct")

        return value


class IntentQuestion(QuizQuestion):
    intent_id: str = Field(min_length=1)


class PendingQuestion(BaseModel):
    batch_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    skill_id: str
    question_index: int = Field(ge=0)
    intent_id: str = Field(min_length=1)
    seed: int
    reference_ids: list[str] = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt_hash: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    generation_parameters: dict[str, GenerationValue]
    validation_status: Literal["valid"] = "valid"
    review_status: Literal["pending"] = "pending"
    generated_at: datetime
    git_commit: str = Field(min_length=1)
    raw_response: str
    question: IntentQuestion


class AttemptAudit(BaseModel):
    batch_id: str = Field(min_length=1)
    skill_id: str
    intent_id: str = Field(min_length=1)
    question_index: int
    attempt_index: int
    seed: int
    reference_ids: list[str] = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt_hash: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    generation_parameters: dict[str, GenerationValue]
    validation_status: AttemptStatus
    validation_error: str | None = None
    generated_at: datetime
    raw_response: str | None = None
    parsed_question: IntentQuestion | None = None


class BatchSummary(BaseModel):
    requested: int
    accepted: int
    rejected: int
    duplicated: int


class BatchResult(BaseModel):
    questions: list[PendingQuestion]
    attempts: list[AttemptAudit]
    summary: BatchSummary
    status: Literal["complete", "incomplete"]


def derive_seed(
    batch_id: str,
    skill_id: str,
    question_index: int,
    attempt_index: int,
    base_seed: int,
) -> int:
    coordinates = "\0".join(
        (
            str(base_seed),
            batch_id,
            skill_id,
            str(question_index),
            str(attempt_index),
        )
    )
    digest = hashlib.sha256(coordinates.encode("utf-8")).digest()

    # Transformers seeds Python, NumPy and Torch together; NumPy requires a
    # 32-bit seed even though Torch accepts a wider integer.
    return int.from_bytes(digest[:4], "big")


def prompt_hash(messages: list[dict[str, str]]) -> str:
    encoded = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()


def normalise_question(text: str) -> str:
    return " ".join(text.casefold().split())


def question_id(batch_id: str, skill_id: str, question_index: int) -> str:
    digest = hashlib.sha256(
        f"{batch_id}\0{skill_id}\0{question_index}".encode("utf-8")
    ).hexdigest()[:16]

    return f"{skill_id}-{digest}"


def current_git_commit() -> str:
    project_root = Path(__file__).resolve().parents[1]

    try:
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            cwd=project_root,
        ).stdout.strip()
        if status:
            raise BatchGenerationError(
                "the working tree is dirty; commit the batch implementation first"
            )

        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            cwd=project_root,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise BatchGenerationError("cannot determine the git commit") from error


def validate_inputs(
    config: BatchConfig,
    model: BatchModel,
    skills: Sequence[SkillDefinition],
    provenance: Sequence[ReferenceProvenance],
) -> tuple[dict[str, SkillDefinition], dict[str, list[ReferenceProvenance]]]:
    if config.prompt_version != PROMPT_VERSION:
        raise BatchGenerationError(
            f"prompt version {config.prompt_version} does not match {PROMPT_VERSION}"
        )

    if config.model_id != model.model_id:
        raise BatchGenerationError(
            f"model id {config.model_id} does not match loaded model {model.model_id}"
        )

    if not model.model_revision.strip():
        raise BatchGenerationError("loaded model has no model revision")

    by_skill = {skill.skill_id: skill for skill in skills}
    references: dict[str, list[ReferenceProvenance]] = {}

    for record in provenance:
        references.setdefault(record.skill_id, []).append(record)

    for skill_id in config.skill_ids:
        skill = by_skill.get(skill_id)
        if skill is None:
            raise BatchGenerationError(f"unknown skill id {skill_id}")
        if skill.generation_strategy != "generated" or not skill.reference_material:
            raise BatchGenerationError(f"{skill_id} is not generation-ready")
        if not references.get(skill_id):
            raise BatchGenerationError(
                f"{skill_id} has no approved reference provenance"
            )
        if any(record.skill_id != skill_id for record in references[skill_id]):
            raise BatchGenerationError(
                f"{skill_id} has a reference id belonging to another skill"
            )

        canonical_passages = set(skill.reference_material)
        provenance_passages = {
            record.reference_material for record in references[skill_id]
        }
        if canonical_passages != provenance_passages:
            raise BatchGenerationError(
                f"{skill_id} canonical references and provenance do not match"
            )

        references[skill_id].sort(key=lambda record: record.reference_id)

    return by_skill, references


def validate_blueprint_config(config: BatchConfig) -> None:
    if config.batch_id != COLD_START_BATCH_ID:
        return
    if config.skill_ids != ["AI-FND-01"]:
        raise BatchGenerationError(
            "the AI-FND-01 cold-start batch must select only AI-FND-01"
        )
    if config.questions_per_skill != 3:
        raise BatchGenerationError(
            "the AI-FND-01 cold-start batch must contain exactly three questions"
        )
    if config.difficulty != "introductory":
        raise BatchGenerationError(
            "the AI-FND-01 cold-start batch must use introductory difficulty"
        )
    if config.allow_intent_reuse:
        raise BatchGenerationError(
            "the AI-FND-01 cold-start batch cannot reuse question intents"
        )


def slot_count(config: BatchConfig, intents: Sequence[QuestionIntent]) -> int:
    """How many reviewed intents this skill contributes to the batch."""
    return config.questions_per_skill or len(intents)


def parse_raw_question(raw_response: str, intent_id: str) -> IntentQuestion:
    response = parse_quiz_messages(raw_response)
    if len(response.questions) != 1:
        raise ValueError(
            f"expected exactly one question, received {len(response.questions)}"
        )

    return IntentQuestion.model_validate(
        {**response.questions[0].model_dump(), "intent_id": intent_id}
    )


def validate_question(question: QuizQuestion) -> None:
    if not question.question.strip():
        raise ValueError("question text is empty")

    normalised_options = [
        " ".join(option.casefold().split()) for option in question.options
    ]
    if len(normalised_options) != len(set(normalised_options)):
        raise ValueError("answer options are not distinct")
    if question.correct_answer not in question.options:
        raise ValueError("correct_answer does not exactly match an option")
    if not question.explanation.strip():
        raise ValueError("explanation is empty")


NUMBER_WORDS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
}
COMPONENT_PATTERNS = {
    "initial state": r"\b(?:initial|start) state\b",
    "actions": r"\bactions?\b",
    "transition model": r"\btransition model\b",
    "goal test": r"\bgoal test\b",
    "path cost": r"\bpath cost\b",
}


def _claimed_component_counts(text: str) -> set[int]:
    pattern = re.compile(
        r"(?:search problem|problem formulation)[^.]{0,50}?"
        r"(?:has|contains|consists of|comprises|requires|uses)\s+"
        r"(\d+|one|two|three|four|five|six|seven)\s+"
        r"(?:canonical\s+)?(?:components|elements|parts)",
        re.IGNORECASE,
    )
    counts = set()
    for match in pattern.finditer(text):
        value = match.group(1).casefold()
        counts.add(int(value) if value.isdigit() else NUMBER_WORDS[value])
    return counts


def _component_signature(text: str) -> frozenset[str] | None:
    components = {
        name for name, pattern in COMPONENT_PATTERNS.items() if re.search(pattern, text, re.I)
    }
    return frozenset(components) if len(components) >= 4 else None


def semantic_component_list_key(
    question: QuizQuestion, intent: QuestionIntent
) -> frozenset[str] | None:
    if intent.skill_id != "AI-SRC-01":
        return None
    # A genuine component-list question places a list in the answer. A
    # missing-component question may name four components in its stem but has
    # a single component as its answer, so it remains a distinct assessment.
    return _component_signature(question.correct_answer)


def validate_pilot_question(
    question: QuizQuestion,
    intent: QuestionIntent,
    accepted_component_keys: set[frozenset[str]],
) -> frozenset[str] | None:
    """Apply deterministic terminology and diversity gates for the v2 pilot."""
    answer_pattern = re.escape(" ".join(question.correct_answer.casefold().split()))
    explanation = " ".join(question.explanation.casefold().split())
    if re.search(
        rf"(?:{answer_pattern})\s+(?:is|are)\s+(?:incorrect|false|not correct)",
        explanation,
    ):
        raise ValueError("explanation contradicts the correct option")
    if intent.skill_id == "AI-FND-01":
        authoritative_text = " ".join(
            (question.question, question.correct_answer, question.explanation)
        )
        lowered = authoritative_text.casefold()
        cold_start_intents = {
            "AI-FND-01-INT-01",
            "AI-FND-01-INT-02",
            "AI-FND-01-INT-03",
        }
        if (
            intent.intent_id in cold_start_intents
            and question.difficulty != "introductory"
        ):
            raise ValueError("AI-FND-01 cold-start questions must be introductory")
        if intent.intent_id == "AI-FND-01-INT-01" and re.search(
            r"\b(?:entire|complete|full) list\b|\blist all\b", lowered
        ):
            raise ValueError("core-capabilities intent must not require full-list recall")
        if intent.intent_id == "AI-FND-01-INT-01":
            capability_pattern = re.compile(
                r"\b(?:problem solving|reasoning|decisions?|decision making|"
                r"learning|learns?|translating|translation|video games?|game playing)\b",
                re.I,
            )
            capable_options = [
                option for option in question.options if capability_pattern.search(option)
            ]
            if len(capable_options) != 1:
                raise ValueError(
                    "core-capabilities intent needs exactly one intelligent-capability option"
                )
        if intent.intent_id == "AI-FND-01-INT-02":
            application_pattern = re.compile(
                r"\b(?:faces?|photographs?|photos?|games?|chess|speech|siri|alexa|"
                r"routes?|navigation|navigator)\b",
                re.I,
            )
            application_options = [
                option for option in question.options if application_pattern.search(option)
            ]
            if len(application_options) != 1 or not application_pattern.search(
                question.correct_answer
            ):
                raise ValueError(
                    "real-world application intent needs exactly one grounded AI application"
                )
            incorrect_options = [
                option
                for option in question.options
                if option != question.correct_answer
            ]
            fixed_pattern = re.compile(
                r"\b(?:fixed|preset|same|exactly|regardless|without interpreting|"
                r"unchanged|pre-programmed|rule-based|calculator|timer|conveyor)\b",
                re.I,
            )
            if any(not fixed_pattern.search(option) for option in incorrect_options):
                raise ValueError(
                    "real-world application distractors must be explicitly fixed operations"
                )
            if re.search(r"\bevery automated (?:system|process) is (?:an? )?ai\b", lowered):
                raise ValueError("automation alone must not be presented as AI")
        if intent.intent_id == "AI-FND-01-INT-03":
            stem = question.question.casefold()
            has_percept = any(
                term in stem
                for term in ("percept", "environment", "detect", "sense", "sensor", "observ")
            )
            has_action = any(
                term in stem
                for term in ("action", "choose", "select", "respond", "move", "stop", "turn")
            )
            if not has_percept or not has_action:
                raise ValueError(
                    "agent intent must connect an environmental percept to an action"
                )
            if re.search(r"\btuple notation\b|\bmathematical function definition\b", lowered):
                raise ValueError("agent intent must not require formal notation")
        return None
    if intent.skill_id != "AI-SRC-01":
        return None

    fields = [question.question, *question.options, question.explanation]
    for text in fields:
        lowered = " ".join(text.casefold().split())
        has_both_synonyms = "initial state" in lowered and "start state" in lowered
        marks_synonyms = any(
            marker in lowered
            for marker in ("synonym", "same component", "also called", "another name")
        )
        if has_both_synonyms and not marks_synonyms:
            raise ValueError(
                "initial state and start state are presented as different components"
            )

    authoritative_text = " ".join(
        (question.question, question.correct_answer, question.explanation)
    )
    counts = _claimed_component_counts(authoritative_text)
    if any(count != 5 for count in counts):
        raise ValueError("canonical search-problem component count must be five")

    if intent.intent_id == "AI-SRC-01-INT-01":
        signature = _component_signature(question.correct_answer)
        has_alternative_component = bool(
            re.search(r"\b(?:state space|action cost)\b", question.correct_answer, re.I)
        )
        if signature != frozenset(COMPONENT_PATTERNS) or has_alternative_component:
            raise ValueError(
                "complete formulation must use exactly the canonical five components"
            )

    lowered_explanation = question.explanation.casefold()
    for component, pattern in COMPONENT_PATTERNS.items():
        if re.search(pattern, question.correct_answer, re.I) and re.search(
            rf"\b{re.escape(component)}\b\s+(?:is|are)\s+(?:not|incorrect)",
            lowered_explanation,
        ):
            raise ValueError("explanation contradicts the correct option")

    if re.search(r"action cost[^.]{0,50}\b(?:accumulated|whole path|entire path)\b", authoritative_text, re.I):
        raise ValueError("action cost is the cost of an individual action")
    if re.search(r"path cost[^.]{0,50}\b(?:individual|single|one) action\b", authoritative_text, re.I):
        raise ValueError("path cost is the accumulated cost of a path")

    answer_counts = _claimed_component_counts(question.correct_answer)
    explanation_counts = _claimed_component_counts(question.explanation)
    if answer_counts and explanation_counts and answer_counts != explanation_counts:
        raise ValueError("explanation contradicts the correct option's component count")

    key = semantic_component_list_key(question, intent)
    if key is not None and key in accepted_component_keys:
        raise ValueError("semantically equivalent component-list question")
    return key


def validate_cold_start_diversity(questions: Sequence[PendingQuestion]) -> None:
    relevant = [
        item for item in questions if item.batch_id == COLD_START_BATCH_ID
    ]
    if not relevant:
        return
    expected = {
        "AI-FND-01-INT-01",
        "AI-FND-01-INT-02",
        "AI-FND-01-INT-03",
    }
    intent_ids = [item.intent_id for item in relevant]
    if len(relevant) != 3 or set(intent_ids) != expected:
        raise ValueError("cold-start batch must cover each of its three intents once")
    stems = [normalise_question(item.question.question) for item in relevant]
    if len(stems) != len(set(stems)):
        raise ValueError("cold-start batch question texts must be distinct")
    if sum(intent_id == "AI-FND-01-INT-03" for intent_id in intent_ids) != 1:
        raise ValueError("cold-start batch must contain exactly one agent question")
    if sum(intent_id == "AI-FND-01-INT-02" for intent_id in intent_ids) < 1:
        raise ValueError("cold-start batch must contain a scenario-based application question")


def corrective_prompt(status: AttemptStatus, validation_error: str) -> str:
    if status == "duplicate":
        return (
            f"Previous validation failure: {validation_error}. Use a different stem "
            "and a different scenario while preserving the assigned intent."
        )
    if "correct_answer" in validation_error or "Correct answer" in validation_error:
        return (
            f"Previous validation failure: {validation_error}. Set correct_answer "
            "by copying one option verbatim."
        )
    return (
        f"Previous validation failure: {validation_error}. Correct this precise "
        "schema or validation violation while preserving the assigned intent."
    )


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_jsonl(path: Path, values: Sequence[BaseModel]) -> None:
    lines = [
        json.dumps(
            value.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        for value in values
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )
    temporary.replace(path)


def read_jsonl(path: Path, model_type: type[BaseModel]) -> list:
    if not path.exists():
        return []
    return [
        model_type.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_artifacts(
    output: Path,
    config: BatchConfig,
    questions: list[PendingQuestion],
    attempts: list[AttemptAudit],
    summary: BatchSummary,
    references: dict[str, list[ReferenceProvenance]],
    generated_at: datetime,
    git_commit: str,
    model_revision: str,
    intents: list[QuestionIntent],
    status: Literal["complete", "incomplete"],
    error: str | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        **config.model_dump(mode="json"),
        "model_revision": model_revision,
        "git_commit": git_commit,
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "status": status,
        "error": error,
        "artifacts": {
            "questions": "pending_questions.jsonl",
            "audit": "audit.jsonl",
            "summary": "summary.json",
        },
        "references": [
            record.model_dump(mode="json")
            for skill_id in config.skill_ids
            for record in references[skill_id]
        ],
        "intents": [intent.model_dump(mode="json") for intent in intents],
        "accepted_intent_ids": [question.intent_id for question in questions],
        "grounding_briefs": {
            skill_id: {
                "version": grounding_brief(skill_id).version,
                "content_hash": grounding_brief(skill_id).content_hash,
            }
            for skill_id in config.skill_ids
        },
        "slot_intent_ids": {
            skill_id: [
                intents[index % len(intents)].intent_id
                for index in range(slot_count(config, intents))
            ]
            for skill_id, intents in {
                skill_id: [intent for intent in intents if intent.skill_id == skill_id]
                for skill_id in config.skill_ids
            }.items()
        },
    }
    write_jsonl(output / "pending_questions.jsonl", questions)
    write_jsonl(output / "audit.jsonl", attempts)
    write_json(output / "manifest.json", manifest)
    write_json(output / "summary.json", summary.model_dump(mode="json"))


def generate_batch(
    config: BatchConfig,
    model: BatchModel,
    output: Path,
    *,
    skills_path: Path,
    references_path: Path,
    provenance_path: Path,
    clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    git_commit: str | None = None,
    resume: bool = False,
) -> BatchResult:
    catalogue = load_skills(skills_path, references_path)
    known_skill_ids = {skill.skill_id for skill in catalogue.skills}
    provenance = load_reference_provenance(provenance_path, known_skill_ids)
    skills, references = validate_inputs(
        config, model, catalogue.skills, provenance
    )
    validate_blueprint_config(config)
    blueprint = load_blueprint_for_batch(config.batch_id)
    if blueprint.prompt_version != config.prompt_version:
        raise BatchGenerationError(
            "batch config prompt version does not match its intent blueprint"
        )
    blueprint_skills = intents_by_skill(blueprint)
    selected_intents: list[QuestionIntent] = []
    reference_by_id = {
        record.reference_id: record
        for records in references.values()
        for record in records
    }
    for skill_id in config.skill_ids:
        pool = blueprint_skills.get(skill_id, [])
        if not pool:
            raise BatchGenerationError(f"{skill_id} has no reviewed question intents")
        requested = slot_count(config, pool)
        if requested > len(pool) and not config.allow_intent_reuse:
            raise BatchGenerationError(
                f"{skill_id} requests {requested} slots but has only "
                f"{len(pool)} distinct intents"
            )
        for intent in pool:
            if config.difficulty == "mixed" and intent.difficulty is None:
                raise BatchGenerationError(
                    f"{intent.intent_id} needs an explicit difficulty for a mixed batch"
                )
            if (
                config.difficulty != "mixed"
                and intent.difficulty is not None
                and intent.difficulty != config.difficulty
            ):
                raise BatchGenerationError(
                    f"{intent.intent_id} requires {intent.difficulty} difficulty"
                )
            if (
                intent.learning_objective is not None
                and intent.learning_objective != skills[skill_id].learning_objective
            ):
                raise BatchGenerationError(
                    f"{intent.intent_id} learning objective is not canonical"
                )
            missing = [
                reference_id
                for reference_id in intent.preferred_reference_ids
                if reference_id not in reference_by_id
                or reference_by_id[reference_id].skill_id != skill_id
            ]
            if missing:
                raise BatchGenerationError(
                    f"{intent.intent_id} has invalid preferred references: {', '.join(missing)}"
                )
            brief_reference_ids = grounding_brief(skill_id).intent_reference_ids.get(
                intent.intent_id
            )
            if (
                brief_reference_ids is not None
                and brief_reference_ids != intent.preferred_reference_ids
            ):
                raise BatchGenerationError(
                    f"{intent.intent_id} preferred references do not match its grounding brief"
                )
        selected_intents.extend(pool)

    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    if manifest_path.exists() and not resume:
        raise BatchGenerationError(
            f"{output} already contains a batch; pass resume=True to continue it"
        )

    commit = git_commit or current_git_commit()
    started_at = clock()
    questions: list[PendingQuestion] = []
    attempts: list[AttemptAudit] = []
    if resume:
        if not manifest_path.exists():
            raise BatchGenerationError("cannot resume: manifest.json does not exist")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest.get("status") == "complete":
            raise BatchGenerationError("a completed generated batch is immutable")
        for field, expected in config.model_dump(mode="json").items():
            if existing_manifest.get(field) != expected:
                raise BatchGenerationError(
                    f"cannot resume: config field {field} does not match the manifest"
                )
        if existing_manifest.get("model_revision") != model.model_revision:
            raise BatchGenerationError("cannot resume with a different model revision")
        questions = read_jsonl(output / "pending_questions.jsonl", PendingQuestion)
        attempts = read_jsonl(output / "audit.jsonl", AttemptAudit)
        started_at = datetime.fromisoformat(
            existing_manifest["generated_at"].replace("Z", "+00:00")
        )
        commit = existing_manifest["git_commit"]

    seen_questions = {normalise_question(item.question.question) for item in questions}
    accepted_stems = [item.question.question for item in questions]
    accepted_intent_ids = {item.intent_id for item in questions}
    accepted_slots = {(item.skill_id, item.question_index) for item in questions}
    intent_lookup = {intent.intent_id: intent for intent in selected_intents}
    accepted_component_keys: set[frozenset[str]] = set()
    for item in questions:
        key = semantic_component_list_key(item.question, intent_lookup[item.intent_id])
        if key is not None:
            accepted_component_keys.add(key)
    failure: str | None = None

    def summary() -> BatchSummary:
        return BatchSummary(
            requested=sum(
                slot_count(config, blueprint_skills[skill_id])
                for skill_id in config.skill_ids
            ),
            accepted=len(questions),
            rejected=sum(a.validation_status == "invalid" for a in attempts),
            duplicated=sum(a.validation_status == "duplicate" for a in attempts),
        )

    def persist(error: str | None = None, complete: bool = False) -> None:
        write_artifacts(
            output,
            config,
            questions,
            attempts,
            summary(),
            references,
            started_at,
            commit,
            model.model_revision,
            selected_intents,
            "complete" if complete else "incomplete",
            error,
        )

    persist()

    for skill_id in config.skill_ids:
        skill = skills[skill_id]
        pool = blueprint_skills[skill_id]

        for question_index in range(slot_count(config, pool)):
            if (skill_id, question_index) in accepted_slots:
                continue
            intent = pool[question_index % len(pool)]
            if intent.intent_id in accepted_intent_ids and not config.allow_intent_reuse:
                failure = f"accepted intent {intent.intent_id} cannot be reused"
                persist(failure)
                break
            source_records = [
                reference_by_id[reference_id]
                for reference_id in intent.preferred_reference_ids
            ]
            reference_ids = [record.reference_id for record in source_records]
            requested_difficulty = (
                intent.difficulty if config.difficulty == "mixed" else config.difficulty
            )
            if requested_difficulty is None:
                raise BatchGenerationError(
                    f"{intent.intent_id} has no generation difficulty"
                )
            request = QuizGenerationRequest(
                topic=skill.name,
                difficulty=requested_difficulty,
                learning_objective=skill.learning_objective,
                question_count=1,
                reference_material=[
                    record.reference_material for record in source_records
                ],
            )
            accepted = False
            prior_attempts = [
                attempt
                for attempt in attempts
                if attempt.skill_id == skill_id
                and attempt.question_index == question_index
            ]
            next_attempt = (
                max(attempt.attempt_index for attempt in prior_attempts) + 1
                if prior_attempts
                else 0
            )
            previous_status = prior_attempts[-1].validation_status if prior_attempts else None
            previous_error = prior_attempts[-1].validation_error if prior_attempts else None

            for attempt_index in range(
                next_attempt, next_attempt + config.max_attempts_per_question
            ):
                seed = derive_seed(
                    config.batch_id,
                    skill_id,
                    question_index,
                    attempt_index,
                    config.base_seed,
                )
                generated_at = clock()
                raw_response: str | None = None
                parsed: IntentQuestion | None = None
                status: AttemptStatus
                validation_error: str | None = None
                feedback = (
                    corrective_prompt(previous_status, previous_error)
                    if previous_status is not None and previous_error is not None
                    else None
                )
                messages = build_grounded_quiz_messages(
                    request,
                    question_intent=intent,
                    grounding_brief=grounding_brief(skill_id),
                    avoid_stems=accepted_stems,
                    corrective_feedback=feedback,
                )
                hashed_prompt = prompt_hash(messages)

                try:
                    raw_response = model.generate(
                        messages, seed, config.generation_parameters
                    )
                    parsed = parse_raw_question(raw_response, intent.intent_id)
                    validate_question(parsed)
                    if parsed.difficulty != requested_difficulty:
                        raise ValueError(
                            "question difficulty does not match the assigned intent"
                        )
                    BankItem(
                        item_id=question_id(
                            config.batch_id, skill_id, question_index
                        ),
                        skill_id=skill_id,
                        provenance="generated",
                        question=parsed,
                    )
                    normalised = normalise_question(parsed.question)
                    if normalised in seen_questions:
                        status = "duplicate"
                        validation_error = "duplicate normalized question text"
                    else:
                        component_key = validate_pilot_question(
                            parsed, intent, accepted_component_keys
                        )
                        quality_issues = generic_quality_issues(
                            parsed,
                            intent_id=intent.intent_id,
                            accepted_intent_ids=accepted_intent_ids,
                        )
                        if quality_issues:
                            issue = quality_issues[0]
                            raise ValueError(f"quality check {issue.code}: {issue.message}")
                        status = "accepted"
                except Exception as error:
                    validation_error = f"{type(error).__name__}: {error}"
                    status = (
                        "duplicate"
                        if "semantically equivalent component-list" in str(error)
                        else "invalid"
                    )

                attempts.append(
                    AttemptAudit(
                        batch_id=config.batch_id,
                        skill_id=skill_id,
                        intent_id=intent.intent_id,
                        question_index=question_index,
                        attempt_index=attempt_index,
                        seed=seed,
                        reference_ids=reference_ids,
                        prompt_version=config.prompt_version,
                        prompt_hash=hashed_prompt,
                        model_id=config.model_id,
                        model_revision=model.model_revision,
                        generation_parameters=config.generation_parameters,
                        validation_status=status,
                        validation_error=validation_error,
                        generated_at=generated_at,
                        raw_response=raw_response,
                        parsed_question=parsed,
                    )
                )

                if status == "accepted" and parsed is not None and raw_response is not None:
                    seen_questions.add(normalise_question(parsed.question))
                    accepted_stems.append(parsed.question)
                    accepted_intent_ids.add(intent.intent_id)
                    if component_key is not None:
                        accepted_component_keys.add(component_key)
                    questions.append(
                        PendingQuestion(
                            batch_id=config.batch_id,
                            question_id=question_id(
                                config.batch_id, skill_id, question_index
                            ),
                            skill_id=skill_id,
                            question_index=question_index,
                            intent_id=intent.intent_id,
                            seed=seed,
                            reference_ids=reference_ids,
                            prompt_version=config.prompt_version,
                            prompt_hash=hashed_prompt,
                            model_id=config.model_id,
                            model_revision=model.model_revision,
                            generation_parameters=config.generation_parameters,
                            generated_at=generated_at,
                            git_commit=commit,
                            raw_response=raw_response,
                            question=parsed,
                        )
                    )
                    accepted = True
                    persist()
                    break
                previous_status = status
                previous_error = validation_error
                persist()

            if not accepted:
                failure = (
                    f"could not generate {skill_id} question {question_index} "
                    f"within {config.max_attempts_per_question} attempts in this run"
                )
                persist(failure)
                break

        if failure:
            break

    result_status: Literal["complete", "incomplete"] = (
        "incomplete" if failure else "complete"
    )
    if not failure:
        try:
            validate_cold_start_diversity(questions)
        except ValueError as error:
            failure = str(error)
            result_status = "incomplete"
    persist(failure, complete=not failure)
    return BatchResult(
        questions=questions,
        attempts=attempts,
        summary=summary(),
        status=result_status,
    )
