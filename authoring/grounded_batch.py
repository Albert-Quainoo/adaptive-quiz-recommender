"""Reproducible, review-first generation of grounded question batches."""

import hashlib
import json
import subprocess
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, Field, field_validator

from api.prompt_builder import PROMPT_VERSION, build_quiz_messages
from api.response_parser import parse_quiz_messages
from api.schemas import QuizGenerationRequest, QuizQuestion, difficulty_level
from taxonomy.loader import load_reference_provenance, load_skills
from taxonomy.schemas import ReferenceProvenance, SkillDefinition

GenerationValue = bool | int | float | str
AttemptStatus = Literal["accepted", "invalid", "duplicate"]


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
    questions_per_skill: int = Field(ge=1)
    base_seed: int = Field(ge=0)
    model_id: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    difficulty: difficulty_level = "intermediate"
    max_attempts_per_question: int = Field(default=3, ge=1, le=20)
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


class PendingQuestion(BaseModel):
    batch_id: str = Field(min_length=1)
    question_id: str = Field(min_length=1)
    skill_id: str
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
    question: QuizQuestion


class AttemptAudit(BaseModel):
    batch_id: str = Field(min_length=1)
    skill_id: str
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
    parsed_question: QuizQuestion | None = None


class BatchSummary(BaseModel):
    requested: int
    accepted: int
    rejected: int
    duplicated: int


class BatchResult(BaseModel):
    questions: list[PendingQuestion]
    attempts: list[AttemptAudit]
    summary: BatchSummary


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


def parse_raw_question(raw_response: str) -> QuizQuestion:
    response = parse_quiz_messages(raw_response)
    if len(response.questions) != 1:
        raise ValueError(
            f"expected exactly one question, received {len(response.questions)}"
        )

    return QuizQuestion.model_validate(response.questions[0].model_dump())


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


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


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
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


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
    error: str | None = None,
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        **config.model_dump(mode="json"),
        "model_revision": model_revision,
        "git_commit": git_commit,
        "generated_at": generated_at.isoformat().replace("+00:00", "Z"),
        "status": "failed" if error else "complete",
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
) -> BatchResult:
    catalogue = load_skills(skills_path, references_path)
    known_skill_ids = {skill.skill_id for skill in catalogue.skills}
    provenance = load_reference_provenance(provenance_path, known_skill_ids)
    skills, references = validate_inputs(
        config, model, catalogue.skills, provenance
    )
    commit = git_commit or current_git_commit()
    started_at = clock()
    questions: list[PendingQuestion] = []
    attempts: list[AttemptAudit] = []
    seen_questions: set[str] = set()
    rejected = duplicated = 0
    failure: str | None = None

    for skill_id in config.skill_ids:
        skill = skills[skill_id]
        source_records = references[skill_id]
        reference_ids = [record.reference_id for record in source_records]
        request = QuizGenerationRequest(
            topic=skill.name,
            difficulty=config.difficulty,
            learning_objective=skill.learning_objective,
            question_count=1,
            reference_material=[record.reference_material for record in source_records],
        )
        messages = build_quiz_messages(request)
        hashed_prompt = prompt_hash(messages)

        for question_index in range(config.questions_per_skill):
            accepted = False

            for attempt_index in range(config.max_attempts_per_question):
                seed = derive_seed(
                    config.batch_id,
                    skill_id,
                    question_index,
                    attempt_index,
                    config.base_seed,
                )
                generated_at = clock()
                raw_response: str | None = None
                parsed: QuizQuestion | None = None
                status: AttemptStatus
                validation_error: str | None = None

                try:
                    raw_response = model.generate(
                        messages, seed, config.generation_parameters
                    )
                    parsed = parse_raw_question(raw_response)
                    validate_question(parsed)
                    normalised = normalise_question(parsed.question)
                    if normalised in seen_questions:
                        status = "duplicate"
                        validation_error = "duplicate normalized question text"
                        duplicated += 1
                    else:
                        status = "accepted"
                        seen_questions.add(normalised)
                except Exception as error:
                    status = "invalid"
                    validation_error = f"{type(error).__name__}: {error}"
                    rejected += 1

                attempts.append(
                    AttemptAudit(
                        batch_id=config.batch_id,
                        skill_id=skill_id,
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
                    questions.append(
                        PendingQuestion(
                            batch_id=config.batch_id,
                            question_id=question_id(
                                config.batch_id, skill_id, question_index
                            ),
                            skill_id=skill_id,
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
                    break

            if not accepted:
                failure = (
                    f"could not generate {skill_id} question {question_index} "
                    f"within {config.max_attempts_per_question} attempts"
                )
                break

        if failure:
            break

    summary = BatchSummary(
        requested=len(config.skill_ids) * config.questions_per_skill,
        accepted=len(questions),
        rejected=rejected,
        duplicated=duplicated,
    )
    write_artifacts(
        output,
        config,
        questions,
        attempts,
        summary,
        references,
        started_at,
        commit,
        model.model_revision,
        failure,
    )

    if failure:
        raise BatchGenerationError(failure)

    return BatchResult(questions=questions, attempts=attempts, summary=summary)
