from typing import Annotated, Literal
from pydantic import BaseModel, Field, field_validator, model_validator

# <course>-<topic>-<number>, e.g. AI-SRC-08. The course and topic codes are
# 2-4 letters so the same format covers courses beyond the AI taxonomy.
SkillId = Annotated[
    str,
    Field(min_length=1, pattern=r"^[A-Z]{2,4}-[A-Z]{2,4}-\d{2}[A-Z]?$"),
]

CognitiveProcess = Literal[
    "remember",
    "understand",
    "apply",
    "analyse",
    "evaluate",
]

COGNITIVE_PROCESS_ALIASES = {
    "analyze": "analyse",
}

GenerationStrategy = Literal[
    "generated",
    "templated",
    "hand_authored",
]

class SkillDefinition(BaseModel):
    skill_id: SkillId
    topic: str = Field(min_length=1)
    subtopic: str = Field(min_length=1)
    name: str = Field(min_length=1)
    learning_objective: str = Field(min_length=1)

    cognitive_process: CognitiveProcess
    generation_strategy: GenerationStrategy
    template_id: str | None = None

    reference_material: list[str] = Field(default_factory=list)
    prerequisite_skill_ids: list[SkillId] = Field(default_factory=list)

    @field_validator("skill_id", mode="before")
    @classmethod
    def normalise_skill_id(cls, value: str) -> str:
        if not isinstance(value, str):
            return value

        return value.strip().upper()

    @field_validator("topic", "subtopic", "name", "learning_objective", mode="before")
    @classmethod
    def strip_text(cls, value: str) -> str:
        if not isinstance(value, str):
            return value

        return value.strip()

    @field_validator("cognitive_process", mode="before")
    @classmethod
    def normalise_cognitive_process(cls, value: str) -> str:
        if not isinstance(value, str):
            return value

        value = value.strip().lower()

        return COGNITIVE_PROCESS_ALIASES.get(value, value)

    @field_validator("generation_strategy", mode="before")
    @classmethod
    def normalise_generation_strategy(cls, value: str) -> str:
        if not isinstance(value, str):
            return value

        return value.strip().lower().replace(" ", "_").replace("-", "_")

    @field_validator("template_id", mode="before")
    @classmethod
    def normalise_template_id(cls, value: str | None) -> str | None:
        if not isinstance(value, str):
            return value

        return value.strip() or None

    @field_validator("prerequisite_skill_ids", mode="before")
    @classmethod
    def normalise_prerequisite_skill_ids(cls, value: list) -> list:
        if not isinstance(value, list):
            return value

        return [
            item.strip().upper() if isinstance(item, str) else item
            for item in value
        ]

    @field_validator("prerequisite_skill_ids")
    @classmethod
    def reject_duplicate_prerequisites(cls, value: list[str]) -> list[str]:
        duplicates = sorted({item for item in value if value.count(item) > 1})

        if duplicates:
            raise ValueError(
                f"Duplicate prerequisite skill ids: {', '.join(duplicates)}"
            )

        return value

    @model_validator(mode="after")
    def check_template_id_matches_strategy(self) -> "SkillDefinition":
        if self.generation_strategy == "templated" and not self.template_id:
            raise ValueError("a templated skill needs a template_id.")

        if self.generation_strategy != "templated" and self.template_id:
            raise ValueError(
                f"a {self.generation_strategy} skill cannot have a template_id."
            )

        return self

    @model_validator(mode="after")
    def reject_self_prerequisite(self) -> "SkillDefinition":
        if self.skill_id in self.prerequisite_skill_ids:
            raise ValueError(
                f"{self.skill_id} cannot be a prerequisite of itself."
            )

        return self


def find_prerequisite_cycle(prerequisites: dict[str, list[str]]) -> list[str] | None:
    visited: set[str] = set()
    path: list[str] = []
    on_path: set[str] = set()

    def visit(skill_id: str) -> list[str] | None:
        if skill_id in on_path:
            return path[path.index(skill_id):] + [skill_id]

        if skill_id in visited:
            return None

        visited.add(skill_id)
        on_path.add(skill_id)
        path.append(skill_id)

        for prerequisite in prerequisites[skill_id]:
            cycle = visit(prerequisite)
            if cycle:
                return cycle

        path.pop()
        on_path.remove(skill_id)

        return None

    for skill_id in prerequisites:
        cycle = visit(skill_id)
        if cycle:
            return cycle

    return None


class SkillCatalogue(BaseModel):
    skills: list[SkillDefinition] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_prerequisite_graph(self) -> "SkillCatalogue":
        prerequisites: dict[str, list[str]] = {}
        duplicates: list[str] = []

        for skill in self.skills:
            if skill.skill_id in prerequisites:
                duplicates.append(skill.skill_id)
            prerequisites[skill.skill_id] = skill.prerequisite_skill_ids

        if duplicates:
            raise ValueError(
                f"Duplicate skill ids: {', '.join(sorted(set(duplicates)))}"
            )

        unknown = sorted({
            f"{skill_id} -> {prerequisite}"
            for skill_id, skill_prerequisites in prerequisites.items()
            for prerequisite in skill_prerequisites
            if prerequisite not in prerequisites
        })

        if unknown:
            raise ValueError(f"Unknown prerequisite skill ids: {', '.join(unknown)}")

        cycle = find_prerequisite_cycle(prerequisites)

        if cycle:
            raise ValueError(f"Prerequisite cycle: {' -> '.join(cycle)}")

        return self


def find_skills_missing_reference_material(catalogue: SkillCatalogue) -> list[str]:
    return [
        skill.skill_id
        for skill in catalogue.skills
        if skill.generation_strategy == "generated" and not skill.reference_material
    ]