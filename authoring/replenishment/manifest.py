"""Course-neutral manifest describing where one course's content lives.

A manifest is the only place course-specific paths and thresholds are named.
Inventory, policy, the job repository, and the worker take a manifest as an
argument and never hardcode a course, so a later course only needs a new
manifest file, not new code.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

MANIFEST_DIRECTORY = Path(__file__).resolve().parent / "manifests"

ManifestStatus = Literal["active", "inactive"]


class CourseManifest(BaseModel):
    course_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    version: str = Field(min_length=1)
    taxonomy_path: Path
    approved_bank_path: Path
    candidate_store_path: Path
    review_store_path: Path
    allowed_domains: tuple[str, ...] = Field(min_length=1)
    low_supply_threshold: int = Field(gt=0)
    target_supply: int = Field(gt=0)
    default_bkt_model_version: str = Field(min_length=1)
    status: ManifestStatus

    @field_validator("course_id", mode="before")
    @classmethod
    def normalise_course_id(cls, value: str) -> str:
        return value.strip().lower() if isinstance(value, str) else value

    @field_validator("target_supply")
    @classmethod
    def target_at_least_threshold(cls, value: int, info) -> int:
        threshold = info.data.get("low_supply_threshold")
        if threshold is not None and value < threshold:
            raise ValueError("target_supply must be at least low_supply_threshold")
        return value

    def skills_path(self) -> Path:
        return self.taxonomy_path / "skills.csv"

    def references_path(self) -> Path:
        return self.taxonomy_path / "references.csv"

    def provenance_path(self) -> Path:
        return self.taxonomy_path / "reference_provenance.csv"


class ManifestError(ValueError):
    pass


def load_course_manifest(course_id: str) -> CourseManifest:
    path = MANIFEST_DIRECTORY / f"{course_id.strip().lower()}.json"
    if not path.is_file():
        raise ManifestError(f"no course manifest for {course_id!r} at {path}")
    return CourseManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))


def active_bank_pointer_path(manifest: CourseManifest) -> Path:
    return manifest.approved_bank_path.parent / f"{manifest.course_id}-active-bank.json"


def active_bank_path(manifest: CourseManifest) -> Path:
    """The bank actually in force: the latest promoted version if one
    exists, else the manifest's own (never-mutated) approved_bank_path."""
    pointer_path = active_bank_pointer_path(manifest)
    if pointer_path.is_file():
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        return Path(pointer["path"])
    return manifest.approved_bank_path


def load_active_manifests() -> list[CourseManifest]:
    manifests = [
        CourseManifest.model_validate(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(MANIFEST_DIRECTORY.glob("*.json"))
    ]
    return [manifest for manifest in manifests if manifest.status == "active"]
