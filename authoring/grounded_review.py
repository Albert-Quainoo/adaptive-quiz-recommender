"""Human review and revision workflow for immutable grounded source batches."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from api.bank import BankItem
from api.schemas import QuizQuestion
from authoring.grounded_batch import PendingQuestion
from authoring.question_quality import QualityIssue, generic_quality_issues


FinalReviewStatus = Literal["pending", "approved", "rejected"]
Recommendation = Literal["propose_revision", "reject"]


class QuestionRevision(BaseModel):
    original_question_id: str = Field(min_length=1)
    revision_id: str = Field(min_length=1)
    editor: str = Field(min_length=1)
    edited_at: datetime
    changed_fields: list[str] = Field(min_length=1)
    review_note: str = Field(min_length=1)
    source_batch_id: str | None = None
    intent_id: str | None = None
    skill_id: str | None = None
    reference_ids: list[str] = Field(default_factory=list)
    model_id: str | None = None
    model_revision: str | None = None
    prompt_version: str | None = None
    prompt_hash: str | None = None
    final_review_status: FinalReviewStatus = "pending"
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    rejection_reason: str | None = None
    question: QuizQuestion

    @model_validator(mode="after")
    def decision_metadata_matches_status(self) -> "QuestionRevision":
        decided = self.final_review_status != "pending"
        if decided != bool(self.reviewed_by and self.reviewed_at):
            raise ValueError("a decided revision needs reviewer and review time")
        if self.final_review_status == "rejected" and not self.rejection_reason:
            raise ValueError("a rejected revision needs a reason")
        return self


class RevisionProvenance(BaseModel):
    source_batch_id: str = Field(min_length=1)
    intent_id: str = Field(min_length=1)
    skill_id: str = Field(min_length=1)
    reference_ids: list[str] = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_revision: str = Field(min_length=1)
    prompt_version: str = Field(min_length=1)
    prompt_hash: str = Field(min_length=1)

    @classmethod
    def from_source(cls, source: PendingQuestion) -> "RevisionProvenance":
        return cls(
            source_batch_id=source.batch_id,
            intent_id=source.intent_id,
            skill_id=source.skill_id,
            reference_ids=source.reference_ids,
            model_id=source.model_id,
            model_revision=source.model_revision,
            prompt_version=source.prompt_version,
            prompt_hash=source.prompt_hash,
        )


class CurationItem(BaseModel):
    original_question_id: str = Field(min_length=1)
    skill_id: str = Field(min_length=1)
    intent_id: str = Field(min_length=1)
    recommendation: Recommendation
    recommendation_reason: str = Field(min_length=1)
    final_review_status: FinalReviewStatus = "pending"
    reviewed_by: str | None = None
    reviewed_at: datetime | None = None
    rejection_reason: str | None = None
    revisions: list[QuestionRevision] = Field(default_factory=list)

    @model_validator(mode="after")
    def decision_metadata_matches_status(self) -> "CurationItem":
        decided = self.final_review_status != "pending"
        if decided != bool(self.reviewed_by and self.reviewed_at):
            raise ValueError("a decided item needs reviewer and review time")
        if self.final_review_status == "rejected" and not self.rejection_reason:
            raise ValueError("a rejected item needs a reason")
        if self.final_review_status == "approved":
            approved = [
                revision
                for revision in self.revisions
                if revision.final_review_status == "approved"
            ]
            if len(approved) != 1:
                raise ValueError("an approved item needs exactly one approved revision")
        return self


class GroundedReview(BaseModel):
    batch_id: str = Field(min_length=1)
    source_hashes: dict[str, str]
    items: list[CurationItem]

    @model_validator(mode="after")
    def original_and_intent_ids_are_unique(self) -> "GroundedReview":
        question_ids = [item.original_question_id for item in self.items]
        intent_ids = [item.intent_id for item in self.items]
        if len(question_ids) != len(set(question_ids)):
            raise ValueError("original question ids must be unique")
        if len(intent_ids) != len(set(intent_ids)):
            raise ValueError("intent ids must be unique")
        return self


class QuestionInspection(BaseModel):
    source_question: PendingQuestion
    references: list[dict]
    attempts: list[dict]
    curation: CurationItem
    quality_issues: list[QualityIssue]


class ApprovedItemProvenance(BaseModel):
    item_id: str
    source_batch_id: str
    original_question_id: str
    revision_id: str
    intent_id: str
    skill_id: str
    reference_ids: list[str]
    model_id: str
    model_revision: str
    prompt_version: str
    prompt_hash: str
    reviewer_id: str
    reviewed_at: datetime
    review_note: str
    changed_fields: list[str]


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(batch_path: Path) -> dict[str, str]:
    return {
        name: file_hash(batch_path / name)
        for name in ("audit.jsonl", "manifest.json", "pending_questions.jsonl", "summary.json")
    }


def assert_immutable_source(batch_path: Path, review: GroundedReview) -> None:
    actual = source_hashes(batch_path)
    if actual != review.source_hashes:
        changed = sorted(set(actual) | set(review.source_hashes))
        raise ValueError(
            "immutable generated source batch hash mismatch: " + ", ".join(changed)
        )


def load_source_questions(batch_path: Path) -> list[PendingQuestion]:
    return [
        PendingQuestion.model_validate_json(line)
        for line in (batch_path / "pending_questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


class GroundedReviewStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> GroundedReview:
        return GroundedReview.model_validate_json(self.path.read_text(encoding="utf-8"))

    def save(self, review: GroundedReview) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(review.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(self.path)

    def replace_item(self, replacement: CurationItem) -> GroundedReview:
        review = self.load()
        if replacement.original_question_id not in {
            item.original_question_id for item in review.items
        }:
            raise KeyError(replacement.original_question_id)
        updated = review.model_copy(
            update={
                "items": [
                    replacement
                    if item.original_question_id == replacement.original_question_id
                    else item
                    for item in review.items
                ]
            }
        )
        self.save(updated)
        return updated


def get_item(review: GroundedReview, question_id: str) -> CurationItem:
    for item in review.items:
        if item.original_question_id == question_id:
            return item
    raise KeyError(question_id)


def list_pending(review: GroundedReview) -> list[CurationItem]:
    return [item for item in review.items if item.final_review_status == "pending"]


def inspect_question(
    batch_path: Path, review: GroundedReview, question_id: str
) -> QuestionInspection:
    assert_immutable_source(batch_path, review)
    source = next(
        (item for item in load_source_questions(batch_path) if item.question_id == question_id),
        None,
    )
    if source is None:
        raise KeyError(question_id)
    manifest = json.loads((batch_path / "manifest.json").read_text(encoding="utf-8"))
    attempts = [
        json.loads(line)
        for line in (batch_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return QuestionInspection(
        source_question=source,
        references=[
            record
            for record in manifest["references"]
            if record["reference_id"] in source.reference_ids
        ],
        attempts=[
            attempt
            for attempt in attempts
            if attempt["skill_id"] == source.skill_id
            and attempt["question_index"] == source.question_index
        ],
        curation=get_item(review, question_id),
        quality_issues=generic_quality_issues(
            source.question,
            intent_id=source.intent_id,
        ),
    )


def propose_revision(
    item: CurationItem,
    original: QuizQuestion,
    edited: QuizQuestion,
    editor: str,
    review_note: str,
    *,
    edited_at: datetime | None = None,
    provenance: RevisionProvenance | None = None,
) -> CurationItem:
    if item.final_review_status != "pending":
        raise ValueError("only a pending item can receive a proposed revision")
    editor = editor.strip()
    review_note = review_note.strip()
    if not editor or not review_note:
        raise ValueError("editor and review note are required")
    original_fields = QuizQuestion.model_validate(original.model_dump()).model_dump()
    edited_fields = edited.model_dump()
    changed_fields = sorted(
        field for field in original_fields if original_fields[field] != edited_fields[field]
    )
    if not changed_fields:
        raise ValueError("a proposed revision must change at least one question field")
    timestamp = edited_at or datetime.now(timezone.utc)
    identity = json.dumps(
        {
            "original_question_id": item.original_question_id,
            "editor": editor,
            "edited_at": timestamp.isoformat(),
            "question": edited_fields,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    revision = QuestionRevision(
        original_question_id=item.original_question_id,
        revision_id=f"{item.original_question_id}-rev-{hashlib.sha256(identity.encode()).hexdigest()[:12]}",
        editor=editor,
        edited_at=timestamp,
        changed_fields=changed_fields,
        review_note=review_note,
        **(provenance.model_dump() if provenance else {}),
        question=edited,
    )
    return item.model_copy(update={"revisions": [*item.revisions, revision]})


def approve_revision(
    item: CurationItem,
    revision_id: str,
    reviewer: str,
    *,
    reviewed_at: datetime | None = None,
) -> CurationItem:
    if item.final_review_status != "pending":
        raise ValueError("only a pending item can be approved")
    reviewer = reviewer.strip()
    if not reviewer:
        raise ValueError("reviewer is required")
    if revision_id not in {revision.revision_id for revision in item.revisions}:
        raise KeyError(revision_id)
    selected = next(
        revision for revision in item.revisions if revision.revision_id == revision_id
    )
    if not all(
        (
            selected.source_batch_id,
            selected.intent_id,
            selected.skill_id,
            selected.reference_ids,
            selected.model_id,
            selected.model_revision,
            selected.prompt_version,
            selected.prompt_hash,
        )
    ):
        raise ValueError("an approved revision needs complete source provenance")
    timestamp = reviewed_at or datetime.now(timezone.utc)
    revisions = [
        revision.model_copy(
            update={
                "final_review_status": "approved",
                "reviewed_by": reviewer,
                "reviewed_at": timestamp,
            }
        )
        if revision.revision_id == revision_id
        else QuestionRevision.model_validate(
            revision.model_dump()
            | {
                "final_review_status": "rejected",
                "reviewed_by": reviewer,
                "reviewed_at": timestamp,
                "rejection_reason": f"Superseded by approved revision {revision_id}.",
            }
        )
        if revision.final_review_status == "pending"
        else revision
        for revision in item.revisions
    ]
    return CurationItem.model_validate(
        item.model_dump()
        | {
            "final_review_status": "approved",
            "reviewed_by": reviewer,
            "reviewed_at": timestamp,
            "revisions": revisions,
        }
    )


def reject_item(
    item: CurationItem,
    reviewer: str,
    reason: str,
    *,
    reviewed_at: datetime | None = None,
) -> CurationItem:
    if item.final_review_status != "pending":
        raise ValueError("only a pending item can be rejected")
    reviewer = reviewer.strip()
    reason = reason.strip()
    if not reviewer or not reason:
        raise ValueError("reviewer and rejection reason are required")
    timestamp = reviewed_at or datetime.now(timezone.utc)
    revisions = [
        QuestionRevision.model_validate(
            revision.model_dump()
            | {
                "final_review_status": "rejected",
                "reviewed_by": reviewer,
                "reviewed_at": timestamp,
                "rejection_reason": reason,
            }
        )
        if revision.final_review_status == "pending"
        else revision
        for revision in item.revisions
    ]
    return CurationItem.model_validate(
        item.model_dump()
        | {
            "final_review_status": "rejected",
            "reviewed_by": reviewer,
            "reviewed_at": timestamp,
            "rejection_reason": reason,
            "revisions": revisions,
        }
    )


def export_approved_bank_items(review: GroundedReview) -> list[BankItem]:
    exported = []
    for item in sorted(review.items, key=lambda value: value.original_question_id):
        if item.final_review_status != "approved":
            continue
        revision = next(
            revision
            for revision in item.revisions
            if revision.final_review_status == "approved"
        )
        exported.append(
            BankItem(
                item_id=item.original_question_id,
                question=revision.question,
                provenance="generated",
                skill_id=item.skill_id,
            )
        )
    return exported


def approved_item_provenance(
    review: GroundedReview, item_id: str
) -> ApprovedItemProvenance:
    item = get_item(review, item_id)
    if item.final_review_status != "approved":
        raise ValueError(f"{item_id} is not approved")
    revision = next(
        revision
        for revision in item.revisions
        if revision.final_review_status == "approved"
    )
    values = {
        "source_batch_id": revision.source_batch_id,
        "intent_id": revision.intent_id,
        "skill_id": revision.skill_id,
        "model_id": revision.model_id,
        "model_revision": revision.model_revision,
        "prompt_version": revision.prompt_version,
        "prompt_hash": revision.prompt_hash,
        "reviewer_id": revision.reviewed_by,
        "reviewed_at": revision.reviewed_at,
    }
    missing = [name for name, value in values.items() if not value]
    if not revision.reference_ids:
        missing.append("reference_ids")
    if missing:
        raise ValueError("approved provenance is incomplete: " + ", ".join(missing))
    return ApprovedItemProvenance(
        item_id=item.original_question_id,
        original_question_id=item.original_question_id,
        revision_id=revision.revision_id,
        reference_ids=revision.reference_ids,
        review_note=revision.review_note,
        changed_fields=revision.changed_fields,
        **values,
    )


def write_approved_bank(path: Path, review: GroundedReview) -> None:
    items = export_approved_bank_items(review)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
            for item in items
        ),
        encoding="utf-8",
    )
    temporary.replace(path)
