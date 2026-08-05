"""Retrieved reference candidates and their review lifecycle.

A candidate is source material, not a reference. Every candidate is created
pending, and only an explicit approval lets export_reference_material hand its
passage to question generation - so nothing a search provider returns can
reach the model without a person having read it first.
"""

from collections.abc import Iterable
from datetime import datetime, timezone
from hashlib import sha256
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from taxonomy.schemas import SkillId

ReviewStatus = Literal["pending", "approved", "rejected"]

MAX_PASSAGE_CHARS = 1200

# Below this a page has not said enough to ground a question - a navigation
# stub, a redirect notice, an image caption. Quoting it would look like
# grounding without being any.
MIN_PASSAGE_CHARS = 200


class SearchResult(BaseModel):
    """One hit from a search provider, before the page itself is read."""

    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    snippet: str = ""


class ReferenceCandidate(BaseModel):
    candidate_id: str = Field(min_length=1)
    skill_id: SkillId
    title: str = Field(min_length=1)
    source_url: str = Field(min_length=1)
    source_domain: str = Field(min_length=1)
    retrieved_at: datetime
    passage: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)

    review_status: ReviewStatus = "pending"
    reviewed_at: datetime | None = None
    reviewer_id: str | None = None
    review_note: str | None = None

    @model_validator(mode="after")
    def check_review_metadata(self) -> "ReferenceCandidate":
        """A decided candidate names who decided it, and when.

        Approval is what lets a passage reach the model, so an approval with
        nobody's name against it is not a record worth keeping.
        """
        if self.review_status == "pending":
            if self.reviewed_at or self.reviewer_id or self.review_note:
                raise ValueError("a pending candidate carries no review metadata.")

            return self

        if not self.reviewer_id or not self.reviewed_at:
            raise ValueError(
                f"a {self.review_status} candidate needs a reviewer and a review time."
            )

        return self


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def flatten(text: str) -> str:
    """One line of plain text, at whatever length it comes to."""
    printable = "".join(
        character if character.isprintable() else " " for character in text
    )

    return " ".join(printable.split())


def normalise_passage(text: str) -> str:
    """Reduce page text to one plain-text passage.

    This is tidying, not a safety boundary. Collapsing whitespace makes a
    passage comparable and quotable; it does not make the words in it
    trustworthy, and nothing downstream may treat a normalised passage as
    anything other than untrusted source data.
    """
    return flatten(text)[:MAX_PASSAGE_CHARS]


def content_hash(passage: str) -> str:
    """Hash the normalised passage, so the same prose hashes alike."""
    return sha256(normalise_passage(passage).casefold().encode("utf-8")).hexdigest()


def new_candidate(
    skill_id: str,
    title: str,
    source_url: str,
    source_domain: str,
    passage: str,
    retrieved_at: datetime,
) -> ReferenceCandidate:
    """Build a pending candidate. This is the only way retrieval makes one."""
    body = normalise_passage(passage)
    digest = content_hash(body)

    return ReferenceCandidate(
        candidate_id=f"{skill_id}-{digest[:12]}",
        skill_id=skill_id,
        title=title,
        source_url=source_url,
        source_domain=source_domain,
        retrieved_at=retrieved_at,
        passage=body,
        content_hash=digest,
        review_status="pending",
    )


def reviewed(
    candidate: ReferenceCandidate,
    status: ReviewStatus,
    reviewer_id: str,
    note: str | None = None,
    reviewed_at: datetime | None = None,
) -> ReferenceCandidate:
    if not reviewer_id.strip():
        raise ValueError("a review decision needs a reviewer.")

    return ReferenceCandidate.model_validate(
        candidate.model_dump()
        | {
            "review_status": status,
            "reviewer_id": reviewer_id.strip(),
            "review_note": note,
            "reviewed_at": reviewed_at or utc_now(),
        }
    )


def approve(
    candidate: ReferenceCandidate,
    reviewer_id: str,
    note: str | None = None,
    reviewed_at: datetime | None = None,
) -> ReferenceCandidate:
    return reviewed(candidate, "approved", reviewer_id, note, reviewed_at)


def reject(
    candidate: ReferenceCandidate,
    reviewer_id: str,
    note: str | None = None,
    reviewed_at: datetime | None = None,
) -> ReferenceCandidate:
    return reviewed(candidate, "rejected", reviewer_id, note, reviewed_at)


def export_reference_material(
    candidates: Iterable[ReferenceCandidate],
) -> dict[str, list[str]]:
    """Approved candidates only, as the reference_material lists skills carry.

    Pending and rejected candidates are not merely omitted from the output -
    they have no route into it, which is what keeps an unreviewed passage out
    of a generation request.
    """
    exported: dict[str, list[str]] = {}

    for candidate in candidates:
        if candidate.review_status != "approved":
            continue

        passages = exported.setdefault(candidate.skill_id, [])

        if candidate.passage not in passages:
            passages.append(candidate.passage)

    return exported
