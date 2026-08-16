"""Immutable audit trail for course-lifecycle status changes.

One CourseApprovalRecord is appended per status change (course-definition
approval/rejection, automatic activation, archiving). Records are never
updated or deleted -- authoring/course_catalog/repository.py's
SQLiteCourseApprovalRepository exposes only append/list/latest, no update
or delete method, matching how bkt/sqlite_repository.py treats
attempt_events as append-only.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

ApprovalDecision = Literal[
    "approved", "rejected", "archived", "activated",
    "preparation_started", "content_prepared",
]


class CourseApprovalRecord(BaseModel):
    record_id: str = Field(min_length=1)
    course_id: str = Field(min_length=1)
    sequence_number: int = Field(gt=0)
    lifecycle_status: str = Field(min_length=1)
    course_profile_version: str = Field(min_length=1)
    taxonomy_version: str | None = None
    approved_bank_version: str | None = None
    bkt_model_version: str | None = None
    decision: ApprovalDecision
    approver_identity: str = Field(min_length=1)
    decided_at: datetime
    notes: str | None = None
    auto_activate_when_ready: bool
