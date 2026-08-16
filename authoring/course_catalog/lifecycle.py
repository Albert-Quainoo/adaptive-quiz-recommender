"""Single choke point for every course-manifest status write and every
CourseApprovalRecord append.

Routing every transition through this module (rather than letting callers
write manifest.status directly) is what makes "one immutable audit record
per status change" true by construction: nothing outside this file ever
calls repository.append() or mutates a CourseManifest's status.
"""

from collections.abc import Callable
from datetime import datetime, timezone

from authoring.course_catalog.readiness import ReadinessReport
from authoring.course_catalog.records import CourseApprovalRecord
from authoring.course_catalog.repository import SQLiteCourseApprovalRepository
from authoring.replenishment.manifest import CourseManifest

Clock = Callable[[], datetime]


class LifecycleError(ValueError):
    """An illegal transition was attempted. Callers turn this into a clean,
    non-zero CLI exit -- never a crash, never a partial state change."""


def _default_clock() -> datetime:
    return datetime.now(timezone.utc)


def approve_course_definition(
    manifest: CourseManifest,
    repository: SQLiteCourseApprovalRepository,
    *,
    approver: str,
    notes: str | None = None,
    auto_activate_when_ready: bool = True,
    clock: Clock = _default_clock,
) -> tuple[CourseManifest, CourseApprovalRecord]:
    """proposed -> approved_for_preparation. The one manual admin gate on the
    happy path: approving the course *definition*, before preparation
    starts -- not a per-question or final-activation approval."""

    if manifest.status != "proposed":
        raise LifecycleError(
            f"{manifest.course_id} is {manifest.status!r}; only a 'proposed' "
            "course can be approved for preparation"
        )
    updated = manifest.model_copy(
        update={
            "status": "approved_for_preparation",
            "auto_activate_when_ready": auto_activate_when_ready,
        }
    )
    record = repository.append(
        course_id=manifest.course_id,
        lifecycle_status="approved_for_preparation",
        course_profile_version=manifest.version,
        decision="approved",
        approver_identity=approver,
        decided_at=clock(),
        auto_activate_when_ready=auto_activate_when_ready,
        notes=notes,
    )
    return updated, record


def reject_course_definition(
    manifest: CourseManifest,
    repository: SQLiteCourseApprovalRepository,
    *,
    approver: str,
    reason: str,
    clock: Clock = _default_clock,
) -> CourseApprovalRecord:
    """Records a rejection without changing manifest.status -- the course
    stays 'proposed', eligible for resubmission once fixed. There is no
    'rejected' lifecycle state in the spec; the rejection itself is what's
    audited, via the approval record's decision field."""

    if manifest.status != "proposed":
        raise LifecycleError(
            f"{manifest.course_id} is {manifest.status!r}; only a 'proposed' "
            "course definition can be rejected"
        )
    return repository.append(
        course_id=manifest.course_id,
        lifecycle_status=manifest.status,
        course_profile_version=manifest.version,
        decision="rejected",
        approver_identity=approver,
        decided_at=clock(),
        auto_activate_when_ready=manifest.auto_activate_when_ready,
        notes=reason,
    )


def begin_preparation(
    manifest: CourseManifest,
    repository: SQLiteCourseApprovalRepository,
    *,
    approver: str,
    notes: str | None = None,
    clock: Clock = _default_clock,
) -> tuple[CourseManifest, CourseApprovalRecord]:
    """approved_for_preparation -> preparing. Marks that content authoring
    (taxonomy, references, blueprints, generation, review) is actively under
    way for this course -- distinct from the one-time definition approval
    that preceded it."""

    if manifest.status != "approved_for_preparation":
        raise LifecycleError(
            f"{manifest.course_id} is {manifest.status!r}; only an "
            "'approved_for_preparation' course can begin preparation"
        )
    updated = manifest.model_copy(update={"status": "preparing"})
    record = repository.append(
        course_id=manifest.course_id,
        lifecycle_status="preparing",
        course_profile_version=manifest.version,
        decision="preparation_started",
        approver_identity=approver,
        decided_at=clock(),
        auto_activate_when_ready=manifest.auto_activate_when_ready,
        notes=notes,
    )
    return updated, record


def advance_to_content_approval(
    manifest: CourseManifest,
    repository: SQLiteCourseApprovalRepository,
    *,
    approver: str,
    notes: str | None = None,
    clock: Clock = _default_clock,
) -> tuple[CourseManifest, CourseApprovalRecord]:
    """preparing -> awaiting_content_approval. Marks that this course's
    content preparation (taxonomy, references, generation, review) is
    complete and the course is parked for a human's consolidated
    content-approval decision -- readiness/auto-activation
    (advance_to_ready_and_activate) is a separate, later step this
    function never triggers."""

    if manifest.status != "preparing":
        raise LifecycleError(
            f"{manifest.course_id} is {manifest.status!r}; only a "
            "'preparing' course can advance to awaiting_content_approval"
        )
    updated = manifest.model_copy(update={"status": "awaiting_content_approval"})
    record = repository.append(
        course_id=manifest.course_id,
        lifecycle_status="awaiting_content_approval",
        course_profile_version=manifest.version,
        decision="content_prepared",
        approver_identity=approver,
        decided_at=clock(),
        auto_activate_when_ready=manifest.auto_activate_when_ready,
        notes=notes,
    )
    return updated, record


def advance_to_ready_and_activate(
    manifest: CourseManifest,
    readiness: ReadinessReport,
    repository: SQLiteCourseApprovalRepository,
    *,
    approver: str,
    clock: Clock = _default_clock,
) -> tuple[CourseManifest, CourseApprovalRecord] | None:
    """awaiting_content_approval -> ready -> active, in one recorded step.

    A no-op (returns None, no mutation, no new record) whenever the course
    isn't in the eligible status, isn't ready, or has opted out of
    auto-activation -- the only caller (authoring/course_catalog/cli.py's
    activate-course, an explicit, --confirm-gated admin command) can call
    this without side effects on a not-yet-ready course. approver is the
    real identity requesting activation, recorded on the audit record --
    there is no more silent/automatic activation path in this codebase
    (inspect-course-readiness, the one that used to trigger this
    unattended, is now strictly read-only)."""

    if manifest.status != "awaiting_content_approval":
        return None
    if not readiness.is_ready:
        return None
    if not manifest.auto_activate_when_ready:
        return None

    updated = manifest.model_copy(update={"status": "active"})
    record = repository.append(
        course_id=manifest.course_id,
        lifecycle_status="active",
        course_profile_version=manifest.version,
        decision="activated",
        approver_identity=approver,
        decided_at=clock(),
        auto_activate_when_ready=True,
        taxonomy_version=readiness.taxonomy_version,
        approved_bank_version=readiness.approved_bank_version,
        bkt_model_version=readiness.bkt_model_version,
        notes="readiness checks passed; awaiting_content_approval -> ready -> active",
    )
    return updated, record


def archive_course(
    manifest: CourseManifest,
    repository: SQLiteCourseApprovalRepository,
    *,
    approver: str,
    notes: str | None = None,
    clock: Clock = _default_clock,
) -> tuple[CourseManifest, CourseApprovalRecord]:
    """active -> archived. Removes the course from load_active_manifests()
    (new learner sessions) without touching any already-recorded attempt or
    mastery data, which lives keyed by skill_id, not by manifest status."""

    if manifest.status != "active":
        raise LifecycleError(
            f"{manifest.course_id} is {manifest.status!r}; only an 'active' "
            "course can be archived"
        )
    updated = manifest.model_copy(update={"status": "archived"})
    record = repository.append(
        course_id=manifest.course_id,
        lifecycle_status="archived",
        course_profile_version=manifest.version,
        decision="archived",
        approver_identity=approver,
        decided_at=clock(),
        auto_activate_when_ready=manifest.auto_activate_when_ready,
        notes=notes,
    )
    return updated, record


def seed_initial_record(
    manifest: CourseManifest,
    repository: SQLiteCourseApprovalRepository,
    *,
    decision: str,
    approver: str,
    notes: str | None = None,
    clock: Clock = _default_clock,
) -> CourseApprovalRecord:
    """One-time catalog-population helper, used only by
    scripts/seed_course_catalog.py to write each course's first audit
    record reflecting the spec-mandated starting state. Not exposed via the
    general CLI: it does not enforce a starting-status precondition the way
    the transition functions above do, because "already active on day one"
    (intro-ai) has no prior state to transition from."""

    return repository.append(
        course_id=manifest.course_id,
        lifecycle_status=manifest.status,
        course_profile_version=manifest.version,
        decision=decision,
        approver_identity=approver,
        decided_at=clock(),
        auto_activate_when_ready=manifest.auto_activate_when_ready,
        notes=notes,
    )
