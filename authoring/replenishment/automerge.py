"""Post-promotion auto-merge eligibility for the replenishment PR.

Whether every item a completed promote_approved_items job promoted this run
was judged low risk by the automated reviewer, so the PR-merge administrative
step (never item-level curation, which stays 100% human) can skip a human
click.

Deliberately does not decide *whether* to merge on its own -- the caller
(scripts/run_replenishment_cycle.py) additionally requires the run's own
PR-open step to have created a brand-new PR, never merging onto a PR that
predates this run, since that could bundle in a previously-blocked batch's
content a human hasn't cleared yet.
"""

from dataclasses import dataclass, field
from pathlib import Path

from authoring.grounded_review import GroundedReviewStore, load_source_questions, resolved_content_hash
from authoring.replenishment.jobs import ReplenishmentJob
from authoring.replenishment.manifest import CourseManifest
from authoring.replenishment.worker import batch_output_dir
from authoring.review.reports import AutomatedReviewReportStore, review_report_path


@dataclass(frozen=True)
class PromotedItemRisk:
    original_question_id: str
    risk_level: str | None  # None only when no report could be resolved
    low_risk: bool


@dataclass(frozen=True)
class AutoMergeEvaluation:
    job_id: str
    eligible: bool
    reasons: list[str] = field(default_factory=list)
    items: list[PromotedItemRisk] = field(default_factory=list)


def evaluate_promotion_job(job: ReplenishmentJob, manifest: CourseManifest) -> AutoMergeEvaluation:
    """Eligibility for exactly one completed promote_approved_items job.

    Every approved item -- whether approve_as_written or a human-approved
    revision -- must have its own resolved content hash score risk_level
    "low" on the matching AutomatedReviewReport. A human approving a revision
    does not bypass that revision's own automated re-review: the revision is
    still fresh content that automated review scored on its own merits.
    """
    review_path_value = job.metadata.get("review_path")
    if not review_path_value:
        return AutoMergeEvaluation(
            job_id=job.job_id, eligible=False,
            reasons=["job has no recorded review_path"],
        )

    review = GroundedReviewStore(Path(review_path_value)).load()
    output_dir = batch_output_dir(manifest, review.batch_id, job.skill_id)
    source_questions = (
        {question.question_id: question for question in load_source_questions(output_dir)}
        if output_dir.is_dir()
        else {}
    )
    report_store = AutomatedReviewReportStore(
        review_report_path(manifest.review_store_path, review.batch_id, job.skill_id)
    )

    approved_items = [item for item in review.items if item.final_review_status == "approved"]
    if not approved_items:
        return AutoMergeEvaluation(
            job_id=job.job_id, eligible=False,
            reasons=["no approved items in this job's review file"],
        )

    rows: list[PromotedItemRisk] = []
    reasons: list[str] = []
    for item in sorted(approved_items, key=lambda value: value.original_question_id):
        content_hash = resolved_content_hash(item, source_questions)
        report = report_store.latest_for_hash(content_hash) if content_hash else None
        if report is None:
            rows.append(PromotedItemRisk(item.original_question_id, None, low_risk=False))
            reasons.append(f"{item.original_question_id}: no automated review report on file")
            continue
        low_risk = report.risk_level == "low"
        rows.append(PromotedItemRisk(item.original_question_id, report.risk_level, low_risk))
        if not low_risk:
            reasons.append(f"{item.original_question_id}: risk_level={report.risk_level}")

    eligible = all(row.low_risk for row in rows)
    return AutoMergeEvaluation(job_id=job.job_id, eligible=eligible, reasons=reasons, items=rows)


def combine_evaluations(evaluations: list[AutoMergeEvaluation]) -> tuple[bool, list[str]]:
    """Run-level eligibility: True only if this run completed at least one
    promotion job AND every one of them was individually eligible. A run
    with zero promotions is never eligible -- there is nothing to vouch for.
    """
    if not evaluations:
        return False, ["no promote_approved_items job completed this run"]
    reasons = [reason for evaluation in evaluations for reason in evaluation.reasons]
    return all(evaluation.eligible for evaluation in evaluations), reasons
