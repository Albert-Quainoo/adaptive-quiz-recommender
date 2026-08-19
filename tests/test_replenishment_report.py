from datetime import datetime, timezone

from authoring.replenishment.jobs import ReplenishmentJob
from authoring.replenishment.policy import ReplenishmentDecision
from authoring.replenishment.report import (
    build_report,
    classify_outcome,
    deficiency_row,
    job_outcome_row,
    pending_approval_row,
    render_markdown,
)

FIXED_TIME = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)


def _job(**overrides) -> ReplenishmentJob:
    fields = dict(
        job_id="job-1", course_id="intro-ai", skill_id="AI-SRC-08",
        job_type="automated_review", status="waiting_for_question_review",
        requested_count=1, attempts=1, created_at=FIXED_TIME,
    )
    fields.update(overrides)
    return ReplenishmentJob(**fields)


def test_classify_outcome_covers_every_documented_bucket():
    assert classify_outcome("waiting_for_question_review") == "awaiting_approval"
    assert classify_outcome("waiting_for_full_human_review") == "awaiting_approval"
    assert classify_outcome("rejected_by_automated_review") == "rejected"
    assert classify_outcome("rejected_deterministically") == "rejected"
    assert classify_outcome("retryable_failure") == "retryable_failure"
    assert classify_outcome("permanent_failure") == "permanent_failure"
    assert classify_outcome("completed") == "generated"
    assert classify_outcome("cancelled") == "cancelled"
    assert classify_outcome("queued") == "in_progress"


def test_deficiency_row_distinguishes_enqueued_from_no_deficiency():
    enqueued = deficiency_row(
        ReplenishmentDecision(
            course_id="intro-ai", skill_id="AI-SRC-08", should_enqueue=True,
            requested_count=6, reason="supply 0 below threshold 3",
        ),
        difficulty="intermediate",
        dry_run=False,
    )
    assert enqueued.decision == "enqueued"
    assert enqueued.difficulty == "intermediate"

    satisfied = deficiency_row(
        ReplenishmentDecision(
            course_id="intro-ai", skill_id="AI-SRC-01", should_enqueue=False,
            requested_count=0, reason="supply 4 meets threshold 3",
        ),
        difficulty="-",
        dry_run=False,
    )
    assert satisfied.decision == "no_deficiency"


def test_deficiency_row_reports_proposed_not_enqueued_during_a_dry_run():
    proposed = deficiency_row(
        ReplenishmentDecision(
            course_id="intro-ai", skill_id="AI-SRC-08", should_enqueue=True,
            requested_count=6, reason="supply 0 below threshold 3",
        ),
        difficulty="intermediate",
        dry_run=True,
    )
    assert proposed.decision == "proposed"


def test_deficiency_row_flags_hand_authored_deficiency_as_manual_action_required():
    row = deficiency_row(
        ReplenishmentDecision(
            course_id="intro-ai", skill_id="AI-ETH-01", should_enqueue=False,
            requested_count=6,
            reason="supply 0 below threshold 3; hand_authored, not eligible for automated generation",
            manual_action_required=True,
        ),
        difficulty="-",
        dry_run=True,
    )
    assert row.decision == "manual_action_required"
    assert row.proposed_job_key == "-"
    assert "hand_authored" in row.reason


def test_deficiency_row_override_blocks_unresolved_difficulty():
    blocked = deficiency_row(
        ReplenishmentDecision(
            course_id="intro-ai", skill_id="AI-SRC-08", should_enqueue=True,
            requested_count=6, reason="supply 0 below threshold 3",
        ),
        difficulty="unknown",
        dry_run=True,
        decision_override="blocked",
        reason_override="blocked: no reviewed blueprint",
    )
    assert blocked.decision == "blocked"
    assert blocked.reason == "blocked: no reviewed blueprint"
    assert blocked.proposed_job_key == "-"


def test_job_outcome_row_carries_outcome_category_and_new_flag():
    row = job_outcome_row(_job(), is_new_candidate=True)
    assert row.outcome_category == "awaiting_approval"
    assert row.is_new_candidate is True
    assert row.error_code is None


def test_pending_approval_row_uses_created_at_as_waiting_since():
    row = pending_approval_row(_job(job_id="job-2"))
    assert row.job_id == "job-2"
    assert row.waiting_since == FIXED_TIME.isoformat()


def test_render_markdown_includes_every_section_and_handles_empty_tables():
    report = build_report(
        dry_run=True,
        deficiencies=[
            deficiency_row(
                ReplenishmentDecision(
                    course_id="intro-ai", skill_id="AI-SRC-08", should_enqueue=True,
                    requested_count=6, reason="supply 0 below threshold 3",
                ),
                difficulty="unknown",
                dry_run=True,
            )
        ],
        job_outcomes=[],
        pending_approvals=[],
        budget={},
        stop_reason="dry run: scan only, no worker processing",
        archived_job_dirs=[],
        clock=lambda: FIXED_TIME,
    )
    markdown = render_markdown(report)
    assert "## Deficiency scan" in markdown
    assert "AI-SRC-08" in markdown
    assert "## Job processing outcomes this run" in markdown
    assert "_none_" in markdown  # empty job-outcomes/pending-approvals tables
    assert "dry run" in markdown


def test_report_round_trips_to_a_plain_dict_for_json_serialization():
    report = build_report(
        dry_run=False, deficiencies=[], job_outcomes=[job_outcome_row(_job(), is_new_candidate=False)],
        pending_approvals=[pending_approval_row(_job())], budget={"ticks": 1},
        stop_reason=None, archived_job_dirs=["outputs/replenishment/intro-ai/job-1"],
        clock=lambda: FIXED_TIME,
    )
    payload = report.to_dict()
    assert payload["job_outcomes"][0]["job_id"] == "job-1"
    assert payload["budget"] == {"ticks": 1}
    assert payload["archived_job_dirs"] == ["outputs/replenishment/intro-ai/job-1"]


def test_build_report_defaults_auto_merge_fields_to_not_eligible():
    report = build_report(
        dry_run=False, deficiencies=[], job_outcomes=[], pending_approvals=[], budget={},
        stop_reason=None, archived_job_dirs=[], clock=lambda: FIXED_TIME,
    )
    assert report.auto_merge_eligible is False
    assert report.auto_merge_reasons == []


def test_render_markdown_shows_auto_merge_eligibility_and_reasons():
    eligible = build_report(
        dry_run=False, deficiencies=[], job_outcomes=[], pending_approvals=[], budget={},
        stop_reason=None, archived_job_dirs=[], clock=lambda: FIXED_TIME,
        auto_merge_eligible=True, auto_merge_reasons=[],
    )
    assert "## Auto-merge eligibility" in render_markdown(eligible)
    assert "Eligible: **True**" in render_markdown(eligible)
    assert "_no blocking reasons_" in render_markdown(eligible)

    blocked = build_report(
        dry_run=False, deficiencies=[], job_outcomes=[], pending_approvals=[], budget={},
        stop_reason=None, archived_job_dirs=[], clock=lambda: FIXED_TIME,
        auto_merge_eligible=False, auto_merge_reasons=["AI-SRC-08-x: risk_level=medium"],
    )
    markdown = render_markdown(blocked)
    assert "Eligible: **False**" in markdown
    assert "AI-SRC-08-x: risk_level=medium" in markdown
