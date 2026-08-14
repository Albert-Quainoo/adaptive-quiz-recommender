"""CLI for reviewing immutable grounded questions and exporting approvals."""

import argparse
import json
from pathlib import Path

from api.schemas import QuizQuestion
from authoring.grounded_batch import PendingQuestion
from authoring.grounded_review import (
    CurationItem,
    GroundedReviewStore,
    RevisionProvenance,
    approve_revision,
    assert_immutable_source,
    build_pending_review,
    export_approved_bank_items,
    inspect_question,
    list_pending,
    load_source_questions,
    propose_revision,
    question_content_hash,
    reject_item,
    write_approved_bank,
)
from authoring.pilot_curation import build_pilot_review
from authoring.question_intents import PILOT_BATCH_ID
from authoring.review.card import format_review_card, review_priority
from authoring.review.models import AutomatedReviewReport
from authoring.review.reports import AutomatedReviewReportStore, review_report_path


def _reports_path(store_path: Path) -> Path:
    """The automated-review reports file sibling to a review store's own path --
    review_report_path()'s naming convention, applied directly since --store already
    is review_store_path / f"{batch_id}__{skill_id}.json"."""
    return review_report_path(store_path.parent, *store_path.stem.split("__", 1))


def _report_for_item(
    item: CurationItem,
    source: PendingQuestion | None,
    report_store: AutomatedReviewReportStore,
) -> AutomatedReviewReport | None:
    if source is None:
        return None
    pending_revisions = [
        revision for revision in item.revisions if revision.final_review_status == "pending"
    ]
    head = pending_revisions[-1].question if pending_revisions else source.question
    return report_store.latest_for_hash(question_content_hash(head))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--batch", type=Path, required=True)
    root.add_argument("--store", type=Path, required=True)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("list")
    inspect = commands.add_parser("inspect")
    inspect.add_argument("question_id")
    propose = commands.add_parser("propose")
    propose.add_argument("question_id")
    propose.add_argument("--question-json", type=Path, required=True)
    propose.add_argument("--editor", required=True)
    propose.add_argument("--note", required=True)
    approve = commands.add_parser("approve")
    approve.add_argument("question_id")
    approve.add_argument("revision_id")
    approve.add_argument("--reviewer", required=True)
    reject = commands.add_parser("reject")
    reject.add_argument("question_id")
    reject.add_argument("--reviewer", required=True)
    reject.add_argument("--reason", required=True)
    export = commands.add_parser("export")
    export.add_argument("--output", type=Path)
    return root


def main(argv=None) -> int:
    arguments = parser().parse_args(argv)
    store = GroundedReviewStore(arguments.store)
    if arguments.command == "prepare":
        if store.path.exists():
            raise ValueError(f"review store already exists: {store.path}")
        manifest = json.loads(
            (arguments.batch / "manifest.json").read_text(encoding="utf-8")
        )
        review = (
            build_pilot_review(arguments.batch)
            if manifest.get("batch_id") == PILOT_BATCH_ID
            else build_pending_review(arguments.batch)
        )
        store.save(review)
        return 0
    review = store.load()
    assert_immutable_source(arguments.batch, review)
    if arguments.command == "list":
        report_store = AutomatedReviewReportStore(_reports_path(arguments.store))
        source_questions = {
            question.question_id: question for question in load_source_questions(arguments.batch)
        }
        pending = list_pending(review)
        reports = {
            item.original_question_id: _report_for_item(
                item, source_questions.get(item.original_question_id), report_store
            )
            for item in pending
        }
        # Prioritize low-risk recommend_human_approval items first, so a reviewer
        # working top-down clears the easy, well-grounded items before anything
        # automated review flagged as risky.
        for item in sorted(pending, key=lambda item: review_priority(reports[item.original_question_id])):
            report = reports[item.original_question_id]
            auto = f"{report.recommendation}/{report.risk_level}" if report else "not yet reviewed"
            print(f"{item.original_question_id}\t{item.intent_id}\t{item.recommendation}\tauto={auto}")
    elif arguments.command == "inspect":
        inspected = inspect_question(arguments.batch, review, arguments.question_id)
        print(inspected.model_dump_json(indent=2))
        report = _report_for_item(
            inspected.curation,
            inspected.source_question,
            AutomatedReviewReportStore(_reports_path(arguments.store)),
        )
        supporting_passages = {
            record["reference_id"]: record["reference_material"] for record in inspected.references
        }
        print()
        print(
            format_review_card(
                inspected.source_question.question,
                inspected.curation,
                report,
                supporting_passages=supporting_passages,
            )
        )
    elif arguments.command == "propose":
        inspected = inspect_question(arguments.batch, review, arguments.question_id)
        edited = QuizQuestion.model_validate_json(arguments.question_json.read_text(encoding="utf-8"))
        store.replace_item(
            propose_revision(
                inspected.curation,
                inspected.source_question.question,
                edited,
                arguments.editor,
                arguments.note,
                provenance=RevisionProvenance.from_source(inspected.source_question),
            )
        )
    elif arguments.command == "approve":
        item = next(item for item in review.items if item.original_question_id == arguments.question_id)
        store.replace_item(approve_revision(item, arguments.revision_id, arguments.reviewer))
    elif arguments.command == "reject":
        item = next(item for item in review.items if item.original_question_id == arguments.question_id)
        store.replace_item(reject_item(item, arguments.reviewer, arguments.reason))
    else:
        if arguments.output:
            write_approved_bank(arguments.output, review)
        else:
            for item in export_approved_bank_items(review):
                print(json.dumps(item.model_dump(mode="json"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
