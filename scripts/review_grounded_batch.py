"""CLI for reviewing immutable grounded questions and exporting approvals."""

import argparse
import json
from pathlib import Path

from api.schemas import QuizQuestion
from authoring.grounded_review import (
    GroundedReviewStore,
    RevisionProvenance,
    approve_revision,
    assert_immutable_source,
    export_approved_bank_items,
    inspect_question,
    list_pending,
    propose_revision,
    reject_item,
    write_approved_bank,
)
from authoring.pilot_curation import build_pilot_review


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
        store.save(build_pilot_review(arguments.batch))
        return 0
    review = store.load()
    assert_immutable_source(arguments.batch, review)
    if arguments.command == "list":
        for item in list_pending(review):
            print(f"{item.original_question_id}\t{item.intent_id}\t{item.recommendation}")
    elif arguments.command == "inspect":
        print(inspect_question(arguments.batch, review, arguments.question_id).model_dump_json(indent=2))
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
