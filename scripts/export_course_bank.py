"""Export a course's approved-bank JSONL directly from its GroundedReviewStore --
the single source of truth after round-3 content fixes -- instead of hand-editing the
bank file. Fixes a real gap in authoring.grounded_review.export_approved_bank_items,
which raises StopIteration for approve_as_written items (approved with zero revisions):
it unconditionally does next(r for r in item.revisions if r.final_review_status ==
"approved"), which has nothing to iterate when there are no revisions at all. That path
is exercised by every item in these three courses' review stores that was approved as
written, and also by authoring/replenishment/worker.py's promotion job (line ~1114),
which would crash the same way if it ever tried to promote a batch containing one.

item_id matches the existing bank convention: original_question_id for an
approve_as_written item, revision.revision_id for an approved revision -- exactly what
is already on disk for these three banks (confirmed by inspecting both ID shapes
present in the committed JSONL).

Run with: python -m scripts.export_course_bank <course>
"""

import argparse
import json
from pathlib import Path

from api.bank import BankItem
from api.schemas import QuizQuestion
from authoring.grounded_review import GroundedReviewStore, load_source_questions

REPO_ROOT = Path(__file__).resolve().parent.parent

COURSES = {
    "dsa": {
        "review_store": REPO_ROOT / "outputs/replenishment/dsa/reviews/grounded-dsa-v1.json",
        "bank_path": REPO_ROOT / "outputs/approved_banks/dsa-approved-bank-28-v1.jsonl",
    },
    "linear-algebra": {
        "review_store": REPO_ROOT / "outputs/replenishment/linear-algebra/reviews/grounded-linear-algebra-v1.json",
        "bank_path": REPO_ROOT / "outputs/approved_banks/linear-algebra-approved-bank-24-v1.jsonl",
    },
    "database-systems": {
        "review_store": REPO_ROOT / "outputs/replenishment/database-systems/reviews/grounded-database-systems-v1.json",
        "bank_path": REPO_ROOT / "outputs/approved_banks/database-systems-approved-bank-28-v1.jsonl",
    },
}


def export_bank_items(review) -> list[BankItem]:
    batch_dir = REPO_ROOT / "outputs" / review.batch_id
    source_by_id = {q.question_id: q for q in load_source_questions(batch_dir)}
    exported = []
    for item in sorted(review.items, key=lambda value: value.original_question_id):
        if item.final_review_status != "approved":
            continue
        approved_revisions = [r for r in item.revisions if r.final_review_status == "approved"]
        if approved_revisions:
            revision = approved_revisions[0]
            exported.append(
                BankItem(
                    item_id=revision.revision_id,
                    question=revision.question,
                    provenance="generated",
                    skill_id=item.skill_id,
                )
            )
        else:
            # approve_as_written: no revision exists, original question content stands.
            source = source_by_id[item.original_question_id]
            exported.append(
                BankItem(
                    item_id=item.original_question_id,
                    question=QuizQuestion(**source.question.model_dump()),
                    provenance="generated",
                    skill_id=item.skill_id,
                )
            )
    return exported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("course", choices=sorted(COURSES))
    arguments = parser.parse_args()
    config = COURSES[arguments.course]
    review = GroundedReviewStore(config["review_store"]).load()

    items = export_bank_items(review)
    lines = [json.dumps(item.model_dump(mode="json"), sort_keys=True) for item in items]
    config["bank_path"].write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    print(f"wrote {len(lines)} items to {config['bank_path']}")


if __name__ == "__main__":
    main()
