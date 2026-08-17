"""One-off round-3 content corrections for the DSA/Linear-Algebra/Database-Systems
three-course review packet, applied through the same GroundedReviewStore data model
review_grounded_batch.py uses (propose_revision/approve_revision), not by hand-editing
the approved-bank JSONL directly.

Each of the six fixed items already has final_review_status "approved" (either
approve_as_written or an approved round-2 revision), so propose_revision/approve_revision's
"only a pending item" guard has to be satisfied first: the item is reopened to pending
(and, for items with a prior approved revision, that revision is marked "rejected" as
superseded -- CurationItem's own validator requires exactly one approved revision on an
approved, non-approve_as_written item) before the new revision is proposed and approved,
exactly mirroring what approve_revision already does automatically for pending sibling
revisions.

Run with: python -m scripts.apply_round3_content_fixes
"""

from datetime import datetime, timezone
from pathlib import Path

from api.schemas import QuizQuestion
from authoring.grounded_batch import PendingQuestion
from authoring.grounded_review import (
    GroundedReviewStore,
    RevisionProvenance,
    approve_revision,
    load_source_questions,
    propose_revision,
)
from authoring.question_intents import QuestionIntent, load_blueprint_for_batch
from authoring.review.deterministic import run_deterministic_checks
from taxonomy.loader import load_skills
from taxonomy.schemas import ReferenceProvenance

REVIEWER = "claude (round-3 correction, delegated by Albert after adversarial audit)"
NOW = datetime(2026, 8, 19, 0, 0, 0, tzinfo=timezone.utc)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _reference_provenance(course_batch_dir: Path, reference_ids: list[str]) -> list[ReferenceProvenance]:
    manifest = __import__("json").loads((course_batch_dir / "manifest.json").read_text(encoding="utf-8"))
    by_id = {record["reference_id"]: record for record in manifest["references"]}
    return [ReferenceProvenance(**by_id[reference_id]) for reference_id in reference_ids]


FIXES = [
    {
        "course": "dsa",
        "review_store": REPO_ROOT / "outputs/replenishment/dsa/reviews/grounded-dsa-v1.json",
        "batch_dir": REPO_ROOT / "outputs/grounded-dsa-v1",
        "original_question_id": "DSA-CPX-01-8990a5065e53ee39",
        "changed_fields_note": "options[3] (D) rewritten from a second true statement to a false one",
        "edit": lambda q: {
            **q,
            "options": [
                q["options"][0],
                q["options"][1],
                q["options"][2],
                "Big-O notation gives the exact number of primitive operations an algorithm performs for a given input size.",
            ],
        },
        "review_note": (
            "Albert's adversarial review: option D ('Asymptotic (Big-O) notation describes how an "
            "algorithm's running time grows as input size increases, not its exact running time') is "
            "itself a true statement, alongside the declared correct answer B -- both are unambiguously "
            "correct, so the item had two defensible answers. Replaced D with a false claim about what "
            "Big-O notation gives (an exact operation count, which it does not), leaving exactly one "
            "defensible option."
        ),
    },
    {
        "course": "dsa",
        "review_store": REPO_ROOT / "outputs/replenishment/dsa/reviews/grounded-dsa-v1.json",
        "batch_dir": REPO_ROOT / "outputs/grounded-dsa-v1",
        "original_question_id": "DSA-SRC-01-e57feb4d11b7c4db",
        "changed_fields_note": "question stem constrained to the standard ascending-order algorithm",
        "edit": lambda q: {
            **q,
            "question": (
                "What property must a list have before the standard (ascending-order) binary "
                "search algorithm can be applied correctly?"
            ),
        },
        "review_note": (
            "Albert's adversarial review: 'must be sorted in ascending order' as an unqualified "
            "requirement is too strong -- binary search can operate on descending-sorted data if its "
            "comparison logic is adapted to match. Constrained the stem to the standard "
            "(ascending-order) binary search algorithm so the declared answer is no longer an "
            "overgeneralization; left the reference-grounded explanation and options unchanged."
        ),
    },
    {
        "course": "dsa",
        "review_store": REPO_ROOT / "outputs/replenishment/dsa/reviews/grounded-dsa-v1.json",
        "batch_dir": REPO_ROOT / "outputs/grounded-dsa-v1",
        "original_question_id": "DSA-STK-01-409871df4ec6fc00",
        "changed_fields_note": "options[3] (D) replaced with a genuinely incorrect data structure/behavior",
        "edit": lambda q: {
            **q,
            "options": [
                q["options"][0],
                q["options"][1],
                q["options"][2],
                "A priority queue to store the orders by priority, regardless of arrival order",
            ],
        },
        "review_note": (
            "Albert's adversarial review: the scenario states orders are processed in the same order "
            "they are received, so 'store in order received' and 'store in order processed' describe "
            "the same queue and both options answered the question -- not mutually exclusive as "
            "distractors. Replaced the fourth option with a priority queue, a genuinely incorrect "
            "data structure for this FIFO scenario, since nothing in the scenario involves priority."
        ),
    },
    {
        "course": "linear-algebra",
        "review_store": REPO_ROOT / "outputs/replenishment/linear-algebra/reviews/grounded-linear-algebra-v1.json",
        "batch_dir": REPO_ROOT / "outputs/grounded-linear-algebra-v1",
        "original_question_id": "LA-DET-01-e0c1dacf1005437e",
        "changed_fields_note": "options[3] (D) replaced with a clearly false, non-equivalent distractor",
        "edit": lambda q: {
            **q,
            "options": [
                q["options"][0],
                q["options"][1],
                q["options"][2],
                "No, the matrix is invertible only if it is a square matrix, regardless of its determinant.",
            ],
        },
        "review_note": (
            "Albert's adversarial review: option D ('the matrix is invertible only if its determinant "
            "is nonzero') restates the same true nonzero-determinant criterion as the declared answer B, "
            "just phrased as a necessary condition -- both correctly imply the matrix is not invertible "
            "here. Replaced D with a false, non-equivalent claim (squareness alone as the criterion, "
            "which this very matrix -- square, det=0, not invertible -- disproves)."
        ),
    },
    {
        "course": "linear-algebra",
        "review_store": REPO_ROOT / "outputs/replenishment/linear-algebra/reviews/grounded-linear-algebra-v1.json",
        "batch_dir": REPO_ROOT / "outputs/grounded-linear-algebra-v1",
        "original_question_id": "LA-EIG-01-f0f72d0aea465d6e",
        "changed_fields_note": "options[2] (C) and options[3] (D) replaced with false claims about similar matrices",
        "edit": lambda q: {
            **q,
            "options": [
                q["options"][0],
                q["options"][1],
                "A and B must have the same eigenvectors.",
                "A and B must both be diagonalizable.",
            ],
        },
        "review_note": (
            "Albert's adversarial review: similar matrices also share determinant and trace (both true "
            "facts), so options C and D were true alongside the declared answer B -- three defensible "
            "options, not one. Replaced C and D with false claims (shared eigenvectors and guaranteed "
            "diagonalizability are not implied by similarity) so the item tests specifically what "
            "similarity does and does not guarantee about eigenvalues."
        ),
    },
    {
        "course": "database-systems",
        "review_store": REPO_ROOT / "outputs/replenishment/database-systems/reviews/grounded-database-systems-v1.json",
        "batch_dir": REPO_ROOT / "outputs/grounded-database-systems-v1",
        "original_question_id": "DB-IDX-01-3209c4dbdff4627e",
        "changed_fields_note": "correct_answer/options[0] and explanation reworded to implementation-neutral 'data entries'",
        "edit": lambda q: {
            **q,
            "correct_answer": (
                "Leaf nodes hold data entries -- records or record pointers, depending on the index's "
                "organization -- together with the indexed key values; internal nodes hold only "
                "separator keys and pointers used to guide the search."
            ),
            "options": [
                (
                    "Leaf nodes hold data entries -- records or record pointers, depending on the "
                    "index's organization -- together with the indexed key values; internal nodes hold "
                    "only separator keys and pointers used to guide the search."
                ),
                q["options"][1],
                q["options"][2],
                q["options"][3],
            ],
            "explanation": (
                "In a B+ tree, leaf nodes hold the data entries for each key -- records or record "
                "pointers, depending on the index's organization -- alongside the key values "
                "themselves; internal nodes hold only separator keys and child pointers used to "
                "navigate the search, never the record data itself."
            ),
        },
        "review_note": (
            "Albert's adversarial review: the marked option flatly stated leaf nodes contain 'pointers "
            "to their corresponding records,' while the explanation already admitted leaves may hold "
            "record data OR record pointers depending on the index's organization -- an internal "
            "contradiction. Reworded the correct option and explanation to the implementation-neutral "
            "'data entries -- records or record pointers, depending on the organization,' consistent "
            "with the approved reference's own generic 'values are stored... in the leaf nodes.'"
        ),
    },
]


def main() -> None:
    for fix in FIXES:
        store = GroundedReviewStore(fix["review_store"])
        review = store.load()
        item = next(i for i in review.items if i.original_question_id == fix["original_question_id"])

        if item.revisions:
            head_revision = next(r for r in item.revisions if r.final_review_status == "approved")
            original = QuizQuestion(**head_revision.question.model_dump())
            provenance = RevisionProvenance(
                source_batch_id=head_revision.source_batch_id,
                intent_id=head_revision.intent_id,
                skill_id=head_revision.skill_id,
                reference_ids=head_revision.reference_ids,
                model_id=head_revision.model_id,
                model_revision=head_revision.model_revision,
                prompt_version=head_revision.prompt_version,
                prompt_hash=head_revision.prompt_hash,
            )
            item = item.model_copy(
                update={
                    "revisions": [
                        r.model_copy(
                            update={
                                "final_review_status": "rejected",
                                "reviewed_by": REVIEWER,
                                "reviewed_at": NOW,
                                "rejection_reason": "Superseded by round-3 correction.",
                            }
                        )
                        if r.revision_id == head_revision.revision_id
                        else r
                        for r in item.revisions
                    ]
                }
            )
        else:
            source = next(
                q
                for q in load_source_questions(fix["batch_dir"])
                if q.question_id == fix["original_question_id"]
            )
            original = QuizQuestion(**source.question.model_dump())
            provenance = RevisionProvenance.from_source(source)

        item = item.model_copy(
            update={
                "final_review_status": "pending",
                "reviewed_by": None,
                "reviewed_at": None,
                "recommendation": "propose_revision",
            }
        )

        edited = QuizQuestion(**fix["edit"](original.model_dump()))
        item = propose_revision(
            item, original, edited, REVIEWER, fix["review_note"], edited_at=NOW, provenance=provenance
        )
        new_revision_id = item.revisions[-1].revision_id
        item = approve_revision(item, new_revision_id, REVIEWER, reviewed_at=NOW)
        store.replace_item(item)

        blueprint = load_blueprint_for_batch(fix["batch_dir"].name)
        intent = next(i for i in blueprint.intents if i.intent_id == provenance.intent_id)
        skills_dir = REPO_ROOT / "taxonomy/data" / fix["course"]
        catalogue = load_skills(skills_dir / "skills.csv", skills_dir / "references.csv")
        skill = next(s for s in catalogue.skills if s.skill_id == provenance.skill_id)
        approved_references = _reference_provenance(fix["batch_dir"], provenance.reference_ids)

        candidate = PendingQuestion(
            batch_id=provenance.source_batch_id,
            question_id=fix["original_question_id"],
            skill_id=provenance.skill_id,
            question_index=0,
            intent_id=provenance.intent_id,
            seed=0,
            reference_ids=provenance.reference_ids,
            prompt_version=provenance.prompt_version,
            prompt_hash=provenance.prompt_hash,
            model_id=provenance.model_id,
            model_revision=provenance.model_revision,
            generation_parameters={},
            generated_at=NOW,
            git_commit="0" * 40,
            raw_response="n/a (hand-edited revision)",
            question={**edited.model_dump(), "intent_id": provenance.intent_id},
        )
        checks = run_deterministic_checks(candidate, skill, intent, approved_references)
        status = "PASS" if checks.all_passed else "FAIL"
        print(f"[{status}] {fix['original_question_id']} ({provenance.intent_id}) -> {new_revision_id}")
        for failure in checks.blocking_failures:
            print(f"    BLOCKING: {failure.code}: {failure.message}")


if __name__ == "__main__":
    main()
