"""Append Claude's independent verification and the live automated-review outcome to
each round-3 revision's review_note (and sync CurationItem.recommendation_reason to
match), mirroring round-2's DB-ERM-01-INT-03 pattern. Deterministic checks and the live
Modal reviewer already ran clean for all six (see apply_round3_content_fixes.py and
run_round3_live_review.py); this only records that outcome in the review store.

Run with: python -m scripts.finalize_round3_review_notes
"""

from pathlib import Path

from authoring.grounded_review import GroundedReviewStore

REPO_ROOT = Path(__file__).resolve().parent.parent

CLAUDE_VERIFICATION = {
    "DSA-CPX-01-8990a5065e53ee39": (
        "Independently verified by Claude: the declared answer B is the only option "
        "grounded in the approved reference's statement that constant-factor changes "
        "shift where growth curves cross without changing which one eventually grows "
        "faster; the new option D (Big-O gives an exact operation count) is false, "
        "leaving exactly one defensible answer."
    ),
    "DSA-SRC-01-e57feb4d11b7c4db": (
        "Independently verified by Claude: with the stem now scoped to the standard "
        "ascending-order algorithm, 'sorted in ascending order' is the only defensible "
        "answer; the approved reference describes binary search purely in terms of "
        "already-sorted data and does not itself claim descending order is impossible, "
        "so no unreferenced claim was introduced."
    ),
    "DSA-STK-01-409871df4ec6fc00": (
        "Independently verified by Claude: the scenario names no priority ordering, so "
        "the new option D is a genuinely incorrect data structure/behavior for it, "
        "leaving 'a queue to store the orders in the order they are received' as the "
        "only defensible answer."
    ),
    "LA-DET-01-e0c1dacf1005437e": (
        "Independently verified by Claude: the new option D's 'square implies "
        "invertible' claim is false and is directly disproved by the question's own "
        "matrix (square, determinant zero, not invertible), leaving the declared answer "
        "as the only defensible one."
    ),
    "LA-EIG-01-f0f72d0aea465d6e": (
        "Independently verified by Claude: per the approved reference, similarity "
        "preserves eigenvalues but does not itself guarantee shared eigenvectors or "
        "diagonalizability, so the new options C and D are false, leaving the declared "
        "answer as the only defensible one."
    ),
    "DB-IDX-01-3209c4dbdff4627e": (
        "Independently verified by Claude: the reworded option and explanation now "
        "consistently use 'data entries -- records or record pointers, depending on the "
        "organization,' matching the approved reference's own generic 'values are "
        "stored... in the leaf nodes' without contradicting itself."
    ),
}

LIVE_REVIEW_SUFFIX = " Live automated review: recommend_human_approval/low, zero blocking reasons."

REVIEW_STORES = [
    REPO_ROOT / "outputs/replenishment/dsa/reviews/grounded-dsa-v1.json",
    REPO_ROOT / "outputs/replenishment/linear-algebra/reviews/grounded-linear-algebra-v1.json",
    REPO_ROOT / "outputs/replenishment/database-systems/reviews/grounded-database-systems-v1.json",
]


def main() -> None:
    for path in REVIEW_STORES:
        store = GroundedReviewStore(path)
        review = store.load()
        changed = False
        new_items = []
        for item in review.items:
            if item.original_question_id not in CLAUDE_VERIFICATION:
                new_items.append(item)
                continue
            revision = next(r for r in item.revisions if r.final_review_status == "approved")
            final_note = revision.review_note + " " + CLAUDE_VERIFICATION[item.original_question_id] + LIVE_REVIEW_SUFFIX
            new_revisions = [
                r.model_copy(update={"review_note": final_note}) if r.revision_id == revision.revision_id else r
                for r in item.revisions
            ]
            new_items.append(item.model_copy(update={"recommendation_reason": final_note, "revisions": new_revisions}))
            changed = True
        if changed:
            store.save(review.model_copy(update={"items": new_items}))
            print(f"updated {path}")


if __name__ == "__main__":
    main()
