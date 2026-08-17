"""Reconcile DB-ERM-01-208a6d1c67c7e626's review-store record with what the approved
bank actually contains.

Discovered while regenerating the bank directly from the review store (per Albert's
"generate the HTML directly from the exact final bank artifacts" instruction): round-2's
commit 9ed5687 replaced this item's disputed cardinality-direction content with a
placement-focused reformulation directly in the approved-bank JSONL, but the
GroundedReviewStore's CurationItem for this original_question_id was never updated to
match -- it still carries recommendation="approve_as_written" with zero revisions, which
means it still points at the original disputed content in
outputs/grounded-database-systems-v1/pending_questions.jsonl. A bank regenerated purely
from the review store (the intended single source of truth) would silently revert to the
disputed question -- exactly the packet/bank mismatch class of bug Albert flagged, one
level deeper.

The real fresh-generation record for the replacement exists at
outputs/grounded-database-systems-v1-erm03-fix/pending_questions.jsonl (git_commit
9ed5687, prompt_hash d8a46010...) and matches the bank's current content exactly. This
records it as a proper QuestionRevision with that genuine provenance, so the review
store and the bank agree.

Run with: python -m scripts.fix_db_erm01_int03_traceability
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
from authoring.question_intents import load_blueprint_for_batch
from authoring.review.deterministic import run_deterministic_checks
from taxonomy.loader import load_skills
from taxonomy.schemas import ReferenceProvenance

REPO_ROOT = Path(__file__).resolve().parent.parent
REVIEWER = "claude (round-3 correction, delegated by Albert after adversarial audit)"
NOW = datetime(2026, 8, 19, 0, 0, 0, tzinfo=timezone.utc)

ORIGINAL_BATCH_DIR = REPO_ROOT / "outputs/grounded-database-systems-v1"
FIX_BATCH_DIR = REPO_ROOT / "outputs/grounded-database-systems-v1-erm03-fix"
REVIEW_STORE_PATH = REPO_ROOT / "outputs/replenishment/database-systems/reviews/grounded-database-systems-v1.json"
QUESTION_ID = "DB-ERM-01-208a6d1c67c7e626"


def main() -> None:
    store = GroundedReviewStore(REVIEW_STORE_PATH)
    review = store.load()
    item = next(i for i in review.items if i.original_question_id == QUESTION_ID)
    assert not item.revisions, "expected zero prior revisions (approve_as_written)"

    original_source = next(q for q in load_source_questions(ORIGINAL_BATCH_DIR) if q.question_id == QUESTION_ID)
    fix_source = next(q for q in load_source_questions(FIX_BATCH_DIR) if q.question_id == QUESTION_ID)

    original = QuizQuestion(**original_source.question.model_dump())
    edited = QuizQuestion(**fix_source.question.model_dump())
    provenance = RevisionProvenance.from_source(fix_source)

    review_note = (
        item.recommendation_reason
        + " Reconciled the review store to match: this replacement was previously recorded "
        "only in the approved bank, with no QuestionRevision -- the review store still "
        "pointed at the original disputed content. Recorded here as a proper revision, "
        "using the fresh candidate's genuine generation provenance "
        f"(outputs/grounded-database-systems-v1-erm03-fix, prompt_hash {provenance.prompt_hash[:12]}...)."
    )

    item = item.model_copy(
        update={
            "final_review_status": "pending",
            "reviewed_by": None,
            "reviewed_at": None,
            "recommendation": "propose_revision",
        }
    )
    item = propose_revision(item, original, edited, REVIEWER, review_note, edited_at=NOW, provenance=provenance)
    new_revision_id = item.revisions[-1].revision_id
    item = approve_revision(item, new_revision_id, REVIEWER, reviewed_at=NOW)
    item = item.model_copy(update={"recommendation_reason": review_note})
    store.replace_item(item)

    blueprint = load_blueprint_for_batch("grounded-database-systems-v1")
    intent = next(i for i in blueprint.intents if i.intent_id == provenance.intent_id)
    skills_dir = REPO_ROOT / "taxonomy/data/database-systems"
    catalogue = load_skills(skills_dir / "skills.csv", skills_dir / "references.csv")
    skill = next(s for s in catalogue.skills if s.skill_id == provenance.skill_id)

    import json

    manifest = json.loads((FIX_BATCH_DIR / "manifest.json").read_text(encoding="utf-8"))
    by_id = {record["reference_id"]: record for record in manifest["references"]}
    approved_references = [ReferenceProvenance(**by_id[rid]) for rid in provenance.reference_ids]

    candidate = PendingQuestion(
        batch_id=provenance.source_batch_id,
        question_id=QUESTION_ID,
        skill_id=provenance.skill_id,
        question_index=2,
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
        raw_response="n/a (traceability reconciliation)",
        question={**edited.model_dump(), "intent_id": provenance.intent_id},
    )
    checks = run_deterministic_checks(candidate, skill, intent, approved_references)
    status = "PASS" if checks.all_passed else "FAIL"
    print(f"[{status}] {QUESTION_ID} ({provenance.intent_id}) -> {new_revision_id}")
    for failure in checks.blocking_failures:
        print(f"    BLOCKING: {failure.code}: {failure.message}")


if __name__ == "__main__":
    main()
