"""Run the live (Modal-backed) semantic reviewer against the six round-3 revised
candidates, so each correction carries a real automated-review report (not just
deterministic checks) alongside Albert's and Claude's independent review, matching the
round-2 precedent ("Live automated review: recommend_human_approval/low, zero blocking
reasons").

Run with: python -m scripts.run_round3_live_review
"""

import json
from pathlib import Path

from dotenv import load_dotenv

from authoring.grounded_batch import PendingQuestion
from authoring.grounded_review import GroundedReviewStore
from authoring.question_intents import load_blueprint_for_batch
from authoring.replenishment.modal_inference import ModalBatchModel
from authoring.review.config import load_review_policy_config
from authoring.review.reports import AutomatedReviewReportStore, review_report_path
from authoring.review.reviewer import ModelBackedContentReviewer
from authoring.review.service import review_candidate
from taxonomy.loader import load_skills
from taxonomy.schemas import ReferenceProvenance

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")

TARGETS = [
    ("dsa", "grounded-dsa-v1", "DSA-CPX-01-8990a5065e53ee39"),
    ("dsa", "grounded-dsa-v1", "DSA-SRC-01-e57feb4d11b7c4db"),
    ("dsa", "grounded-dsa-v1", "DSA-STK-01-409871df4ec6fc00"),
    ("linear-algebra", "grounded-linear-algebra-v1", "LA-DET-01-e0c1dacf1005437e"),
    ("linear-algebra", "grounded-linear-algebra-v1", "LA-EIG-01-f0f72d0aea465d6e"),
    ("database-systems", "grounded-database-systems-v1", "DB-IDX-01-3209c4dbdff4627e"),
]

COURSE_REVIEW_STORE_DIR = {
    "dsa": REPO_ROOT / "outputs/replenishment/dsa/reviews",
    "linear-algebra": REPO_ROOT / "outputs/replenishment/linear-algebra/reviews",
    "database-systems": REPO_ROOT / "outputs/replenishment/database-systems/reviews",
}


def _reference_provenance(batch_dir: Path, reference_ids: list[str]) -> list[ReferenceProvenance]:
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    by_id = {record["reference_id"]: record for record in manifest["references"]}
    return [ReferenceProvenance(**by_id[reference_id]) for reference_id in reference_ids]


def main() -> None:
    reviewer = ModelBackedContentReviewer(ModalBatchModel())
    config = load_review_policy_config({"QUIZ_REVIEW_REVIEWER_PROVIDER": "modal"})

    for course, batch_id, question_id in TARGETS:
        review_store_dir = COURSE_REVIEW_STORE_DIR[course]
        store = GroundedReviewStore(review_store_dir / f"{batch_id}.json")
        review = store.load()
        item = next(i for i in review.items if i.original_question_id == question_id)
        revision = next(r for r in item.revisions if r.final_review_status == "approved")

        batch_dir = REPO_ROOT / "outputs" / batch_id
        blueprint = load_blueprint_for_batch(batch_id)
        intent = next(i for i in blueprint.intents if i.intent_id == revision.intent_id)
        skills_dir = REPO_ROOT / "taxonomy/data" / course
        catalogue = load_skills(skills_dir / "skills.csv", skills_dir / "references.csv")
        skill = next(s for s in catalogue.skills if s.skill_id == revision.skill_id)
        approved_references = _reference_provenance(batch_dir, revision.reference_ids)

        candidate = PendingQuestion(
            batch_id=revision.source_batch_id,
            question_id=question_id,
            skill_id=revision.skill_id,
            question_index=0,
            intent_id=revision.intent_id,
            seed=0,
            reference_ids=revision.reference_ids,
            prompt_version=revision.prompt_version,
            prompt_hash=revision.prompt_hash,
            model_id=revision.model_id,
            model_revision=revision.model_revision,
            generation_parameters={},
            generated_at=revision.edited_at,
            git_commit="0" * 40,
            raw_response="n/a (hand-edited revision)",
            question={**revision.question.model_dump(), "intent_id": revision.intent_id},
        )

        report_store = AutomatedReviewReportStore(
            review_report_path(review_store_dir, batch_id, revision.skill_id)
        )
        try:
            report = review_candidate(
                candidate,
                skill,
                intent,
                approved_references,
                reviewer=reviewer,
                config=config,
                report_store=report_store,
            )
            print(
                f"{question_id}: {report.recommendation}/{report.risk_level}, "
                f"blocking={report.blocking_reasons}, warnings={report.warnings}"
            )
        except Exception as exc:
            print(f"{question_id}: LIVE REVIEW FAILED ({type(exc).__name__}: {exc})")


if __name__ == "__main__":
    main()
