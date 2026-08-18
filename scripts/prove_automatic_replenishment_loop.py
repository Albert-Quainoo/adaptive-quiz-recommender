"""Live, disposable proof that a real replenishment job advances itself
automatically through every stage once its blocking condition clears, using
the real Modal-backed model and reviewer -- not the deterministic fakes the
test suite uses.

Everything here lives under a throwaway temp directory: a scratch copy of
AI-SRC-08's real taxonomy row (read-only, never mutated), a scratch
blueprint/grounding-brief (monkeypatched module state, restored after), a
throwaway approved bank path, and a throwaway SQLite job database. Nothing
in outputs/, taxonomy/data/, or the real approved bank is read or written.
The one deliberate human action is the question-review approval step
(mirrors a reviewer using scripts/review_grounded_batch.py) -- everything
else in the chain below it fires with no human intervention, which is
exactly the claim this script exists to verify.

    python -m scripts.prove_automatic_replenishment_loop
"""

import json
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

import authoring.grounding_briefs as grounding_briefs
import authoring.question_intents as question_intents
from authoring.grounded_batch import PendingQuestion, read_jsonl
from authoring.grounded_review import (
    GroundedReviewStore,
    RevisionProvenance,
    approve_revision,
    propose_revision,
)
from authoring.grounding_briefs import CanonicalGroundingBrief
from authoring.question_intents import PilotBlueprint, QuestionIntent
from authoring.replenishment.demand import (
    compute_blueprint_slot_demand,
    compute_demand_fingerprint,
    load_approved_item_ids,
)
from authoring.replenishment.jobs import SQLiteReplenishmentJobRepository
from authoring.replenishment.manifest import CourseManifest, active_bank_pointer_path
from authoring.replenishment.modal_inference import ModalBatchModel
from authoring.replenishment.worker import process_job, ready_to_resume
from authoring.review.config import ReviewPolicyConfig
from authoring.review.reviewer import ModelBackedContentReviewer
from authoring.retrieval.models import new_candidate, approve as approve_candidate
from authoring.retrieval.store import CandidateStore
from api.schemas import QuizQuestion
from taxonomy.loader import course_paths

SKILL_ID = "AI-SRC-08"
FIXED_TIME = datetime(2026, 8, 18, 12, 0, tzinfo=timezone.utc)

# A real, previously-reviewed reference passage for AI-SRC-08 (redblobgames.com,
# already an allowed AI-course domain) -- disposable-store data, never written to
# the real candidate store or reference_provenance.csv.
PASSAGE = (
    "A heuristic function estimates the cost of the cheapest path from a given "
    "state to a goal state. It lets an informed search order the frontier by "
    "how promising a state looks rather than by how far it already is, which "
    "is what separates informed search from uninformed search."
)


def clock() -> datetime:
    return FIXED_TIME


def build_disposable_manifest(tmp_path: Path) -> CourseManifest:
    taxonomy_dir = tmp_path / "taxonomy"
    taxonomy_dir.mkdir()
    skills_path, _ = course_paths("ai")
    shutil.copy(skills_path, taxonomy_dir / "skills.csv")
    (taxonomy_dir / "references.csv").write_text(
        "skill_id,reference_material\n", encoding="utf-8"
    )
    return CourseManifest(
        course_id="ai",
        title="disposable-proof",
        version="1",
        taxonomy_path=taxonomy_dir,
        approved_bank_path=tmp_path / "bank" / "ai-bank-v0.jsonl",
        bkt_model_path=tmp_path / "model.pkl",
        candidate_store_path=tmp_path / "reference_candidates.json",
        review_store_path=tmp_path / "reviews",
        allowed_domains=("redblobgames.com",),
        low_supply_threshold=3,
        target_supply=6,
        default_bkt_model_version="test-v1",
        status="active",
    )


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="disposable-replenishment-proof-") as raw_tmp:
        tmp_path = Path(raw_tmp)
        manifest = build_disposable_manifest(tmp_path)

        candidate = new_candidate(
            SKILL_ID,
            "Heuristics",
            "https://redblobgames.com/pathfinding/a-star/introduction.html",
            "redblobgames.com",
            PASSAGE,
            FIXED_TIME,
            relevance_score=20,
            matched_terms=["heuristic", "frontier"],
        )
        store = CandidateStore(manifest.candidate_store_path)
        store.add([candidate])
        store.replace(approve_candidate(candidate, "albert", reviewed_at=FIXED_TIME))
        print(f"disposable approved candidate: {candidate.candidate_id}")

        blueprint_dir = tmp_path / "blueprints"
        blueprint_dir.mkdir()
        intent = QuestionIntent(
            intent_id=f"{SKILL_ID}-INT-01",
            skill_id=SKILL_ID,
            assessment_focus="What a heuristic estimates",
            question_archetype="definition recall",
            preferred_reference_ids=[candidate.candidate_id],
            required_concepts=["heuristic", "estimate"],
            prohibited_conflations=["heuristic equals exact cost"],
            difficulty="intermediate",
        )
        blueprint = PilotBlueprint(
            batch_id="disposable-proof-01",
            prompt_version=question_intents.PILOT_PROMPT_VERSION,
            review_status="blueprint-approved",
            reviewer_id="albert",
            reviewed_at=FIXED_TIME,
            base_seed=1,
            intents=[intent],
        )
        (blueprint_dir / "disposable-proof-01.json").write_text(
            json.dumps(blueprint.model_dump(mode="json")), encoding="utf-8"
        )
        original_blueprint_dir = question_intents.BLUEPRINT_DIRECTORY
        original_briefs = grounding_briefs.PILOT_GROUNDING_BRIEFS
        question_intents.BLUEPRINT_DIRECTORY = blueprint_dir
        grounding_briefs.PILOT_GROUNDING_BRIEFS = {
            SKILL_ID: CanonicalGroundingBrief(
                skill_id=SKILL_ID,
                version="test-v1",
                statements=["A heuristic estimates the remaining cost to the goal."],
            )
        }

        try:
            run(manifest, blueprint)
        finally:
            question_intents.BLUEPRINT_DIRECTORY = original_blueprint_dir
            grounding_briefs.PILOT_GROUNDING_BRIEFS = original_briefs

    return 0


def run(manifest: CourseManifest, blueprint: PilotBlueprint) -> None:
    repository = SQLiteReplenishmentJobRepository(":memory:")
    repository.initialize_schema()

    review_config = ReviewPolicyConfig()

    def model_factory():
        return ModalBatchModel()

    def reviewer_factory():
        return ModelBackedContentReviewer(
            ModalBatchModel(), max_new_tokens=review_config.reviewer_max_new_tokens
        )

    approved_before = load_approved_item_ids(manifest)
    demand_before = compute_blueprint_slot_demand(blueprint, SKILL_ID, approved_before)
    fingerprint_before = compute_demand_fingerprint(
        blueprint, SKILL_ID, approved_before, difficulty="intermediate", target_supply=6
    )
    print(f"\nbefore: approved_items={len(approved_before)} "
          f"deficient_slots={sum(not s.satisfied for s in demand_before)} "
          f"fingerprint={fingerprint_before[:12]}")

    job = repository.enqueue(course_id="ai", skill_id=SKILL_ID, requested_count=1, clock=clock)
    print(f"\nenqueued job {job.job_id}, status={job.status}\n")

    human_approved_once = False
    for step in range(1, 13):
        job = repository.get(job.job_id)
        if job.status in ("completed", "permanent_failure", "rejected_by_automated_review", "rejected_deterministically"):
            break

        if job.status in ("waiting_for_question_review", "waiting_for_full_human_review") and not human_approved_once:
            print(f"[{step}] AUTOMATION REACHED THE HUMAN-REVIEW BOUNDARY on its own "
                  f"(status={job.status}) -- approving the pending question now, "
                  f"the one deliberate human action in this run.")
            approve_pending_question(manifest, job)
            human_approved_once = True
            repository.mark_queued(job.job_id, job_type="promote_approved_items")
            job = repository.get(job.job_id)
            print(f"    -> queued for promotion automatically: status={job.status}, job_type={job.job_type}")
            continue

        if job.status not in ("queued",):
            print(f"[{step}] job stuck at status={job.status} job_type={job.job_type} "
                  f"error={job.error_code}:{job.error_message}")
            break

        before_type = job.job_type
        process_job(
            job, manifest,
            job_repository=repository,
            search_provider=None, fetcher=None,
            model_factory=model_factory,
            reviewer_factory=reviewer_factory,
            review_config=review_config,
            clock=clock,
        )
        after = repository.get(job.job_id)
        print(f"[{step}] {before_type} -> status={after.status} job_type={after.job_type} "
              f"(automatic, no human action)")
        job = after

    final = repository.get(job.job_id)
    print(f"\nfinal job status: {final.status}")

    if final.status != "completed":
        print("DID NOT REACH PROMOTION -- stopping short of the bank-write proof.")
        return

    pointer_path = active_bank_pointer_path(manifest)
    pointer = json.loads(pointer_path.read_text())
    bank_items = [json.loads(line) for line in Path(pointer["path"]).read_text().splitlines()]
    print(f"promoted bank now has {len(bank_items)} item(s), version={pointer['version']}")

    approved_after = load_approved_item_ids(manifest)
    demand_after = compute_blueprint_slot_demand(blueprint, SKILL_ID, approved_after)
    fingerprint_after = compute_demand_fingerprint(
        blueprint, SKILL_ID, approved_after, difficulty="intermediate", target_supply=6
    )
    print(f"\nafter:  approved_items={len(approved_after)} "
          f"deficient_slots={sum(not s.satisfied for s in demand_after)} "
          f"fingerprint={fingerprint_after[:12]}")

    print("\n=== VERDICT ===")
    print(f"supply +1:        {len(approved_after) - len(approved_before) == 1}")
    print(f"demand decreased: {sum(not s.satisfied for s in demand_after) < sum(not s.satisfied for s in demand_before)}")
    print(f"fingerprint changed: {fingerprint_before != fingerprint_after}")
    print(f"reached promotion with exactly one human action: {human_approved_once}")


def approve_pending_question(manifest: CourseManifest, job) -> None:
    review_path = Path(job.metadata["review_path"])
    review = GroundedReviewStore(review_path).load()
    item = review.items[0]
    output_dir = review_path.parent.parent / "batches" / f"{job.metadata['batch_id']}__{SKILL_ID}"

    source = next(
        q for q in read_jsonl(output_dir / "pending_questions.jsonl", PendingQuestion)
        if q.question_id == item.original_question_id
    )
    # Use the latest automated revision if one exists (real automated review may
    # have already proposed one), never a stale pre-revision draft -- see project
    # memory on pilot_curation_v3.py's approve-without-re-review gap.
    head = item.revisions[-1].question if item.revisions else source.question
    revised = QuizQuestion.model_validate(
        head.model_dump() | {"explanation": head.explanation + " (approved by disposable proof script)"}
    )
    proposed = propose_revision(
        item, source.question, revised, "albert",
        "disposable proof: human confirms the generated question",
        edited_at=FIXED_TIME,
        provenance=RevisionProvenance.from_source(source),
    )
    GroundedReviewStore(review_path).replace_item(proposed)
    approved = approve_revision(
        proposed, proposed.revisions[-1].revision_id, "albert", reviewed_at=FIXED_TIME
    )
    GroundedReviewStore(review_path).replace_item(approved)


if __name__ == "__main__":
    raise SystemExit(main())
