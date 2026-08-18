"""Controlled end-to-end pilot: generation -> automated review -> disposable
human-approval -> promotion -> demand reconciliation, for one newly declared
blueprint slot per course (AI foundations, DSA, Linear Algebra, Database Systems).

Proves that PR #10's newly declared blueprint capacity can be turned into
approved-bank supply safely, with demand decreasing correctly after promotion --
using the real blueprints under authoring/blueprints/, the real taxonomy under
taxonomy/data/, the real generation pipeline (authoring/grounded_batch.py via the
real worker dispatch in authoring/replenishment/worker.py), the real automated
review pipeline (authoring/review/service.py), and the real bank-promotion path
(authoring/replenishment/worker.py::_handle_promote_approved_items) -- against the
live Modal inference endpoint configured in .env.

Every mutation happens in a disposable per-skill directory (job repository is an
in-memory SQLite database; the taxonomy directory is a byte-for-byte copy of the
real one, and the approved bank is a throwaway copy). The taxonomy copy matters:
authoring/replenishment/worker.py's _handle_generate_questions calls
scripts/import_reference_candidates.py::import_candidates(), which unconditionally
rewrites references.csv/reference_provenance.csv in canonical form on every run --
harmless when nothing new was imported (byte-identical content), but a real write
to whatever path it's given, so this pilot must never point it at the real
taxonomy/data/ files directly. The real
approved-bank files under outputs/approved_banks/ are only ever read, never
written -- verified below by a before/after byte comparison.

Scoping note: authoring/replenishment/worker.py's _handle_generate_questions
always sizes a batch to a skill's FULL blueprint pool (by design -- a real
replenishment job fills every deficient slot for a skill in one pass). To keep
this pilot to exactly one representative slot per skill (per the task's explicit
"do not generate across all empty slots yet" instruction) while still exercising
that real, unmodified worker function, every OTHER slot in the skill's pool is
pre-marked satisfied in the disposable bank copy before the job runs: a slot
already satisfied in the real production bank is copied verbatim (real content,
real provenance); a slot still genuinely empty in production but outside this
pilot's scope gets a clearly-labeled synthetic filler row (never presented as
real generated content -- see FILLER_MARKER below, and every filler id is
reported separately in the pilot output). This is the one place this pilot uses
a synthetic fixture on the success path, and it exists only to bound scope/cost,
never to fabricate evidence about generation or review quality.
"""

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from api.bank import BankItem
from api.schemas import QuizQuestion
from app.bootstrap import load_approved_bank
from authoring.grounded_batch import question_id as compute_question_id
from authoring.grounded_review import (
    GroundedReviewStore,
    RevisionProvenance,
    approve_as_written,
    approve_revision,
    approved_item_provenance,
    load_source_questions,
    propose_revision,
    question_content_hash,
)
from authoring.question_intents import intents_by_skill, load_blueprint_for_batch
from authoring.replenishment.demand import (
    compute_blueprint_slot_demand,
    compute_demand_fingerprint,
    deficient_slots,
    load_approved_item_ids,
)
from authoring.replenishment.jobs import SQLiteReplenishmentJobRepository
from authoring.replenishment.manifest import active_bank_path, load_course_manifest
from authoring.replenishment.modal_inference import ModalBatchModel
from authoring.replenishment.worker import _blueprint_generation_difficulty, process_job
from authoring.review.card import format_review_card
from authoring.review.config import ReviewPolicyConfig
from authoring.review.reports import AutomatedReviewReportStore, review_report_path
from authoring.review.reviewer import ModelBackedContentReviewer

FILLER_MARKER = "[PILOT-SCOPING FILLER -- not real content, never shown to a learner]"

PILOT_TARGETS = [
    {"course": "intro-ai", "batch_id": "grounded-ai-fnd-release-v1", "skill_id": "AI-FND-03", "intent_id": "AI-FND-03-INT-04"},
    {"course": "dsa", "batch_id": "grounded-dsa-v1", "skill_id": "DSA-SRT-01", "intent_id": "DSA-SRT-01-INT-05"},
    {"course": "linear-algebra", "batch_id": "grounded-linear-algebra-v1", "skill_id": "LA-DET-01", "intent_id": "LA-DET-01-INT-05"},
    {"course": "database-systems", "batch_id": "grounded-database-systems-v1", "skill_id": "DB-REL-01", "intent_id": "DB-REL-01-INT-06"},
]

TERMINAL_STATUSES = {
    "waiting_for_question_review",
    "waiting_for_full_human_review",
    "rejected_by_automated_review",
    "rejected_deterministically",
    "permanent_failure",
    "waiting_for_model",
    "completed",
}


def clock() -> datetime:
    return datetime.now(timezone.utc)


class MetadataCapturingModel:
    """Wraps a real ModalBatchModel, delegating every call unchanged, purely to
    record latency/tokens/finish_reason for this pilot's own report --
    grounded_batch.py's AttemptAudit never stores that metadata itself."""

    def __init__(self, inner: ModalBatchModel, calls: list, stage: str) -> None:
        self._inner = inner
        self._calls = calls
        self._stage = stage

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    @property
    def model_revision(self) -> str:
        return self._inner.model_revision

    def _call(self, messages, seed, generation_parameters):
        started = time.perf_counter()
        outcome = self._inner.generate_with_metadata(messages, seed, generation_parameters)
        elapsed = time.perf_counter() - started
        self._calls.append(
            {
                "stage": self._stage,
                "seed": seed,
                "latency_seconds": round(elapsed, 3),
                "finish_reason": outcome.finish_reason,
                "input_tokens": outcome.input_tokens,
                "output_tokens": outcome.output_tokens,
                "max_new_tokens": outcome.max_new_tokens,
            }
        )
        return outcome

    def generate(self, messages, seed, generation_parameters) -> str:
        return self._call(messages, seed, generation_parameters).text

    def generate_with_metadata(self, messages, seed, generation_parameters):
        return self._call(messages, seed, generation_parameters)


def _seed_bank_items(blueprint, skill_id: str, target_index: int, real_bank_path: Path):
    """Every slot except target_index, marked satisfied -- a real approved item
    copied where production genuinely has one, else a FILLER_MARKER row. Returns
    (items, filler_ids)."""
    pool = intents_by_skill(blueprint)[skill_id]
    real_items = [item for item in load_approved_bank(real_bank_path) if item.skill_id == skill_id]
    by_id = {item.item_id: item for item in real_items}

    items, filler_ids = [], []
    for index, intent in enumerate(pool):
        if index == target_index:
            continue
        slot_id = compute_question_id(blueprint.batch_id, skill_id, index)
        match = by_id.get(slot_id) or next(
            (item for item_id, item in by_id.items() if item_id and item_id.startswith(f"{slot_id}-rev-")),
            None,
        )
        if match is not None:
            items.append(match)
            continue
        filler_ids.append(slot_id)
        items.append(
            BankItem(
                item_id=slot_id,
                skill_id=skill_id,
                provenance="generated",
                question=QuizQuestion(
                    question=f"{FILLER_MARKER} {intent.intent_id}",
                    options=["a", "b", "c", "d"],
                    correct_answer="a",
                    explanation=f"{FILLER_MARKER} inserted only to scope this pilot to its one target slot.",
                    concept=intent.assessment_focus[:80],
                    difficulty=intent.difficulty or "intermediate",
                ),
            )
        )
    return items, filler_ids


def run_target(target: dict, pilot_root: Path) -> dict:
    course, batch_id, skill_id, intent_id = (
        target["course"], target["batch_id"], target["skill_id"], target["intent_id"],
    )
    print(f"\n=== {skill_id} / {intent_id} ({course}, {batch_id}) ===", flush=True)

    real_manifest = load_course_manifest(course)
    blueprint = load_blueprint_for_batch(batch_id)
    pool = intents_by_skill(blueprint)[skill_id]
    target_index = next(i for i, it in enumerate(pool) if it.intent_id == intent_id)
    difficulty = _blueprint_generation_difficulty(skill_id, pool)

    target_dir = pilot_root / skill_id
    target_dir.mkdir(parents=True, exist_ok=True)

    # A byte-for-byte copy, never the real directory: _handle_generate_questions
    # unconditionally rewrites references.csv/reference_provenance.csv on every
    # run (see module docstring) -- this must land on a disposable copy, not the
    # real taxonomy/data/ files, regardless of whether anything semantic changes.
    disposable_taxonomy_dir = target_dir / "taxonomy"
    shutil.copytree(real_manifest.taxonomy_path, disposable_taxonomy_dir, dirs_exist_ok=True)

    disposable_bank_path = target_dir / f"{course}-disposable-bank.jsonl"
    disposable_manifest = real_manifest.model_copy(
        update={
            "taxonomy_path": disposable_taxonomy_dir,
            "approved_bank_path": disposable_bank_path,
            "candidate_store_path": target_dir / "candidates.json",
            "review_store_path": target_dir / "reviews",
            "bkt_model_path": target_dir / "model.pkl",
        }
    )

    real_bank_path = active_bank_path(real_manifest)
    real_bank_bytes_before = real_bank_path.read_bytes()

    seed_items, filler_ids = _seed_bank_items(blueprint, skill_id, target_index, real_bank_path)
    disposable_bank_path.parent.mkdir(parents=True, exist_ok=True)
    disposable_bank_path.write_text(
        "".join(json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n" for item in seed_items),
        encoding="utf-8",
    )
    print(f"  seeded disposable bank: {len(seed_items)} slots ({len(filler_ids)} scoping filler, "
          f"{len(seed_items) - len(filler_ids)} real production items copied)", flush=True)

    approved_before = load_approved_item_ids(disposable_manifest)
    demand_before = compute_blueprint_slot_demand(blueprint, skill_id, approved_before)
    fingerprint_before = compute_demand_fingerprint(
        blueprint, skill_id, approved_before, difficulty=difficulty, target_supply=disposable_manifest.target_supply
    )
    deficient_before = [slot.intent_id for slot in deficient_slots(demand_before)]
    assert deficient_before == [intent_id], (
        f"expected only {intent_id} deficient before generation, got {deficient_before}"
    )

    result: dict = {
        "course": course,
        "batch_id": batch_id,
        "skill_id": skill_id,
        "intent_id": intent_id,
        "target_index": target_index,
        "difficulty": difficulty,
        "filler_slot_ids": filler_ids,
        "demand_before": [slot.model_dump() for slot in demand_before],
        "fingerprint_before": fingerprint_before,
        "model_calls": [],
        "stages": [],
    }

    model_calls: list = []
    gen_model = MetadataCapturingModel(ModalBatchModel(), model_calls, "generation")
    review_model = MetadataCapturingModel(ModalBatchModel(), model_calls, "review")

    repository = SQLiteReplenishmentJobRepository(":memory:")
    repository.initialize_schema()
    repository.enqueue(
        course_id=disposable_manifest.course_id, skill_id=skill_id, requested_count=1,
        clock=clock, job_type="generate_questions",
    )

    review_config = ReviewPolicyConfig(reviewer_provider="modal")

    job = None
    for _ in range(8):
        job = repository.claim_next(clock=clock)
        if job is None:
            break
        print(f"  processing job_type={job.job_type} status={job.status} ...", flush=True)
        process_job(
            job, disposable_manifest, job_repository=repository,
            search_provider=None, fetcher=None,
            model_factory=lambda: gen_model,
            reviewer_factory=lambda: ModelBackedContentReviewer(review_model),
            review_config=review_config, clock=clock,
        )
        after = repository.get(job.job_id)
        result["stages"].append(
            {"job_type": job.job_type, "status": after.status,
             "error_code": after.error_code, "error_message": after.error_message}
        )
        print(f"    -> status={after.status} error_code={after.error_code}", flush=True)
        if after.status in TERMINAL_STATUSES and after.status != "queued":
            break

    result["model_calls"] = model_calls
    final = repository.get(job.job_id) if job else None
    result["final_status"] = final.status if final else None

    if final is None or final.status not in ("waiting_for_question_review",):
        result["human_review_boundary"] = (
            f"stopped: job status is {final.status if final else 'no job'!r}, "
            "not eligible for this pilot's disposable human-approval step"
        )
        real_bank_bytes_after = real_bank_path.read_bytes()
        result["real_bank_untouched"] = real_bank_bytes_after == real_bank_bytes_before
        return result

    review_path = Path(final.metadata["review_path"])
    review = GroundedReviewStore(review_path).load()
    item = review.items[0]
    output_dir = disposable_manifest.review_store_path.parent / "batches" / f"{batch_id}__{skill_id}"
    source = next(q for q in load_source_questions(output_dir) if q.question_id == item.original_question_id)
    content_hash = question_content_hash(source.question)
    report_store = AutomatedReviewReportStore(review_report_path(disposable_manifest.review_store_path, batch_id, skill_id))
    report = report_store.latest_for_hash(content_hash)

    result["automated_review_report"] = report.model_dump(mode="json") if report else None
    result["human_review_packet"] = format_review_card(source.question, item, report)
    print(f"  automated review recommendation: {item.recommendation}", flush=True)

    if item.recommendation == "approve_as_written":
        approved_item = approve_as_written(item, "pilot-e2e-disposable-human-reviewer", reviewed_at=clock())
        GroundedReviewStore(review_path).replace_item(approved_item)
        result["human_action"] = "approve_as_written"
    elif item.recommendation == "propose_revision":
        revised_question = QuizQuestion.model_validate(
            source.question.model_dump()
            | {"explanation": source.question.explanation + " (confirmed by disposable pilot human reviewer)"}
        )
        proposed = propose_revision(
            item, source.question, revised_question, "pilot-e2e-disposable-human-reviewer",
            "Automated review recommended human approval; confirming content with a minor clarifying note.",
            edited_at=clock(), provenance=RevisionProvenance.from_source(source),
        )
        GroundedReviewStore(review_path).replace_item(proposed)
        revision_id = proposed.revisions[0].revision_id
        approved_item = approve_revision(proposed, revision_id, "pilot-e2e-disposable-human-reviewer", reviewed_at=clock())
        GroundedReviewStore(review_path).replace_item(approved_item)
        result["human_action"] = "propose_revision+approve_revision"
    else:
        result["human_action"] = (
            f"stopped_at_boundary: item.recommendation={item.recommendation!r} "
            "is not eligible for this pilot's disposable auto-approval"
        )
        real_bank_bytes_after = real_bank_path.read_bytes()
        result["real_bank_untouched"] = real_bank_bytes_after == real_bank_bytes_before
        return result

    print(f"  disposable human action: {result['human_action']}", flush=True)
    repository.mark_queued(job.job_id, job_type="promote_approved_items")
    promote_job = repository.claim_next(clock=clock)
    process_job(
        promote_job, disposable_manifest, job_repository=repository,
        search_provider=None, fetcher=None,
        model_factory=lambda: gen_model,
        reviewer_factory=lambda: ModelBackedContentReviewer(review_model),
        review_config=review_config, clock=clock,
    )
    after_promote = repository.get(promote_job.job_id)
    result["stages"].append(
        {"job_type": "promote_approved_items", "status": after_promote.status,
         "error_code": after_promote.error_code, "error_message": after_promote.error_message}
    )
    result["final_status"] = after_promote.status
    print(f"  promotion status: {after_promote.status}", flush=True)

    approved_after = load_approved_item_ids(disposable_manifest)
    demand_after = compute_blueprint_slot_demand(blueprint, skill_id, approved_after)
    fingerprint_after = compute_demand_fingerprint(
        blueprint, skill_id, approved_after, difficulty=difficulty, target_supply=disposable_manifest.target_supply
    )
    result["demand_after"] = [slot.model_dump() for slot in demand_after]
    result["fingerprint_after"] = fingerprint_after
    result["supply_before"] = len(approved_before)
    result["supply_after"] = len(approved_after)

    if result["human_action"] == "approve_as_written":
        # approved_item_provenance() requires an approved revision; an
        # approve_as_written item has none by definition -- the immutable
        # source PendingQuestion already carries the same provenance fields.
        result["promoted_item_provenance"] = {
            "item_id": source.question_id,
            "original_question_id": item.original_question_id,
            "reference_ids": source.reference_ids,
            "model_id": source.model_id,
            "model_revision": source.model_revision,
            "prompt_version": source.prompt_version,
            "prompt_hash": source.prompt_hash,
        }
    else:
        provenance = approved_item_provenance(GroundedReviewStore(review_path).load(), item.original_question_id)
        result["promoted_item_provenance"] = provenance.model_dump(mode="json")

    real_bank_bytes_after = real_bank_path.read_bytes()
    result["real_bank_untouched"] = real_bank_bytes_after == real_bank_bytes_before

    real_approved_now = load_approved_item_ids(real_manifest)
    real_demand_now = compute_blueprint_slot_demand(blueprint, skill_id, real_approved_now)
    real_target_slot = next(slot for slot in real_demand_now if slot.intent_id == intent_id)
    result["real_production_target_slot_still_deficient"] = not real_target_slot.satisfied

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--only-skill", action="append", default=None,
                         help="Restrict this run to one or more skill_ids from PILOT_TARGETS (repeatable).")
    args = parser.parse_args()

    targets = (
        [t for t in PILOT_TARGETS if t["skill_id"] in args.only_skill]
        if args.only_skill else PILOT_TARGETS
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for target in targets:
        try:
            results.append(run_target(target, args.out_dir))
        except Exception as exc:  # noqa: BLE001 -- a failure here is real pilot evidence, not a bug to hide
            results.append({**target, "exception": f"{type(exc).__name__}: {exc}"})
            print(f"  EXCEPTION: {type(exc).__name__}: {exc}", flush=True)

    report_path = args.out_dir / "pilot_results.json"
    report_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
