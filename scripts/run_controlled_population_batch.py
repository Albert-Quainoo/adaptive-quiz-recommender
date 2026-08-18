"""Drive one controlled bank-population batch through the real production
replenishment pipeline: enqueue -> worker (retrieval -> generation ->
automated review -> human-review boundary), using the same job repository,
manifests, and worker.process_one() the real replenishment CLI
(authoring/replenishment/cli.py) uses.

This is the normal production generation path, not a disposable pilot copy:
jobs are written to the real job database (data/adaptive_quiz.sqlite3 by
default) and real per-skill output/review artifacts under outputs/replenishment/.
It never touches the approved bank -- promotion (_handle_promote_approved_items)
only fires once a human has set final_review_status="approved" on a pending
review item, which nothing here ever does; a batch job that reaches
waiting_for_question_review or waiting_for_full_human_review simply stays
there until a human acts (authoring/replenishment/worker.py::ready_to_resume).

Only enqueues the skills named in --batch-file (never authoring.replenishment.
cli's scan(), which would enqueue every skill under its aggregate low-supply
threshold regardless of this phase's controlled scope) and stops once every
job in the batch reaches a terminal or human-blocked status, rather than
polling forever like the real worker daemon.

Records, per model call (generation and review both route through Modal), a
timing/token/finish_reason log -- the one piece of section 5/9's requested
metrics ("latency, token usage where available") that no existing artifact on
disk already carries (authoring/grounded_batch.py's AttemptAudit has no such
fields).

    python -m scripts.run_controlled_population_batch outputs/controlled_population/batch_001.json
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv

load_dotenv(REPO_ROOT / ".env")

from authoring.replenishment.cli import (
    DEFAULT_DATABASE_PATH,
    _fetcher_factory,
    _max_attempts,
    _review_config,
    _search_provider_factory,
)
from authoring.replenishment.jobs import open_repository
from authoring.replenishment.manifest import load_preparation_eligible_manifests
from authoring.replenishment.modal_inference import ModalBatchModel
from authoring.replenishment.worker import process_one
from authoring.review.reviewer import ModelBackedContentReviewer

# A job in one of these statuses needs either nothing more from this run, or
# an explicit human decision it cannot get from this script -- looping past
# this point would either spin forever or (for the review-boundary statuses)
# never resume without a human touching the review store, per
# worker.py's ready_to_resume().
BLOCKED_OR_TERMINAL = {
    "completed",
    "permanent_failure",
    "cancelled",
    "no_longer_needed",
    "waiting_for_reference_review",
    "waiting_for_question_review",
    "waiting_for_full_human_review",
    "rejected_by_automated_review",
    "rejected_deterministically",
}

MAX_ITERATIONS = 400
MAX_WALL_SECONDS = 3600
IDLE_POLL_SECONDS = 10


class InstrumentedModalBatchModel(ModalBatchModel):
    """Same ModalBatchModel the real 'modal' provider uses, plus a shared
    call log (timestamp, elapsed seconds, tokens, finish_reason) recorded on
    every real network call -- generation calls route through generate(),
    review calls through generate_with_metadata(); both funnel through the
    same _request() this only wraps, never duplicating a live call."""

    def __init__(self, *, call_log: list, role: str, **kwargs):
        super().__init__(**kwargs)
        self._call_log = call_log
        self._role = role

    def _timed_request(self, messages, seed, generation_parameters):
        started_monotonic = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        try:
            outcome = self._request(messages, seed, generation_parameters)
        except Exception as exc:
            self._call_log.append(
                {
                    "role": self._role,
                    "started_at": started_at,
                    "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
                    "error": str(exc),
                }
            )
            raise
        self._call_log.append(
            {
                "role": self._role,
                "started_at": started_at,
                "elapsed_seconds": round(time.monotonic() - started_monotonic, 3),
                "finish_reason": outcome.finish_reason,
                "input_tokens": outcome.input_tokens,
                "output_tokens": outcome.output_tokens,
            }
        )
        return outcome

    def generate(self, messages, seed, generation_parameters):
        return self._timed_request(messages, seed, generation_parameters).text

    def generate_with_metadata(self, messages, seed, generation_parameters):
        return self._timed_request(messages, seed, generation_parameters)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_file", type=Path)
    parser.add_argument("--database", type=str, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--out", type=Path, default=None, help="where to write the run report JSON")
    args = parser.parse_args(argv)

    batch = json.loads(args.batch_file.read_text(encoding="utf-8"))
    slots = batch["slots"]

    repository = open_repository(args.database, read_only=False)
    manifests = {m.course_id: m for m in load_preparation_eligible_manifests()}

    call_log: list[dict] = []

    def model_factory():
        return InstrumentedModalBatchModel(call_log=call_log, role="generation")

    def reviewer_factory():
        return ModelBackedContentReviewer(
            InstrumentedModalBatchModel(call_log=call_log, role="review"),
            max_new_tokens=_review_config().reviewer_max_new_tokens,
        )

    batch_job_ids: dict[str, str] = {}  # job_id -> "course_id/skill_id"
    for slot in slots:
        job = repository.enqueue(
            course_id=slot["course_id"],
            skill_id=slot["skill_id"],
            requested_count=slot["requested_count"],
            metadata={"controlled_population_batch": batch["batch"]},
        )
        batch_job_ids[job.job_id] = f"{slot['course_id']}/{slot['skill_id']}"
        print(f"enqueued/active: {job.job_id}  {slot['course_id']}/{slot['skill_id']}  status={job.status}")

    review_config = _review_config()
    started = time.monotonic()
    iterations = 0

    while True:
        current = {job_id: repository.get(job_id) for job_id in batch_job_ids}
        if all(job.status in BLOCKED_OR_TERMINAL for job in current.values()):
            break
        if iterations >= MAX_ITERATIONS or (time.monotonic() - started) > MAX_WALL_SECONDS:
            print(f"\nSTOP: iteration/time budget exhausted ({iterations} iterations).", file=sys.stderr)
            break

        iterations += 1
        processed = process_one(
            repository,
            manifests,
            search_provider_factory=_search_provider_factory,
            fetcher_factory=_fetcher_factory,
            model_factory=model_factory,
            reviewer_factory=reviewer_factory,
            review_config=review_config,
            max_attempts=_max_attempts(),
        )
        if processed is not None:
            after = repository.get(processed.job_id)
            tag = batch_job_ids.get(processed.job_id, "(other job, not in this batch)")
            print(f"[{iterations}] {processed.job_id}  {tag}  {processed.job_type} -> {after.status}")
        else:
            time.sleep(IDLE_POLL_SECONDS)

    final_jobs = {job_id: repository.get(job_id) for job_id in batch_job_ids}
    report = {
        "batch": batch["batch"],
        "database": str(args.database),
        "iterations": iterations,
        "wall_seconds": round(time.monotonic() - started, 1),
        "jobs": [
            {
                "job_id": job_id,
                "course_skill": batch_job_ids[job_id],
                "final_status": job.status,
                "job_type": job.job_type,
                "attempts": job.attempts,
                "error_code": job.error_code,
                "error_message": job.error_message,
                "metadata": job.metadata,
            }
            for job_id, job in final_jobs.items()
        ],
        "model_call_log": call_log,
    }

    print(f"\n=== Batch {batch['batch']} finished: {iterations} iterations, {report['wall_seconds']}s ===")
    for row in report["jobs"]:
        print(f"  {row['course_skill']:<30} {row['final_status']:<28} attempts={row['attempts']}")
    print(f"\n{len(call_log)} real model calls logged (generation + review).")

    out_path = args.out or args.batch_file.with_name(f"batch_{batch['batch']:03d}_run_report.json")
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
