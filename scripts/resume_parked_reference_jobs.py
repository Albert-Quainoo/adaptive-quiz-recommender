"""Re-run reference retrieval for the 8 jobs parked at
waiting_for_reference_review with zero pending candidates -- the fossil of
the pre-fix cross-course context bug (retrieval ran, found nothing, and
parked "waiting for review" with nothing to review).

Targets each job by its exact job_id via authoring.replenishment.worker
.process_job(), never claim_next()/process_one() -- claim_next() claims the
oldest *any* active job across every course, which could process an
unrelated job instead (see project memory: "Job-targeting gotcha, learned
live 2026-08-13"). Every job here has job_type="retrieve_references", so
process_job() only ever calls _handle_retrieve_references() for it -- no
model call, no generation, no promotion, whatever the outcome.

This does not enqueue anything new: every job_id must already exist and
already be job_type="retrieve_references". Job identity, history and
attempts counters are exactly worker.py's own -- nothing here writes to the
candidate store directly.
"""

import argparse
import sys
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
    _search_provider_factory,
)
import time

from authoring.replenishment.jobs import open_repository
from authoring.replenishment.manifest import load_preparation_eligible_manifests
from authoring.retrieval.diagnostics import RetrievalDiagnostics, summary
from authoring.retrieval.store import CandidateStore
import authoring.replenishment.worker as worker_module
from authoring.replenishment.worker import process_job

# One retrieve_candidates call already spends up to 10 Brave Search requests;
# firing 8 of these back-to-back with no gap risks silent rate-limit
# degradation (a live run once returned empty results for every one of the 8
# with no error raised). A short pause between jobs is cheap insurance.
PAUSE_BETWEEN_JOBS_SECONDS = 5

_original_retrieve_candidates = worker_module.retrieve_candidates


def _retrieve_candidates_with_diagnostics(*args, **kwargs):
    diagnostics = kwargs.get("diagnostics") or RetrievalDiagnostics()
    kwargs["diagnostics"] = diagnostics
    result = _original_retrieve_candidates(*args, **kwargs)
    print(summary(diagnostics, "  retrieval"))
    return result


worker_module.retrieve_candidates = _retrieve_candidates_with_diagnostics

PARKED_JOB_IDS = (
    "976f16ae-964f-4e49-9e18-044a3308ad14",  # database-systems / DB-ALG-01
    "1bb642dc-3cd4-4fea-92f3-645701f681f1",  # dsa / DSA-CPX-01
    "72ebc6cd-ab09-4fa5-a3aa-b4efabe4cd15",  # linear-algebra / LA-DET-01
    "a785506c-ee5c-478e-aa66-d3e807318adb",  # intro-ai / AI-FND-03
    "43605fab-615c-401a-bde8-ec4ceed154c3",  # database-systems / DB-ERM-01
    "2f62ddba-9b96-449e-b319-a51ebfc72408",  # dsa / DSA-HSH-01
    "8d9c05af-bdcd-429e-96bf-f9a152c3b759",  # linear-algebra / LA-EIG-01
    "0769da69-4c07-4779-bd29-4a6fbba75f41",  # intro-ai / AI-FND-04
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=str, default=DEFAULT_DATABASE_PATH)
    args = parser.parse_args()

    repository = open_repository(args.database, read_only=False)
    manifests = {m.course_id: m for m in load_preparation_eligible_manifests()}

    for index, job_id in enumerate(PARKED_JOB_IDS):
        if index > 0:
            time.sleep(PAUSE_BETWEEN_JOBS_SECONDS)

        job = repository.get(job_id)
        if job is None:
            print(f"SKIP {job_id}: no such job")
            continue
        if job.job_type != "retrieve_references":
            print(f"SKIP {job_id}: job_type is {job.job_type!r}, not retrieve_references")
            continue

        manifest = manifests[job.course_id]
        before_pending = sum(
            1
            for candidate in CandidateStore(manifest.candidate_store_path).load()
            if candidate.skill_id == job.skill_id
        )

        process_job(
            job,
            manifest,
            job_repository=repository,
            search_provider=_search_provider_factory(manifest),
            fetcher=_fetcher_factory(manifest),
            max_attempts=_max_attempts(),
        )

        after = repository.get(job_id)
        after_candidates = [
            candidate
            for candidate in CandidateStore(manifest.candidate_store_path).load()
            if candidate.skill_id == job.skill_id
        ]
        print(
            f"{job.course_id}/{job.skill_id}  {job_id}\n"
            f"  status: waiting_for_reference_review -> {after.status}\n"
            f"  candidates for this skill: {before_pending} -> {len(after_candidates)}"
            f" ({sum(c.review_status == 'pending' for c in after_candidates)} pending,"
            f" {sum(c.review_status == 'approved' for c in after_candidates)} approved,"
            f" {sum(c.review_status == 'rejected' for c in after_candidates)} rejected)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
