"""Content-replenishment CLI: scan for low supply, work the queue, report status.

    python -m authoring.replenishment.cli scan
    python -m authoring.replenishment.cli worker --once
    python -m authoring.replenishment.cli worker
    python -m authoring.replenishment.cli status

This never runs inside Streamlit. It is the only thing that calls Brave
Search or Llama inference in this pipeline.
"""

import argparse
import os
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from authoring.replenishment.inventory import compute_course_inventory
from authoring.replenishment.jobs import SQLiteReplenishmentJobRepository
from authoring.replenishment.manifest import CourseManifest, load_active_manifests
from authoring.replenishment.modal_inference import ModalBatchModel
from authoring.replenishment.policy import ReplenishmentPolicyConfig, decide_replenishment
from authoring.replenishment.worker import BatchModel, LlamaBatchModel, process_one
from authoring.retrieval.brave import BraveSearchProvider
from authoring.retrieval.fetcher import HttpPageFetcher
from authoring.review.backpressure import apply_review_backlog_caps
from authoring.review.config import ReviewPolicyConfig, load_review_policy_config
from authoring.review.reports import count_pending_review_items
from authoring.review.reviewer import ContentReviewer, ModelBackedContentReviewer

DEFAULT_DATABASE_PATH = Path(os.getenv("QUIZ_DATABASE_PATH", "data/adaptive_quiz.sqlite3"))

INFERENCE_PROVIDERS: dict[str, Callable[[], BatchModel]] = {
    "local": LlamaBatchModel,
    "modal": ModalBatchModel,
}

# Review-specific: "fake" is deliberately absent here -- it is a test double with no
# usable standalone behavior, never a real CLI-selectable provider. A real worker run
# always reuses one of the same authenticated BatchModel providers generation already
# uses, wrapped by ModelBackedContentReviewer. max_new_tokens is read from
# ReviewPolicyConfig at call time (not baked into the lambda) so
# QUIZ_REVIEW_MAX_NEW_TOKENS is honored independently of generation's own limit.
REVIEW_PROVIDERS: dict[str, Callable[[], ContentReviewer]] = {
    "local": lambda: ModelBackedContentReviewer(
        LlamaBatchModel(), max_new_tokens=_review_config().reviewer_max_new_tokens
    ),
    "modal": lambda: ModelBackedContentReviewer(
        ModalBatchModel(), max_new_tokens=_review_config().reviewer_max_new_tokens
    ),
}


def _search_provider_factory(manifest: CourseManifest) -> BraveSearchProvider:
    return BraveSearchProvider.from_environment()


def _fetcher_factory(manifest: CourseManifest) -> HttpPageFetcher:
    return HttpPageFetcher(manifest.allowed_domains)


def _model_factory() -> Callable[[], BatchModel]:
    provider = os.getenv("QUIZ_REPLENISHMENT_INFERENCE_PROVIDER", "local")
    try:
        return INFERENCE_PROVIDERS[provider]
    except KeyError:
        raise SystemExit(
            f"QUIZ_REPLENISHMENT_INFERENCE_PROVIDER={provider!r} is not one of "
            f"{sorted(INFERENCE_PROVIDERS)}"
        ) from None


def _reviewer_factory() -> Callable[[], ContentReviewer]:
    provider = os.getenv("QUIZ_REVIEW_REVIEWER_PROVIDER", "local")
    try:
        return REVIEW_PROVIDERS[provider]
    except KeyError:
        raise SystemExit(
            f"QUIZ_REVIEW_REVIEWER_PROVIDER={provider!r} is not one of "
            f"{sorted(REVIEW_PROVIDERS)}"
        ) from None


def _review_config() -> ReviewPolicyConfig:
    return load_review_policy_config()


def _max_attempts() -> int:
    return int(os.getenv("QUIZ_REPLENISHMENT_MAX_ATTEMPTS", "5"))


def _poll_seconds() -> int:
    return int(os.getenv("QUIZ_REPLENISHMENT_POLL_SECONDS", "30"))


def _policy_config(manifest: CourseManifest) -> ReplenishmentPolicyConfig:
    low_supply_threshold = int(
        os.getenv(
            "QUIZ_REPLENISHMENT_LOW_SUPPLY_THRESHOLD", str(manifest.low_supply_threshold)
        )
    )
    target_supply = int(
        os.getenv("QUIZ_REPLENISHMENT_TARGET_SUPPLY", str(manifest.target_supply))
    )
    return ReplenishmentPolicyConfig(
        low_supply_threshold=low_supply_threshold, target_supply=target_supply
    )


def _open_repository(arguments: argparse.Namespace) -> SQLiteReplenishmentJobRepository:
    repository = SQLiteReplenishmentJobRepository(arguments.database)
    repository.initialize_schema()
    return repository


def scan(arguments: argparse.Namespace) -> int:
    repository = _open_repository(arguments)
    review_config = _review_config()
    enqueued = 0

    for manifest in load_active_manifests():
        inventory = compute_course_inventory(manifest, repository)
        decisions = decide_replenishment(inventory, _policy_config(manifest))
        # A review backlog is a normal waiting state, not a failure -- this only ever
        # downgrades an already-eligible decision, never enqueues anything new.
        decisions = apply_review_backlog_caps(
            decisions,
            pending_per_skill={
                skill_id: row.pending_generated_questions for skill_id, row in inventory.items()
            },
            total_pending_backlog=count_pending_review_items(manifest.review_store_path),
            config=review_config,
        )

        for decision in decisions:
            if not decision.should_enqueue:
                continue
            job = repository.enqueue(
                course_id=decision.course_id,
                skill_id=decision.skill_id,
                requested_count=decision.requested_count,
            )
            print(f"{job.job_id}  {job.course_id}/{job.skill_id}  {decision.reason}")
            enqueued += 1

    print(f"\n{enqueued} job(s) enqueued.")
    return 0


def worker(arguments: argparse.Namespace) -> int:
    repository = _open_repository(arguments)
    manifests = {manifest.course_id: manifest for manifest in load_active_manifests()}
    review_config = _review_config()

    def run_once() -> bool:
        job = process_one(
            repository,
            manifests,
            search_provider_factory=_search_provider_factory,
            fetcher_factory=_fetcher_factory,
            model_factory=_model_factory(),
            reviewer_factory=_reviewer_factory(),
            review_config=review_config,
            max_attempts=_max_attempts(),
        )
        if job is not None:
            processed = repository.get(job.job_id)
            print(f"{job.job_id}  {job.course_id}/{job.skill_id}  -> {processed.status}")
        return job is not None

    if arguments.once:
        processed = run_once()
        if not processed:
            print("No claimable job.")
        return 0

    print(f"Polling every {_poll_seconds()}s. Ctrl-C to stop.")
    try:
        while True:
            if not run_once():
                time.sleep(_poll_seconds())
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


def status(arguments: argparse.Namespace) -> int:
    repository = _open_repository(arguments)

    for manifest in load_active_manifests():
        inventory = compute_course_inventory(manifest, repository)
        print(f"\n== {manifest.course_id} ({manifest.title}) ==")
        for skill_id in sorted(inventory):
            row = inventory[skill_id]
            job_note = (
                f"{row.latest_job.job_type}/{row.latest_job.status}"
                if row.latest_job is not None
                else "-"
            )
            failure = (
                f"  [{row.latest_job.error_code}] {row.latest_job.error_message}"
                if row.latest_job is not None and row.latest_job.error_message
                else ""
            )
            print(
                f"{skill_id:<12} {row.readiness:<20} approved={row.total_approved_items:<3} "
                f"pending_refs={row.pending_reference_candidates:<3} "
                f"pending_qs={row.pending_generated_questions:<3} job={job_note}{failure}"
            )

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m authoring.replenishment.cli",
        description="Monitor content supply and drive the replenishment pipeline.",
    )
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)

    commands = parser.add_subparsers(dest="command", required=True)

    scan_command = commands.add_parser(
        "scan", help="enqueue replenishment for every active skill below its threshold"
    )
    scan_command.set_defaults(handler=scan)

    worker_command = commands.add_parser("worker", help="claim and process queued jobs")
    worker_command.add_argument(
        "--once", action="store_true", help="process at most one job, then exit"
    )
    worker_command.set_defaults(handler=worker)

    status_command = commands.add_parser(
        "status", help="print inventory and job state for every active skill"
    )
    status_command.set_defaults(handler=status)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
