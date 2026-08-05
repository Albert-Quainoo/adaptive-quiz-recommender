"""The reviewer's side of retrieval-assisted authoring.

Retrieval fills a store with pending candidates; a person reads them and
decides. Nothing here approves anything on its own, and export writes to
stdout rather than to references.csv - the taxonomy changes when Albert
changes it.
"""

import argparse
import csv
import sys
from collections.abc import Sequence
from pathlib import Path

from authoring.retrieval.brave import BraveSearchProvider, MissingCredentials
from authoring.retrieval.diagnostics import RetrievalDiagnostics, summary
from authoring.retrieval.fetcher import HttpPageFetcher
from authoring.retrieval.models import (
    ReferenceCandidate,
    approve,
    export_reference_material,
    reject,
)
from authoring.retrieval.pilot import (
    DEFAULT_STORE_PATH,
    PILOT_ALLOWED_DOMAINS,
    load_pilot_catalogue,
    plan_pilot,
    run_pilot,
)
from authoring.retrieval.search import SEARCH_LIMIT
from authoring.retrieval.store import CandidateStore, StoreError


def describe(candidate: ReferenceCandidate) -> str:
    reviewer = f" by {candidate.reviewer_id}" if candidate.reviewer_id else ""

    return (
        f"{candidate.candidate_id}  {candidate.skill_id}  "
        f"[{candidate.review_status}{reviewer}]  {candidate.source_domain}\n"
        f"    {candidate.title}\n"
        f"    {candidate.passage[:160]}"
    )


def retrieve(arguments: argparse.Namespace) -> int:
    catalogue = load_pilot_catalogue()

    if arguments.dry_run:
        print("Dry run: no search, no fetch, no store, no references.csv.\n")

        for skill_id, queries in plan_pilot(catalogue).items():
            print(skill_id)

            for query in queries:
                print(f"  {query}")

        return 0

    try:
        provider = BraveSearchProvider.from_environment()
    except MissingCredentials as error:
        print(error, file=sys.stderr)

        return 2

    fetcher = HttpPageFetcher(PILOT_ALLOWED_DOMAINS)
    store = CandidateStore(arguments.store)
    diagnostics = RetrievalDiagnostics()

    added = store.add(
        run_pilot(
            catalogue, provider, fetcher, limit=arguments.limit, diagnostics=diagnostics
        )
    )

    print(f"Retrieved {len(added)} new candidate(s) into {arguments.store}.\n")

    for candidate in added:
        print(describe(candidate))

    print(summary(diagnostics))

    return 0 if diagnostics.candidates_created else 1


def list_candidates(arguments: argparse.Namespace) -> int:
    candidates = CandidateStore(arguments.store).load()

    if arguments.status:
        candidates = [
            candidate
            for candidate in candidates
            if candidate.review_status == arguments.status
        ]

    for candidate in candidates:
        print(describe(candidate))

    print(f"\n{len(candidates)} candidate(s) in {arguments.store}.")

    return 0


def review(arguments: argparse.Namespace, decide) -> int:
    store = CandidateStore(arguments.store)

    try:
        decided = store.replace(
            decide(store.get(arguments.candidate_id), arguments.reviewer, arguments.note)
        )
    except StoreError as error:
        print(error, file=sys.stderr)

        return 2

    print(describe(decided))

    return 0


def export(arguments: argparse.Namespace) -> int:
    exported = export_reference_material(CandidateStore(arguments.store).load())

    if not exported:
        print("No approved candidates to export.", file=sys.stderr)

        return 1

    writer = csv.writer(sys.stdout)
    writer.writerow(("skill_id", "reference_material"))

    for skill_id in sorted(exported):
        for passage in exported[skill_id]:
            writer.writerow((skill_id, passage))

    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m authoring.retrieval.cli",
        description="Retrieve and review reference candidates for the pilot skills.",
    )
    parser.add_argument(
        "--store",
        type=Path,
        default=DEFAULT_STORE_PATH,
        help=f"candidate store (default: {DEFAULT_STORE_PATH})",
    )

    commands = parser.add_subparsers(dest="command", required=True)

    retrieve_command = commands.add_parser(
        "retrieve", help="search and read pages for AI-SRC-01, AI-SRC-02 and AI-SRC-08"
    )
    retrieve_command.add_argument(
        "--dry-run",
        action="store_true",
        help="print the queries only: no network, no store, no references.csv",
    )
    retrieve_command.add_argument("--limit", type=int, default=SEARCH_LIMIT)
    retrieve_command.set_defaults(handler=retrieve)

    list_command = commands.add_parser("list", help="list candidates and their statuses")
    list_command.add_argument(
        "--status", choices=("pending", "approved", "rejected"), default=None
    )
    list_command.set_defaults(handler=list_candidates)

    for name, decide in (("approve", approve), ("reject", reject)):
        command = commands.add_parser(name, help=f"{name} one candidate by id")
        command.add_argument("candidate_id")
        command.add_argument("--reviewer", required=True, help="who is deciding")
        command.add_argument("--note", default=None)
        command.set_defaults(
            handler=lambda arguments, decide=decide: review(arguments, decide)
        )

    export_command = commands.add_parser(
        "export", help="print approved candidates as references.csv rows, on stdout"
    )
    export_command.set_defaults(handler=export)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
