"""The reviewer's side of retrieval-assisted authoring.

Retrieval fills a store with pending candidates; a person reads them and
decides. Nothing here approves anything on its own, and export writes to
stdout rather than to references.csv - the taxonomy changes when Albert
changes it.
"""

import argparse
import csv
import html
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
    PILOT_CLOSURE_SKILL_IDS,
    PILOT_SKILL_IDS,
    load_pilot_catalogue,
    plan_pilot,
    run_pilot,
)
from authoring.retrieval.search import SEARCH_LIMIT
from authoring.retrieval.store import CandidateStore, StoreError


def describe(candidate: ReferenceCandidate) -> str:
    """One candidate as a reviewer reads it, reasons included.

    The score and the terms behind it are shown because relevance filtering
    is the reason this candidate is here rather than one of the others, and a
    filter whose workings a reviewer cannot see is one they cannot correct.
    """
    reviewer = f" by {candidate.reviewer_id}" if candidate.reviewer_id else ""
    matched = ", ".join(candidate.matched_terms)

    return (
        f"{candidate.candidate_id}  {candidate.skill_id}  "
        f"[{candidate.review_status}{reviewer}]  {candidate.source_domain}\n"
        f"    {candidate.title}\n"
        f"    relevance {candidate.relevance_score}: {matched or 'not scored'}\n"
        f"    {candidate.passage[:160]}"
    )


def retrieve(arguments: argparse.Namespace) -> int:
    catalogue = load_pilot_catalogue()
    skill_ids = arguments.skill_ids or PILOT_SKILL_IDS

    if arguments.dry_run:
        print("Dry run: no search, no fetch, no store, no references.csv.\n")

        for skill_id, queries in plan_pilot(catalogue, skill_ids).items():
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
    by_skill: dict[str, RetrievalDiagnostics] = {}

    # Read before the run so this run looks for the shortfall rather than
    # rediscovering what the last one already put there.
    held = store.load()

    added = store.add(
        run_pilot(
            catalogue,
            provider,
            fetcher,
            limit=arguments.limit,
            diagnostics=diagnostics,
            by_skill=by_skill,
            known=held,
            skill_ids=skill_ids,
        )
    )

    print(f"Retrieved {len(added)} new candidate(s) into {arguments.store}.\n")

    for candidate in added:
        print(describe(candidate))

    # Per skill first, then the run: a skill that found nothing is the thing
    # worth acting on, and the total is where it hides.
    for skill_id, for_skill in by_skill.items():
        print(f"\n{summary(for_skill, skill_id)}")

    print(f"\n{summary(diagnostics, 'whole run')}")

    # A run that added nothing because every skill was already full has done
    # its job. Only a run that fell short of its targets has failed.
    if diagnostics.candidates_created or diagnostics.targets_reached == len(by_skill):
        return 0

    return 1


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


def review_bundle(arguments: argparse.Namespace) -> int:
    """Write a deterministic, strongest-first closure review bundle."""
    catalogue = load_pilot_catalogue()
    objectives = {
        skill.skill_id: skill.learning_objective for skill in catalogue.skills
    }
    candidates = [
        candidate
        for candidate in CandidateStore(arguments.store).load()
        if candidate.skill_id in PILOT_CLOSURE_SKILL_IDS
        and candidate.review_status == "pending"
    ]
    domain_best: dict[tuple[str, str, str], int] = {}

    for candidate in candidates:
        group = (
            candidate.skill_id,
            candidate.learning_objective_facet,
            candidate.source_domain,
        )
        domain_best[group] = max(
            domain_best.get(group, candidate.relevance_score),
            candidate.relevance_score,
        )

    candidates.sort(
        key=lambda item: (
            PILOT_CLOSURE_SKILL_IDS.index(item.skill_id),
            item.learning_objective_facet,
            -domain_best[
                (item.skill_id, item.learning_objective_facet, item.source_domain)
            ],
            item.source_domain,
            -item.relevance_score,
            item.source_url,
            item.candidate_id,
        )
    )

    lines = [
        "# Pilot-closure reference review",
        "",
        "All candidates are pending. Retrieval does not approve references.",
    ]
    current_skill = current_facet = current_domain = None

    for candidate in candidates:
        if candidate.skill_id != current_skill:
            current_skill = candidate.skill_id
            current_facet = current_domain = None
            for_skill = [item for item in candidates if item.skill_id == current_skill]
            domains = sorted({item.source_domain for item in for_skill})
            lines += [
                "",
                f"## {current_skill}",
                "",
                f"Learning objective: {html.escape(objectives[current_skill])}",
                "",
                (
                    f"Coverage: {len(for_skill)} pending candidate(s) across "
                    f"{len(domains)} approved domain(s): "
                    f"{html.escape(', '.join(domains))}. Target: approximately 5."
                ),
            ]
        if candidate.learning_objective_facet != current_facet:
            current_facet = candidate.learning_objective_facet
            current_domain = None
            lines += ["", f"### {html.escape(current_facet)}"]
        if candidate.source_domain != current_domain:
            current_domain = candidate.source_domain
            lines += ["", f"#### {html.escape(current_domain)}"]

        passage = html.escape(candidate.passage).replace("```", "` ` `")
        lines += [
            "",
            f"- **{candidate.candidate_id}** — relevance {candidate.relevance_score}",
            f"  - Title: {html.escape(candidate.title)}",
            f"  - URL: {candidate.source_url}",
            f"  - Matched: {html.escape(', '.join(candidate.matched_terms))}",
            "",
            "  ```text",
            f"  {passage}",
            "  ```",
        ]

    if not candidates:
        lines += ["", "No pending pilot-closure candidates are available."]

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {len(candidates)} pending candidate(s) to {arguments.output}.")

    return 0 if candidates else 1


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
        "retrieve", help="search and read pages for selected AI taxonomy skills"
    )
    retrieve_command.add_argument(
        "--skill-id",
        dest="skill_ids",
        action="append",
        help="taxonomy skill to retrieve (repeatable; defaults to the original pilot)",
    )
    retrieve_command.add_argument(
        "--dry-run",
        action="store_true",
        help="print the queries only: no network, no store, no references.csv",
    )
    retrieve_command.add_argument(
        "--limit",
        type=int,
        default=SEARCH_LIMIT,
        help=(
            f"usable candidates wanted per skill, not results read "
            f"(default: {SEARCH_LIMIT})"
        ),
    )
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

    bundle_command = commands.add_parser(
        "bundle", help="write the pending pilot-closure reference review bundle"
    )
    bundle_command.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/pilot_closure_reference_review.md"),
    )
    bundle_command.set_defaults(handler=review_bundle)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
