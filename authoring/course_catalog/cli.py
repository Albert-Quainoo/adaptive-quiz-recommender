"""Protected offline admin CLI for the course catalog lifecycle.

    python -m authoring.course_catalog.cli inspect-course-readiness <course_id>
    python -m authoring.course_catalog.cli approve-course <course_id> --approver <id>
    python -m authoring.course_catalog.cli reject-course <course_id> --reason ...
    python -m authoring.course_catalog.cli archive-course <course_id> --approver <id>

This never runs inside Streamlit and has no auth flags of its own -- like
authoring/replenishment/cli.py, it is "protected" purely by being
unreachable from the learner-facing app process. It is never a substitute
for, and never overrides, individual question-review decisions made in
authoring/grounded_review.py.
"""

import argparse
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from authoring.course_catalog.lifecycle import (
    LifecycleError,
    advance_to_content_approval,
    advance_to_ready_and_activate,
    approve_course_definition,
    archive_course,
    begin_preparation,
    reject_course_definition,
)
from authoring.course_catalog.readiness import inspect_readiness
from authoring.course_catalog.repository import SQLiteCourseApprovalRepository
from authoring.replenishment.manifest import MANIFEST_DIRECTORY, load_course_manifest

DEFAULT_DATABASE_PATH: str | Path = os.getenv("QUIZ_DATABASE_URL") or Path(
    os.getenv("QUIZ_DATABASE_PATH", "data/adaptive_quiz.sqlite3")
)


def _open_repository(arguments: argparse.Namespace) -> SQLiteCourseApprovalRepository:
    repository = SQLiteCourseApprovalRepository(arguments.database)
    repository.initialize_schema()
    return repository


def _write_manifest(manifest) -> None:
    path = MANIFEST_DIRECTORY / f"{manifest.course_id}.json"
    path.write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def inspect_course_readiness(arguments: argparse.Namespace) -> int:
    manifest = load_course_manifest(arguments.course_id)
    report = inspect_readiness(manifest)

    if report.is_ready:
        print(f"{manifest.course_id}: ready")
    else:
        print(f"{manifest.course_id}: not ready")
        for blocker in report.blockers:
            print(f"  - {blocker}")

    repository = _open_repository(arguments)
    outcome = advance_to_ready_and_activate(manifest, report, repository)
    if outcome is not None:
        updated_manifest, record = outcome
        _write_manifest(updated_manifest)
        print(
            f"{manifest.course_id}: auto-activated "
            f"(record {record.record_id}, sequence {record.sequence_number})"
        )
    return 0


def approve_course(arguments: argparse.Namespace) -> int:
    manifest = load_course_manifest(arguments.course_id)
    repository = _open_repository(arguments)
    try:
        updated_manifest, record = approve_course_definition(
            manifest,
            repository,
            approver=arguments.approver,
            notes=arguments.notes,
            auto_activate_when_ready=not arguments.no_auto_activate,
        )
    except LifecycleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _write_manifest(updated_manifest)
    print(
        f"{manifest.course_id}: approved for preparation "
        f"(record {record.record_id}, sequence {record.sequence_number})"
    )
    return 0


def begin_preparation_command(arguments: argparse.Namespace) -> int:
    manifest = load_course_manifest(arguments.course_id)
    repository = _open_repository(arguments)
    try:
        updated_manifest, record = begin_preparation(
            manifest,
            repository,
            approver=arguments.approver,
            notes=arguments.notes,
        )
    except LifecycleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _write_manifest(updated_manifest)
    print(
        f"{manifest.course_id}: preparing "
        f"(record {record.record_id}, sequence {record.sequence_number})"
    )
    return 0


def advance_to_content_approval_command(arguments: argparse.Namespace) -> int:
    manifest = load_course_manifest(arguments.course_id)
    repository = _open_repository(arguments)
    try:
        updated_manifest, record = advance_to_content_approval(
            manifest,
            repository,
            approver=arguments.approver,
            notes=arguments.notes,
        )
    except LifecycleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _write_manifest(updated_manifest)
    print(
        f"{manifest.course_id}: awaiting_content_approval "
        f"(record {record.record_id}, sequence {record.sequence_number})"
    )
    return 0


def reject_course(arguments: argparse.Namespace) -> int:
    manifest = load_course_manifest(arguments.course_id)
    repository = _open_repository(arguments)
    try:
        record = reject_course_definition(
            manifest,
            repository,
            approver=arguments.approver,
            reason=arguments.reason,
        )
    except LifecycleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(
        f"{manifest.course_id}: rejected "
        f"(record {record.record_id}, sequence {record.sequence_number})"
    )
    return 0


def archive_course_command(arguments: argparse.Namespace) -> int:
    manifest = load_course_manifest(arguments.course_id)
    repository = _open_repository(arguments)
    try:
        updated_manifest, record = archive_course(
            manifest,
            repository,
            approver=arguments.approver,
            notes=arguments.notes,
        )
    except LifecycleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    _write_manifest(updated_manifest)
    print(
        f"{manifest.course_id}: archived "
        f"(record {record.record_id}, sequence {record.sequence_number})"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m authoring.course_catalog.cli",
        description="Inspect and drive the course-approval lifecycle.",
    )
    parser.add_argument("--database", type=str, default=DEFAULT_DATABASE_PATH)

    commands = parser.add_subparsers(dest="command", required=True)

    readiness_command = commands.add_parser(
        "inspect-course-readiness",
        help="report unmet activation requirements; auto-activate if all pass",
    )
    readiness_command.add_argument("course_id")
    readiness_command.set_defaults(handler=inspect_course_readiness)

    approve_command = commands.add_parser(
        "approve-course", help="approve a proposed course definition for preparation"
    )
    approve_command.add_argument("course_id")
    approve_command.add_argument("--approver", required=True)
    approve_command.add_argument("--notes", default=None)
    approve_command.add_argument(
        "--no-auto-activate",
        action="store_true",
        help="require a further manual step instead of activating automatically",
    )
    approve_command.set_defaults(handler=approve_course)

    begin_preparation_parser = commands.add_parser(
        "begin-preparation", help="move an approved-for-preparation course into preparing"
    )
    begin_preparation_parser.add_argument("course_id")
    begin_preparation_parser.add_argument("--approver", required=True)
    begin_preparation_parser.add_argument("--notes", default=None)
    begin_preparation_parser.set_defaults(handler=begin_preparation_command)

    advance_to_content_approval_parser = commands.add_parser(
        "advance-to-content-approval",
        help="move a preparing course to awaiting_content_approval",
    )
    advance_to_content_approval_parser.add_argument("course_id")
    advance_to_content_approval_parser.add_argument("--approver", required=True)
    advance_to_content_approval_parser.add_argument("--notes", default=None)
    advance_to_content_approval_parser.set_defaults(
        handler=advance_to_content_approval_command
    )

    reject_command = commands.add_parser(
        "reject-course", help="record a rejection of a proposed course definition"
    )
    reject_command.add_argument("course_id")
    reject_command.add_argument("--reason", required=True)
    reject_command.add_argument("--approver", default="unspecified")
    reject_command.set_defaults(handler=reject_course)

    archive_command = commands.add_parser("archive-course", help="archive an active course")
    archive_command.add_argument("course_id")
    archive_command.add_argument("--approver", required=True)
    archive_command.add_argument("--notes", default=None)
    archive_command.set_defaults(handler=archive_course_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    return arguments.handler(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
