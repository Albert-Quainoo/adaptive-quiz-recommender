"""Protected, explicit CLI for the course_id schema migration.

    python -m scripts.migrate_course_ownership --database <path-or-DSN>
    python -m scripts.migrate_course_ownership --database <path-or-DSN> --confirm

This is the only supported way to run bkt.sqlite_repository.
run_course_ownership_migration() against a real database. It never runs
automatically -- not from build_controller(), not from course selection,
login, or any Streamlit request, not from any test fixture. Run it
manually, from a machine with network access to the target database,
before opening traffic to a migration-aware deployment (see
RUNBOOK_course_id_migration.md).

Without --confirm this only reports the current schema status (read-only,
via check_schema_status) and exits -- it never mutates anything. --confirm
is required to actually run the migration.
"""

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from bkt.sqlite_repository import (
    DEFAULT_MIGRATION_COURSE_ID,
    SchemaStatus,
    check_schema_status,
    run_course_ownership_migration,
)
from database import create_engine_for

DEFAULT_DATABASE_PATH: str | Path = os.getenv("QUIZ_DATABASE_URL") or Path(
    os.getenv("QUIZ_DATABASE_PATH", "data/adaptive_quiz.sqlite3")
)


def _status(database: str | Path) -> SchemaStatus:
    engine = create_engine_for(database)
    try:
        with engine.connect() as connection:
            return check_schema_status(connection)
    finally:
        engine.dispose()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.migrate_course_ownership",
        description=(
            "Explicit, protected course_id schema migration. Never invoked "
            "automatically by the application -- see "
            "RUNBOOK_course_id_migration.md."
        ),
    )
    parser.add_argument("--database", type=str, default=DEFAULT_DATABASE_PATH)
    parser.add_argument(
        "--course-id",
        default=DEFAULT_MIGRATION_COURSE_ID,
        help="course_id backfilled onto every pre-migration row (default: %(default)s)",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="actually run the migration; without this flag, only reports status",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)

    status = _status(arguments.database)
    print(f"database: {arguments.database}")
    print(f"schema status: {status.value}")

    if status is SchemaStatus.CURRENT:
        print("nothing to do: schema is already current")
        return 0

    if not arguments.confirm:
        print("dry run: pass --confirm to run the migration")
        return 0

    print(f"running migration, backfilling course_id={arguments.course_id!r} ...")
    try:
        backfilled = run_course_ownership_migration(
            arguments.database, course_id=arguments.course_id
        )
    except Exception as exc:
        print(f"migration failed, rolled back: {exc}", file=sys.stderr)
        return 1

    print(f"migration succeeded: {backfilled}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
