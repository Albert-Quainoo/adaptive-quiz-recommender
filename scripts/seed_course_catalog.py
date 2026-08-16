"""One-time catalog population: writes each of the 4 initial courses' first
CourseApprovalRecord, reflecting the spec-mandated starting state.

    python -m scripts.seed_course_catalog

intro-ai starts already active (it migrates the existing AI course), so it
has no 'proposed' state to transition from -- lifecycle.seed_initial_record
is used instead of the general approve-course CLI command, which enforces a
'proposed' precondition this course never had. The other three start
approved_for_preparation, pre-approved to begin background preparation.

Safe to re-run: append() always adds a new sequence-numbered record, so
running this twice just adds a second seed record per course rather than
raising -- intentional for a one-time bootstrap script, but it does mean
this should be run exactly once against a given database in practice.
"""

import os
from pathlib import Path

from authoring.course_catalog.lifecycle import seed_initial_record
from authoring.course_catalog.repository import SQLiteCourseApprovalRepository
from authoring.replenishment.manifest import load_course_manifest

DEFAULT_DATABASE_PATH: str | Path = os.getenv("QUIZ_DATABASE_URL") or Path(
    os.getenv("QUIZ_DATABASE_PATH", "data/adaptive_quiz.sqlite3")
)

SEED_APPROVER = "catalog-migration"


def main() -> int:
    repository = SQLiteCourseApprovalRepository(DEFAULT_DATABASE_PATH)
    repository.initialize_schema()

    intro_ai = load_course_manifest("intro-ai")
    record = seed_initial_record(
        intro_ai,
        repository,
        decision="activated",
        approver=SEED_APPROVER,
        notes="migrated from the original single-course AI deployment; already active",
    )
    print(f"{intro_ai.course_id}: seeded (record {record.record_id})")

    for course_id in ("dsa", "linear-algebra", "database-systems"):
        manifest = load_course_manifest(course_id)
        record = seed_initial_record(
            manifest,
            repository,
            decision="approved",
            approver=SEED_APPROVER,
            notes="registered and preapproved for background preparation",
        )
        print(f"{manifest.course_id}: seeded (record {record.record_id})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
