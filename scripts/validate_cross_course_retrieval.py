"""Retrieval-only validation for the cross-course context fix.

Runs authoring.retrieval.search.retrieve_candidates live (real Brave Search,
real page fetches) for one representative skill per course, using the real
course manifest and taxonomy - the same course_anchor/context_vocabulary
worker.py's _handle_retrieve_references now passes for a real replenishment
job. No model calls. Writes nothing: the candidate store, the job database
and references.csv are all untouched, so this can be re-run freely.
"""

import sys

from authoring.replenishment.manifest import load_course_manifest
from authoring.retrieval.brave import BraveSearchProvider, MissingCredentials
from authoring.retrieval.diagnostics import RetrievalDiagnostics, summary
from authoring.retrieval.fetcher import HttpPageFetcher
from authoring.retrieval.relevance import course_context_vocabulary
from authoring.retrieval.search import SEARCH_LIMIT, retrieve_candidates
from taxonomy.loader import load_skills

REPRESENTATIVE_SKILLS = {
    "intro-ai": "AI-SRC-08",
    "dsa": "DSA-HSH-01",
    "linear-algebra": "LA-VSP-01",
    "database-systems": "DB-NRM-01",
}


def main() -> int:
    try:
        provider = BraveSearchProvider.from_environment()
    except MissingCredentials as error:
        print(error, file=sys.stderr)
        return 2

    overall_retained = 0
    zero_retained_courses: list[str] = []

    for course_id, skill_id in REPRESENTATIVE_SKILLS.items():
        manifest = load_course_manifest(course_id)
        catalogue = load_skills(manifest.skills_path(), manifest.references_path())
        skill = next(s for s in catalogue.skills if s.skill_id == skill_id)
        fetcher = HttpPageFetcher(manifest.allowed_domains)
        diagnostics = RetrievalDiagnostics()

        candidates = retrieve_candidates(
            skill,
            provider,
            fetcher,
            manifest.allowed_domains,
            limit=SEARCH_LIMIT,
            diagnostics=diagnostics,
            course_anchor=manifest.title,
            context_vocabulary=course_context_vocabulary(
                manifest.title, catalogue.skills
            ),
        )

        print(f"\n=== {course_id} / {skill_id} ({manifest.title}) ===")
        print(summary(diagnostics, skill_id))
        print(f"retained candidates: {len(candidates)}")
        for candidate in candidates:
            print(
                f"  [{candidate.relevance_score}] {candidate.source_domain}"
                f" -- {candidate.title!r}"
            )

        overall_retained += len(candidates)
        if not candidates:
            zero_retained_courses.append(course_id)

    print("\n=== overall ===")
    print(f"total retained across all courses: {overall_retained}")
    if zero_retained_courses:
        print(f"COURSES WITH ZERO RETAINED CANDIDATES: {zero_retained_courses}")
        return 1

    print("every course produced at least one retained candidate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
