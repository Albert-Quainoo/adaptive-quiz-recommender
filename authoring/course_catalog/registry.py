"""Conversational-alias resolution against the full course registry.

Looks across every registered manifest regardless of status -- including
'proposed'/'preparing' courses -- so a learner asking for a course that
exists but isn't active yet can still be recognized and shown the required
"being prepared" message, rather than a generic not-found error.
"""

from authoring.replenishment.manifest import CourseManifest, load_all_manifests


def _normalise(value: str) -> str:
    return " ".join(value.strip().casefold().split())


def resolve_course(query: str, manifests: list[CourseManifest] | None = None) -> CourseManifest | None:
    manifests = load_all_manifests() if manifests is None else manifests
    normalised_query = _normalise(query)
    if not normalised_query:
        return None
    for manifest in manifests:
        candidates = (manifest.course_id, manifest.title, *manifest.aliases)
        if normalised_query in {_normalise(candidate) for candidate in candidates}:
            return manifest
    return None
