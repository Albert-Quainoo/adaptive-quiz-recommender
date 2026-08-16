"""Composition seam between the read-only course registry
(authoring/course_catalog/registry.py) and per-course learner runtimes.

Course controllers are built lazily: constructing a CourseCatalogController
loads only the course catalog itself (manifest JSON files -- cheap, no
bank/taxonomy/BKT-model I/O). A specific course's ApplicationController
(bank + taxonomy + fitted model + repository) is built on first access to
that course_id, then cached for the lifetime of this instance. A
'preparing'/'approved_for_preparation' course is never built at all, since
nothing ever calls resolve_active() for a course that isn't active.

ApplicationController itself stays course-agnostic (app/controller.py is
unchanged) -- this module is the only place that decides *which* controller
a learner's request uses, and *when* it gets built.
"""

import threading
from dataclasses import dataclass

from app.bootstrap import AppSettings, BootstrapError, build_controller_for_course
from app.controller import ApplicationController
from app.perf import phase
from authoring.course_catalog.registry import resolve_course
from authoring.replenishment.manifest import (
    CourseManifest,
    load_active_manifests,
    load_all_manifests,
    load_course_manifest,
)


@dataclass(frozen=True)
class ActiveCourse:
    course_id: str
    controller: ApplicationController
    manifest: CourseManifest


@dataclass(frozen=True)
class UnavailableCourse:
    course_id: str
    manifest: CourseManifest


@dataclass(frozen=True)
class UnrecognizedCourse:
    pass


CourseSelectionResult = ActiveCourse | UnavailableCourse | UnrecognizedCourse


class CourseCatalogController:
    """Not itself a learner-specific object -- safe to cache process-wide
    (e.g. behind Streamlit's st.cache_resource, one instance shared by every
    learner session). Its only mutable state is the lazy controller cache,
    keyed by course_id and guarded by a lock; nothing here ever holds a
    specific learner's session state."""

    def __init__(self, settings: AppSettings) -> None:
        self._settings = settings
        self._controllers: dict[str, ApplicationController] = {}
        self._lock = threading.Lock()

    def resolve_for_learner(
        self, query: str, *, correlation_id: str | None = None
    ) -> CourseSelectionResult:
        manifest = resolve_course(query, load_all_manifests())
        if manifest is None:
            return UnrecognizedCourse()
        if manifest.status != "active":
            return UnavailableCourse(course_id=manifest.course_id, manifest=manifest)
        controller = self.resolve_active(manifest.course_id, correlation_id=correlation_id)
        return ActiveCourse(
            course_id=manifest.course_id, controller=controller, manifest=manifest
        )

    def resolve_active(
        self, course_id: str, *, correlation_id: str | None = None
    ) -> ApplicationController:
        """Builds this course's controller on first call, from a cold read
        of its bank/taxonomy/BKT model; every call after that for the same
        course_id returns the cached instance instead. Raises if the course
        isn't (or is no longer) active -- a controller is never built, let
        alone cached, for a preparing or archived course."""

        with self._lock:
            cached = self._controllers.get(course_id)
            if cached is not None:
                with phase(
                    "controller_resolution",
                    correlation_id=correlation_id,
                    course_id=course_id,
                    cache_hit=True,
                ):
                    return cached

            with phase(
                "controller_resolution",
                correlation_id=correlation_id,
                course_id=course_id,
                cache_hit=False,
            ):
                manifest = load_course_manifest(course_id)
                if manifest.status != "active":
                    raise KeyError(f"{course_id!r} is not an active course")

                controller = build_controller_for_course(
                    self._settings, manifest, correlation_id=correlation_id
                )
                self._controllers[course_id] = controller
                return controller

    def list_active_courses(self) -> list[tuple[str, str]]:
        """Reads manifests only -- never touches a course's bank/model, so
        listing every active course's title never forces one to load."""
        return sorted(
            (manifest.course_id, manifest.title) for manifest in load_active_manifests()
        )

    def list_preparing_courses(self) -> list[tuple[str, str]]:
        return sorted(
            (manifest.course_id, manifest.title)
            for manifest in load_all_manifests()
            if manifest.status
            in {"approved_for_preparation", "preparing", "awaiting_content_approval"}
        )


def build_course_catalog(settings: AppSettings) -> CourseCatalogController:
    """Cheap: loads no course's runtime resources. Safe to call once at
    process startup and cache (app/main.py does, via st.cache_resource) --
    each course's real cost (bank/taxonomy/model I/O) is deferred to
    CourseCatalogController.resolve_active, the first time a learner
    actually selects that course."""
    if not load_active_manifests():
        raise BootstrapError("No active course is currently configured.")
    return CourseCatalogController(settings)
