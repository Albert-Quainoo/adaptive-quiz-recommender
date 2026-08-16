"""Configuration-only startup for the learner-facing application."""

import json
import logging
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Mapping

from pyBKT.models import Model as PyBKTModel
from pydantic import ValidationError

from api.bank import BankItem
from authoring.replenishment.manifest import active_bank_path
from bkt.model import BKTModel
from bkt.service import BKTService
from bkt.sqlite_repository import SchemaMigrationRequiredError
from recommendation.policy import RecommendationPolicyConfig
from recommendation.service import RecommendationService
from recommendation.sqlite_repository import SQLiteRecommendationRepository
from taxonomy.loader import course_paths, load_skills

from app.controller import ApplicationController
from app.perf import phase


LOGGER = logging.getLogger(__name__)


class BootstrapError(RuntimeError):
    user_message = "The quiz service could not be initialized. Please try again later."


@dataclass(frozen=True)
class AppSettings:
    database_path: str | Path
    approved_bank_path: Path
    bkt_model_path: Path
    skills_path: Path
    references_path: Path
    model_version: str
    policy_version: str
    initial_mastery_probability: float = 0.20
    prerequisite_mastery_threshold: float = 0.75
    admin_status_enabled: bool = False
    maintenance_mode: bool = False
    max_session_questions: int | None = None

    @classmethod
    def from_sources(
        cls, secrets: Mapping[str, object] | None = None
    ) -> "AppSettings":
        secrets = secrets or {}

        def setting(name: str, default: str | None = None) -> str:
            value = os.getenv(name)
            if value is None and name in secrets:
                value = str(secrets[name])
            if value is None:
                if default is None:
                    raise BootstrapError(f"Required setting {name} is not configured")
                value = default
            return value

        default_skills, default_references = course_paths(setting("QUIZ_COURSE", "ai"))
        database_url = os.getenv("QUIZ_DATABASE_URL")
        if database_url is None and "QUIZ_DATABASE_URL" in secrets:
            database_url = str(secrets["QUIZ_DATABASE_URL"])
        database_path: str | Path = (
            database_url
            if database_url
            else Path(setting("QUIZ_DATABASE_PATH", "data/adaptive_quiz.sqlite3"))
        )
        return cls(
            database_path=database_path,
            approved_bank_path=Path(setting("QUIZ_APPROVED_BANK_PATH")),
            bkt_model_path=Path(setting("QUIZ_BKT_MODEL_PATH")),
            skills_path=Path(setting("QUIZ_SKILLS_PATH", str(default_skills))),
            references_path=Path(
                setting("QUIZ_REFERENCES_PATH", str(default_references))
            ),
            model_version=setting("QUIZ_BKT_MODEL_VERSION"),
            policy_version=setting(
                "QUIZ_RECOMMENDATION_POLICY_VERSION", "recommendation-policy-v1"
            ),
            initial_mastery_probability=float(
                setting("QUIZ_INITIAL_MASTERY_PROBABILITY", "0.20")
            ),
            prerequisite_mastery_threshold=float(
                setting("QUIZ_PREREQUISITE_MASTERY_THRESHOLD", "0.75")
            ),
            # Defaults disabled: the read-only replenishment/admin sidebar is
            # an operator debugging aid, not a learner-facing feature, and
            # must be explicitly opted into (never on by default in
            # production) -- see app/main.py's render_admin_status.
            admin_status_enabled=setting("QUIZ_ADMIN_STATUS_ENABLED", "false").strip().lower()
            in ("1", "true", "yes", "on"),
            # Defaults disabled: when enabled, prevents learner writes and
            # shows a maintenance message while still allowing basic startup
            # diagnostics -- see app/main.py's gate on this flag, set
            # explicitly by an operator during a migration deployment (see
            # RUNBOOK_course_id_migration.md), never automatically.
            maintenance_mode=setting("QUIZ_MAINTENANCE_MODE", "false").strip().lower()
            in ("1", "true", "yes", "on"),
        )


def load_approved_bank(path: Path) -> list[BankItem]:
    items: list[BankItem] = []
    with path.open(encoding="utf-8") as bank_file:
        for line_number, line in enumerate(bank_file, start=1):
            if not line.strip():
                continue
            try:
                item = BankItem.model_validate(json.loads(line))
            except (json.JSONDecodeError, ValidationError) as exc:
                raise BootstrapError(
                    f"Invalid approved bank record at line {line_number}"
                ) from exc
            if item.item_id is None or item.skill_id is None:
                raise BootstrapError(
                    f"Approved bank record at line {line_number} needs item_id and skill_id"
                )
            items.append(item)
    if not items:
        raise BootstrapError("The approved learner-facing bank is empty")
    return items


def load_fitted_bkt_model(path: Path, *, model_version: str, course_id: str) -> BKTModel:
    """Load a trusted offline-fitted pyBKT artifact; never fit in app startup."""

    engine = PyBKTModel(parallel=False)
    engine.load(str(path))
    if engine.fit_model is None:
        raise BootstrapError("The configured BKT artifact is not fitted")
    return BKTModel(engine, course_id=course_id, model_version=model_version, fitted=True)


def _build_low_supply_reporter(
    settings: AppSettings, *, course_id: str
) -> Callable[[str], None] | None:
    """An idempotent, DB-only enqueue -- never network or model calls, and
    never allowed to block or fail application startup."""
    try:
        from authoring.replenishment.jobs import SQLiteReplenishmentJobRepository

        job_repository = SQLiteReplenishmentJobRepository(settings.database_path)
        job_repository.initialize_schema()
    except Exception:
        LOGGER.warning(
            "replenishment job repository unavailable; low-supply reporting disabled",
            exc_info=True,
        )
        return None

    def report(skill_id: str) -> None:
        job_repository.enqueue(course_id=course_id, skill_id=skill_id, requested_count=1)

    return report


def build_controller(
    settings: AppSettings,
    *,
    course_id: str = "intro-ai",
    correlation_id: str | None = None,
) -> ApplicationController:
    """Build shared, learner-agnostic services and initialize durable storage
    for one course. Defaults to "intro-ai" -- the catalog id the original
    single-course "ai" deployment was migrated to (see
    authoring/replenishment/manifests/intro-ai.json and the course_id
    backfill migration) -- so a single-course caller that omits course_id
    stays consistent with that migration's backfilled data instead of
    silently writing a different, unmigrated "ai" course_id going forward.

    Runs once per course_id per process (see app/multi_course.py's
    CourseCatalogController, which caches the returned controller) -- the
    database_engine_acquisition/schema_status_check phases below never
    repeat for a course after this first cold build."""

    try:
        if not settings.approved_bank_path.is_file():
            raise BootstrapError("The approved question bank is unavailable.")
        if not settings.bkt_model_path.is_file():
            raise BootstrapError("The configured BKT model is unavailable.")
        catalogue = load_skills(settings.skills_path, settings.references_path)
        items = load_approved_bank(settings.approved_bank_path)
        known_skills = {skill.skill_id for skill in catalogue.skills}
        unknown = sorted({item.skill_id for item in items} - known_skills)
        if unknown:
            raise BootstrapError(
                "Approved bank contains unknown skills: " + ", ".join(unknown)
            )
        model = load_fitted_bkt_model(
            settings.bkt_model_path,
            model_version=settings.model_version,
            course_id=course_id,
        )
        model_parameters = model.get_parameters()
        model_skill_ids = set(model_parameters.index.get_level_values("skill"))
        bank_skill_ids = {item.skill_id for item in items if item.skill_id}
        missing_model_skills = sorted(bank_skill_ids - model_skill_ids)
        if missing_model_skills:
            raise BootstrapError(
                "The configured BKT model is missing approved-bank skills: "
                + ", ".join(missing_model_skills)
            )
        priors = model_parameters.xs("prior", level="param")["value"]
        if any(
            abs(float(prior) - settings.initial_mastery_probability) > 1e-9
            for prior in priors
        ):
            raise BootstrapError(
                "The configured initial mastery does not match the BKT model prior."
            )
        if isinstance(settings.database_path, Path):
            settings.database_path.parent.mkdir(parents=True, exist_ok=True)
        with phase(
            "database_engine_acquisition",
            correlation_id=correlation_id,
            course_id=course_id,
        ):
            repository = SQLiteRecommendationRepository(
                settings.database_path,
                course_id=course_id,
                skills=catalogue.skills,
                items=items,
            )
        # Runtime-safe: never migrates. Creates the latest schema only when
        # the database is genuinely empty, verifies the version otherwise,
        # and raises SchemaMigrationRequiredError (caught below) rather than
        # mutating an existing production schema. The explicit migration
        # itself only ever runs via scripts/migrate_course_ownership.py.
        # This is also this repository's first real connection/round trip
        # to the database (engine acquisition above is lazy).
        with phase(
            "schema_status_check", correlation_id=correlation_id, course_id=course_id
        ):
            repository.initialize_schema()
        policy = RecommendationPolicyConfig(
            initial_mastery_probability=settings.initial_mastery_probability,
            prerequisite_mastery_threshold=settings.prerequisite_mastery_threshold,
            policy_version=settings.policy_version,
        )
        return ApplicationController(
            course_id=course_id,
            skills=catalogue.skills,
            items=items,
            repository=repository,
            recommendation_service=RecommendationService(
                repository,
                course_id=course_id,
                model_version=settings.model_version,
                config=policy,
            ),
            bkt_service=BKTService(model, repository),
            low_supply_reporter=_build_low_supply_reporter(settings, course_id=course_id),
            max_session_questions=settings.max_session_questions,
        )
    except BootstrapError:
        LOGGER.exception("Application bootstrap failed")
        raise
    except SchemaMigrationRequiredError as exc:
        LOGGER.error("Schema migration required for course %s: %s", course_id, exc)
        raise BootstrapError(exc.user_message) from exc
    except Exception as exc:
        LOGGER.exception("Application bootstrap failed")
        raise BootstrapError("Application bootstrap failed") from exc


def build_controller_for_course(
    settings: AppSettings, manifest, *, correlation_id: str | None = None
) -> ApplicationController:
    """Build the ApplicationController for exactly one course, deriving its
    bank/model/taxonomy paths from that course's own manifest rather than
    settings' single-course fields (those stay only as an explicit override
    for local dev/tests/scripts that build one AppSettings directly, e.g.
    scripts/replay_bkt_mastery.py).

    This is the unit app/multi_course.py's CourseCatalogController calls
    lazily, once per course, only when a learner actually selects that
    course -- never eagerly for every active course at process startup.
    Sharing settings.database_path across every call is what gives course
    isolation for free (see authoring/course_catalog's design notes): each
    controller's self._items/self._skills can only ever hold that course's
    own bank/taxonomy, so cross-course leakage is structurally impossible."""

    course_settings = replace(
        settings,
        approved_bank_path=active_bank_path(manifest),
        bkt_model_path=manifest.bkt_model_path,
        skills_path=manifest.skills_path(),
        references_path=manifest.references_path(),
        model_version=manifest.default_bkt_model_version,
        max_session_questions=manifest.max_session_questions,
    )
    return build_controller(
        course_settings, course_id=manifest.course_id, correlation_id=correlation_id
    )
