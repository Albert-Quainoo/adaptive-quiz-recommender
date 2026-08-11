"""Configuration-only startup for the learner-facing application."""

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from pyBKT.models import Model as PyBKTModel
from pydantic import ValidationError

from api.bank import BankItem
from bkt.model import BKTModel
from bkt.service import BKTService
from recommendation.policy import RecommendationPolicyConfig
from recommendation.service import RecommendationService
from recommendation.sqlite_repository import SQLiteRecommendationRepository
from taxonomy.loader import course_paths, load_skills

from app.controller import ApplicationController


LOGGER = logging.getLogger(__name__)


class BootstrapError(RuntimeError):
    user_message = "The quiz service could not be initialized. Please try again later."


@dataclass(frozen=True)
class AppSettings:
    database_path: Path
    approved_bank_path: Path
    bkt_model_path: Path
    skills_path: Path
    references_path: Path
    model_version: str
    policy_version: str
    prerequisite_mastery_threshold: float = 0.75

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
        return cls(
            database_path=Path(setting("QUIZ_DATABASE_PATH", "data/adaptive_quiz.sqlite3")),
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
            prerequisite_mastery_threshold=float(
                setting("QUIZ_PREREQUISITE_MASTERY_THRESHOLD", "0.75")
            ),
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


def load_fitted_bkt_model(path: Path, *, model_version: str) -> BKTModel:
    """Load a trusted offline-fitted pyBKT artifact; never fit in app startup."""

    engine = PyBKTModel(parallel=False)
    engine.load(str(path))
    if engine.fit_model is None:
        raise BootstrapError("The configured BKT artifact is not fitted")
    return BKTModel(engine, model_version=model_version, fitted=True)


def build_controller(settings: AppSettings) -> ApplicationController:
    """Build shared, learner-agnostic services and initialize durable storage."""

    try:
        catalogue = load_skills(settings.skills_path, settings.references_path)
        items = load_approved_bank(settings.approved_bank_path)
        known_skills = {skill.skill_id for skill in catalogue.skills}
        unknown = sorted({item.skill_id for item in items} - known_skills)
        if unknown:
            raise BootstrapError(
                "Approved bank contains unknown skills: " + ", ".join(unknown)
            )
        model = load_fitted_bkt_model(
            settings.bkt_model_path, model_version=settings.model_version
        )
        settings.database_path.parent.mkdir(parents=True, exist_ok=True)
        repository = SQLiteRecommendationRepository(
            settings.database_path, skills=catalogue.skills, items=items
        )
        repository.initialize_schema()
        policy = RecommendationPolicyConfig(
            prerequisite_mastery_threshold=settings.prerequisite_mastery_threshold,
            policy_version=settings.policy_version,
        )
        return ApplicationController(
            skills=catalogue.skills,
            items=items,
            repository=repository,
            recommendation_service=RecommendationService(
                repository, model_version=settings.model_version, config=policy
            ),
            bkt_service=BKTService(model, repository),
        )
    except BootstrapError:
        LOGGER.exception("Application bootstrap failed")
        raise
    except Exception as exc:
        LOGGER.exception("Application bootstrap failed")
        raise BootstrapError("Application bootstrap failed") from exc
