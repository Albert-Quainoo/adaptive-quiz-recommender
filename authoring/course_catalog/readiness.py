"""Structural/data readiness gate for a course, ahead of automatic activation.

Every check here is data validation only: no network calls, no test-runner
invocation. "Course-isolation and learner-flow tests pass" is proven by the
project's pytest suite (tests/test_multi_course_runtime.py and friends), not
by this module -- this module never shells out to pytest.

Each check is caught and turned into a blocker line rather than raised, so
inspect_readiness() never raises for an unready course: the CLI must be able
to report unmet requirements and exit cleanly, per the "fail safely" spec
requirement.
"""

import hashlib
from pathlib import Path

from pydantic import BaseModel, Field

from app.bootstrap import load_approved_bank, load_fitted_bkt_model
from authoring.replenishment.manifest import CourseManifest, active_bank_path
from authoring.review.reports import count_pending_review_items
from scripts.validate_approved_bank import validate_bank
from taxonomy.loader import TaxonomyError, load_skills
from taxonomy.schemas import find_skills_missing_reference_material

# Matches app/bootstrap.py's AppSettings default -- recommendation-policy
# configuration (the prior every course's BKT model must be fitted against)
# is a global, env-driven setting, not a per-course manifest field. Readiness
# checks it against this same default unless the caller passes the real
# configured value.
DEFAULT_INITIAL_MASTERY_PROBABILITY = 0.20


class ReadinessReport(BaseModel):
    course_id: str
    is_ready: bool
    blockers: list[str] = Field(default_factory=list)
    taxonomy_version: str | None = None
    approved_bank_version: str | None = None
    bkt_model_version: str | None = None


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inspect_readiness(
    manifest: CourseManifest,
    *,
    initial_mastery_probability: float = DEFAULT_INITIAL_MASTERY_PROBABILITY,
) -> ReadinessReport:
    blockers: list[str] = []
    taxonomy_version: str | None = None
    approved_bank_version: str | None = None
    bkt_model_version: str | None = None

    catalogue = None
    try:
        catalogue = load_skills(manifest.skills_path(), manifest.references_path())
    except TaxonomyError as exc:
        blockers.append(f"taxonomy is invalid: {exc}")
    except OSError as exc:
        blockers.append(f"taxonomy could not be read: {exc}")
    else:
        taxonomy_version = hashlib.sha256(
            manifest.skills_path().read_bytes() + manifest.references_path().read_bytes()
        ).hexdigest()
        missing_references = find_skills_missing_reference_material(catalogue)
        if missing_references:
            blockers.append(
                "skills missing approved reference material: "
                + ", ".join(sorted(missing_references))
            )

    bank_path = active_bank_path(manifest)
    if catalogue is not None:
        try:
            items = load_approved_bank(bank_path)
        except (OSError, ValueError) as exc:
            blockers.append(f"approved bank is unavailable: {exc}")
        else:
            required_skill_ids = sorted(skill.skill_id for skill in catalogue.skills)
            try:
                validate_bank(
                    bank_path,
                    course=manifest.course_id,
                    expected_count=len(items),
                    required_skill_ids=required_skill_ids,
                    skills_path=manifest.skills_path(),
                    references_path=manifest.references_path(),
                )
            except ValueError as exc:
                blockers.append(f"approved bank does not meet coverage requirements: {exc}")
            else:
                approved_bank_version = f"{bank_path.name}:{_file_hash(bank_path)}"

    pending_review_count = count_pending_review_items(manifest.review_store_path)
    if pending_review_count:
        blockers.append(
            f"{pending_review_count} seed question(s) are still awaiting human review"
        )

    if not manifest.bkt_model_path.is_file():
        blockers.append(f"BKT model artifact is missing: {manifest.bkt_model_path}")
    else:
        try:
            model = load_fitted_bkt_model(
                manifest.bkt_model_path,
                model_version=manifest.default_bkt_model_version,
                course_id=manifest.course_id,
            )
        except Exception as exc:  # BootstrapError, or a bare exception from pyBKT itself
            blockers.append(f"BKT model artifact could not be loaded: {exc}")
        else:
            bkt_model_version = manifest.default_bkt_model_version
            model_parameters = model.get_parameters()
            model_skill_ids = set(model_parameters.index.get_level_values("skill"))
            if catalogue is not None:
                bank_skill_ids = {skill.skill_id for skill in catalogue.skills}
                missing_model_skills = sorted(bank_skill_ids - model_skill_ids)
                if missing_model_skills:
                    blockers.append(
                        "BKT model is missing skills: " + ", ".join(missing_model_skills)
                    )
            priors = model_parameters.xs("prior", level="param")["value"]
            if any(
                abs(float(prior) - initial_mastery_probability) > 1e-9 for prior in priors
            ):
                blockers.append(
                    "BKT model prior does not match the configured recommendation "
                    "policy's initial mastery probability"
                )

    return ReadinessReport(
        course_id=manifest.course_id,
        is_ready=not blockers,
        blockers=blockers,
        taxonomy_version=taxonomy_version,
        approved_bank_version=approved_bank_version,
        bkt_model_version=bkt_model_version,
    )
