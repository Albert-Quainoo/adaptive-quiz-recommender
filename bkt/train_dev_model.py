"""Fit a development-only pyBKT model from deterministic synthetic responses."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
from pyBKT.models import Model, Roster

from app.bootstrap import load_approved_bank
from bkt.adapter import PyBKTAdapter
from bkt.schemas import AttemptEvent


SYNTHETIC_LEARNER_COUNT = 8
OPPORTUNITY_COUNT = 10
DEVELOPMENT_MODEL_DESCRIPTION = (
    "Development model fitted on synthetic data; it is not learner-trained or "
    "empirically calibrated."
)
MODERATED_PILOT_DESCRIPTION = (
    "Development model using a fixed, reviewed pilot parameter profile; it is "
    "not learner-trained or empirically calibrated."
)
ParameterProfile = Literal["fitted-synthetic", "moderated-pilot"]
MODERATED_PILOT_PARAMETERS = {
    "prior": 0.20,
    "learns": 0.04,
    "guesses": 0.40,
    "slips": 0.20,
    "forgets": 0.0,
}


def generate_synthetic_attempts(
    skill_ids: Sequence[str],
    *,
    seed: int,
    learner_count: int = SYNTHETIC_LEARNER_COUNT,
    opportunity_count: int = OPPORTUNITY_COUNT,
) -> list[AttemptEvent]:
    """Create ordered, reproducible response histories with an improving trend."""

    if learner_count < 2:
        raise ValueError("synthetic training requires at least two learners")
    if opportunity_count < 2:
        raise ValueError("synthetic training requires at least two opportunities")

    rng = np.random.default_rng(seed)
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    attempts: list[AttemptEvent] = []

    for skill_id in sorted(set(skill_ids)):
        for learner_index in range(learner_count):
            learner_id = f"synthetic-learner-{learner_index + 1}"
            learner_adjustment = (learner_index - (learner_count - 1) / 2) * 0.015
            for opportunity in range(1, opportunity_count + 1):
                progress = (opportunity - 1) / (opportunity_count - 1)
                probability_correct = float(
                    np.clip(0.15 + 0.75 * progress + learner_adjustment, 0.05, 0.95)
                )
                correct = bool(rng.random() < probability_correct)

                # Guarantee that every history contains both outcomes while the
                # seeded middle opportunities retain realistic variation.
                if opportunity == 1:
                    correct = False
                elif opportunity == opportunity_count:
                    correct = True

                suffix = f"{skill_id}-{learner_index + 1}-{opportunity}"
                attempts.append(
                    AttemptEvent(
                        attempt_id=f"synthetic-attempt-{suffix}",
                        presentation_id=f"synthetic-presentation-{suffix}",
                        learner_id=learner_id,
                        item_id=f"synthetic-item-{skill_id}",
                        skill_id=skill_id,
                        selected_option_id="synthetic-option",
                        correct=correct,
                        attempt_order=opportunity,
                        occurred_at=start + timedelta(seconds=opportunity),
                    )
                )

    return attempts


def validate_roster_support(model: Model, skill_ids: Sequence[str]) -> None:
    """Prove that every bank skill can initialize and update a loaded Roster."""

    for skill_id in skill_ids:
        learner_id = f"artifact-validation-{skill_id}"
        roster = Roster(students=[learner_id], skills=skill_id, model=model)
        for correct in (0, 1):
            roster.update_state(skill_id, learner_id, correct)
            probability = float(roster.get_mastery_prob(skill_id, learner_id))
            if not 0.0 <= probability <= 1.0:
                raise ValueError(
                    f"loaded model returned invalid mastery for skill {skill_id}"
                )


def mastery_trajectory(
    model: Model, skill_id: str, outcomes: Sequence[bool]
) -> list[float]:
    learner_id = f"trajectory-validation-{skill_id}"
    roster = Roster(students=[learner_id], skills=skill_id, model=model)
    probabilities = [float(roster.get_mastery_prob(skill_id, learner_id))]
    for correct in outcomes:
        roster.update_state(skill_id, learner_id, int(correct))
        probabilities.append(float(roster.get_mastery_prob(skill_id, learner_id)))
    return probabilities


def validate_moderated_trajectories(model: Model, skill_ids: Sequence[str]) -> None:
    for skill_id in skill_ids:
        incorrect_then_correct = mastery_trajectory(model, skill_id, [False, True])
        repeated_correct = mastery_trajectory(model, skill_id, [True, True, True])
        if not incorrect_then_correct[1] < incorrect_then_correct[0]:
            raise ValueError(f"incorrect response did not lower {skill_id} mastery")
        if not incorrect_then_correct[1] < incorrect_then_correct[2] < 0.60:
            raise ValueError(
                f"incorrect-then-correct trajectory is not moderated for {skill_id}"
            )
        if repeated_correct != sorted(repeated_correct):
            raise ValueError(f"correct responses did not increase {skill_id} mastery")
        if repeated_correct[1] > 0.40 or repeated_correct[2] > 0.60:
            raise ValueError(f"early correct responses overstate {skill_id} mastery")
        if repeated_correct[-1] > 0.80:
            raise ValueError(f"three correct responses overstate {skill_id} mastery")


def train_dev_model(
    bank_path: Path,
    output_path: Path,
    metadata_output_path: Path,
    *,
    model_version: str,
    seed: int,
    parameter_profile: ParameterProfile = "fitted-synthetic",
    clock: Callable[[], datetime] | None = None,
) -> dict[str, object]:
    """Fit, save, reload, and validate a synthetic development model artifact."""

    if not model_version.strip():
        raise ValueError("model_version is required")

    bank_items = load_approved_bank(bank_path)
    skill_ids = sorted({item.skill_id for item in bank_items if item.skill_id})
    attempts = generate_synthetic_attempts(skill_ids, seed=seed)
    training_frame = PyBKTAdapter().to_dataframe(attempts)

    model = Model(seed=seed, num_fits=1, parallel=False)
    if parameter_profile == "moderated-pilot":
        model.coef_ = {
            skill_id: {
                "prior": MODERATED_PILOT_PARAMETERS["prior"],
                "learns": np.array([MODERATED_PILOT_PARAMETERS["learns"]]),
                "guesses": np.array([MODERATED_PILOT_PARAMETERS["guesses"]]),
                "slips": np.array([MODERATED_PILOT_PARAMETERS["slips"]]),
                "forgets": np.array([MODERATED_PILOT_PARAMETERS["forgets"]]),
            }
            for skill_id in skill_ids
        }
        model.fit(data=training_frame, fixed=True)
    else:
        model.fit(data=training_frame)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    loaded_model = Model(parallel=False)
    loaded_model.load(str(output_path))
    if loaded_model.fit_model is None:
        raise ValueError("saved pyBKT development model is not fitted")
    validate_roster_support(loaded_model, skill_ids)
    if parameter_profile == "moderated-pilot":
        validate_moderated_trajectories(loaded_model, skill_ids)

    created_at = (clock or (lambda: datetime.now(timezone.utc)))()
    metadata: dict[str, object] = {
        "model_version": model_version.strip(),
        "pybkt_version": version("pyBKT"),
        "seed": seed,
        "created_at": created_at.isoformat(),
        "training_source": "synthetic",
        "included_skill_ids": skill_ids,
        "synthetic_learner_count": SYNTHETIC_LEARNER_COUNT,
        "opportunity_count": OPPORTUNITY_COUNT,
        "model_artifact_path": str(output_path),
        "description": (
            MODERATED_PILOT_DESCRIPTION
            if parameter_profile == "moderated-pilot"
            else DEVELOPMENT_MODEL_DESCRIPTION
        ),
        "parameter_profile": parameter_profile,
        "parameters": (
            MODERATED_PILOT_PARAMETERS
            if parameter_profile == "moderated-pilot"
            else None
        ),
    }
    temporary_metadata_path = metadata_output_path.with_suffix(
        metadata_output_path.suffix + ".tmp"
    )
    temporary_metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary_metadata_path.replace(metadata_output_path)
    return metadata


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=DEVELOPMENT_MODEL_DESCRIPTION,
    )
    parser.add_argument("--bank", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    parser.add_argument("--model-version", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument(
        "--parameter-profile",
        choices=("fitted-synthetic", "moderated-pilot"),
        default="fitted-synthetic",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    metadata = train_dev_model(
        arguments.bank,
        arguments.output,
        arguments.metadata_output,
        model_version=arguments.model_version,
        seed=arguments.seed,
        parameter_profile=arguments.parameter_profile,
    )
    print(metadata["description"])
    print(f"Saved model artifact to {metadata['model_artifact_path']}")
    print(f"Saved metadata to {arguments.metadata_output}")


if __name__ == "__main__":
    main()
