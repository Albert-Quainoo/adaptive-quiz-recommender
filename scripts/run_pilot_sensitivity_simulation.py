"""Sensitivity simulation for the exhaustion-unlock policy (recommendation/policy.py).

Drives many synthetic personas -- varied ability, per-skill jitter, a small
practice-improves-accuracy curve -- through the real recommend/submit/BKT
pipeline against a throwaway database, and measures how often skills unlock
via true mastery vs. via exhausting a prerequisite's item bank below mastery.

Three independent sweeps, each isolating one variable:
  scale      multiple seeds x personas, at the shipped threshold/params
  threshold  prerequisite_mastery_threshold, at one seed
  parameters BKT guess/slip/learn variants, at one seed

This is sensitivity testing, not training evidence: results should inform
whether more approved items or a different threshold/parameter profile are
worth pursuing -- never used to calibrate the production BKT model.
"""

import argparse
import json
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from api.presentation import option_id
from app.bootstrap import AppSettings, build_controller, load_approved_bank
from app.controller import (
    AllEligibleItemsAttemptedError,
    ApplicationController,
    BankExhaustedBelowMasteryError,
)
from bkt.adapter import PyBKTAdapter
from bkt.model import BKTModel
from bkt.service import BKTService
from bkt.train_dev_model import generate_synthetic_attempts
from pyBKT.models import Model as PyBKTModel
from recommendation.policy import RecommendationPolicyConfig
from recommendation.service import RecommendationService
from recommendation.sqlite_repository import SQLiteRecommendationRepository
from taxonomy.loader import course_paths, load_skills


BASELINE_PARAMETERS = {"prior": 0.20, "learns": 0.04, "guesses": 0.40, "slips": 0.20, "forgets": 0.0}
PARAMETER_VARIANTS = {
    "baseline": BASELINE_PARAMETERS,
    "lower_guess": {**BASELINE_PARAMETERS, "guesses": 0.25},
    "higher_learn": {**BASELINE_PARAMETERS, "learns": 0.10},
    "lower_slip": {**BASELINE_PARAMETERS, "slips": 0.10},
}


def persona_correctness_probability(
    base_ability: float, skill_jitter: float, attempts_on_skill: int
) -> float:
    practice_bonus = min(0.15, 0.02 * attempts_on_skill)
    return float(np.clip(base_ability + skill_jitter + practice_bonus, 0.05, 0.97))


def simulate_persona(
    controller: ApplicationController,
    learner_id: str,
    rng: np.random.Generator,
    *,
    base_ability: float,
    skill_jitter_scale: float,
    correct_answers: dict[str, str],
    max_attempts: int,
) -> dict:
    excluded: list[str] = []
    attempts_on_skill: dict[str, int] = defaultdict(int)
    skill_jitters: dict[str, float] = {}
    stop_reason = "max_attempts_reached"

    for _ in range(max_attempts):
        try:
            question = controller.recommend_question(learner_id, excluded)
        except (BankExhaustedBelowMasteryError, AllEligibleItemsAttemptedError) as exc:
            stop_reason = type(exc).__name__
            break

        skill_jitter = skill_jitters.setdefault(
            question.skill_id,
            float(rng.uniform(-skill_jitter_scale, skill_jitter_scale)),
        )
        probability_correct = persona_correctness_probability(
            base_ability, skill_jitter, attempts_on_skill[question.skill_id]
        )
        correct_option = option_id(question.item_id, correct_answers[question.item_id])
        option_ids = [choice.option_id for choice in question.options]
        if rng.random() < probability_correct:
            selected = correct_option
        else:
            wrong = [oid for oid in option_ids if oid != correct_option]
            selected = str(rng.choice(wrong)) if wrong else correct_option

        controller.submit_answer(learner_id, question.presentation_id, selected)
        excluded.append(question.item_id)
        attempts_on_skill[question.skill_id] += 1
    else:
        stop_reason = "max_attempts_reached"

    return {"learner_id": learner_id, "attempts": len(excluded), "stop_reason": stop_reason}


def classify_unlocks(controller: ApplicationController, learner_id: str, skills_by_id: dict) -> list[tuple[str, str]]:
    """First-occurrence (skill_id, reason) for every gated skill this learner reached."""
    seen: set[str] = set()
    unlocks: list[tuple[str, str]] = []
    for event in controller.repository.list_recommendations(learner_id=learner_id):
        if event.skill_id in seen:
            continue
        seen.add(event.skill_id)
        skill = skills_by_id.get(event.skill_id)
        if skill is not None and skill.prerequisite_skill_ids:
            unlocks.append((event.skill_id, event.reason))
    return unlocks


def run_personas(
    controller: ApplicationController,
    *,
    seed: int,
    persona_count: int,
    scenario_tag: str,
    skills_by_id: dict,
    correct_answers: dict[str, str],
    bank_size: int,
    max_attempts: int,
) -> dict:
    rng = np.random.default_rng(seed)
    persona_results = []
    mastery_unlocks = 0
    exhaustion_unlocks = 0
    exhaustion_unlock_skill_counts: Counter = Counter()
    exhaustion_unlock_final_masteries: list[float] = []
    skill_attempt_counts: Counter = Counter()

    for index in range(persona_count):
        learner_id = f"sens-{scenario_tag}-s{seed}-p{index:04d}"
        base_ability = float(rng.uniform(0.15, 0.95))
        outcome = simulate_persona(
            controller,
            learner_id,
            rng,
            base_ability=base_ability,
            skill_jitter_scale=0.08,
            correct_answers=correct_answers,
            max_attempts=max_attempts,
        )
        outcome["base_ability"] = base_ability
        outcome["full_bank_completed"] = outcome["attempts"] == bank_size
        persona_results.append(outcome)

        for skill_id, reason in classify_unlocks(controller, learner_id, skills_by_id):
            if reason == "prerequisite_exhausted_unlock":
                exhaustion_unlocks += 1
                exhaustion_unlock_skill_counts[skill_id] += 1
                progress = controller.get_progress(learner_id, skill_id)
                exhaustion_unlock_final_masteries.append(progress.mastery_probability)
            else:
                mastery_unlocks += 1

        for attempt in controller.repository.list_attempts(learner_id=learner_id):
            skill_attempt_counts[attempt.skill_id] += 1

    total_unlocks = mastery_unlocks + exhaustion_unlocks
    total_attempts_per_skill = sum(skill_attempt_counts.values())
    distinct_skill_slots = sum(1 for c in skill_attempt_counts.values() if c > 0)

    return {
        "seed": seed,
        "persona_count": persona_count,
        "mastery_unlock_rate": mastery_unlocks / total_unlocks if total_unlocks else None,
        "exhaustion_unlock_rate": exhaustion_unlocks / total_unlocks if total_unlocks else None,
        "total_unlock_events": total_unlocks,
        "average_attempts_per_skill_slot": (
            total_attempts_per_skill / distinct_skill_slots if distinct_skill_slots else None
        ),
        "skills_most_often_exhaustion_unlocked": exhaustion_unlock_skill_counts.most_common(),
        "exhaustion_unlock_final_mastery": {
            "count": len(exhaustion_unlock_final_masteries),
            "mean": (
                float(np.mean(exhaustion_unlock_final_masteries))
                if exhaustion_unlock_final_masteries
                else None
            ),
            "p50": (
                float(np.median(exhaustion_unlock_final_masteries))
                if exhaustion_unlock_final_masteries
                else None
            ),
        },
        "full_bank_completion_rate": sum(p["full_bank_completed"] for p in persona_results) / persona_count,
        "average_attempts_per_persona": sum(p["attempts"] for p in persona_results) / persona_count,
    }


def build_pybkt_variant_model(bank_items, skills, seed: int, parameters: dict) -> BKTModel:
    skill_ids = sorted({item.skill_id for item in bank_items if item.skill_id})
    attempts = generate_synthetic_attempts(skill_ids, seed=seed)
    training_frame = PyBKTAdapter().to_dataframe(attempts)

    model = PyBKTModel(seed=seed, num_fits=1, parallel=False)
    model.coef_ = {
        skill_id: {
            "prior": parameters["prior"],
            "learns": np.array([parameters["learns"]]),
            "guesses": np.array([parameters["guesses"]]),
            "slips": np.array([parameters["slips"]]),
            "forgets": np.array([parameters["forgets"]]),
        }
        for skill_id in skill_ids
    }
    model.fit(data=training_frame, fixed=True)
    return BKTModel(model, model_version="sensitivity-sweep", fitted=True)


def build_manual_controller(
    bank_items, skills, database_path: Path, *, threshold: float, initial_mastery: float, bkt_model: BKTModel
) -> ApplicationController:
    repository = SQLiteRecommendationRepository(database_path, skills=skills, items=bank_items)
    repository.initialize_schema()
    policy = RecommendationPolicyConfig(
        initial_mastery_probability=initial_mastery,
        prerequisite_mastery_threshold=threshold,
        policy_version="sensitivity-sweep",
    )
    return ApplicationController(
        skills=skills,
        items=bank_items,
        repository=repository,
        recommendation_service=RecommendationService(
            repository, model_version="sensitivity-sweep", config=policy
        ),
        bkt_service=BKTService(bkt_model, repository),
    )


def run_scale_sweep(args, bank_items, skills, skills_by_id, correct_answers, work_dir: Path) -> list[dict]:
    settings = AppSettings(
        database_path=work_dir / "scale.sqlite3",
        approved_bank_path=args.bank,
        bkt_model_path=args.model,
        skills_path=args.skills_path,
        references_path=args.references_path,
        model_version=args.model_version,
        policy_version="sensitivity-sweep",
        initial_mastery_probability=args.initial_mastery,
        prerequisite_mastery_threshold=0.75,
    )
    controller = build_controller(settings)
    return [
        run_personas(
            controller,
            seed=seed,
            persona_count=args.scale_personas_per_seed,
            scenario_tag="scale",
            skills_by_id=skills_by_id,
            correct_answers=correct_answers,
            bank_size=len(bank_items),
            max_attempts=args.max_attempts,
        )
        for seed in args.scale_seeds
    ]


def run_threshold_sweep(args, bank_items, skills, skills_by_id, correct_answers, work_dir: Path) -> list[dict]:
    results = []
    for threshold in args.thresholds:
        settings = AppSettings(
            database_path=work_dir / f"threshold-{threshold}.sqlite3",
            approved_bank_path=args.bank,
            bkt_model_path=args.model,
            skills_path=args.skills_path,
            references_path=args.references_path,
            model_version=args.model_version,
            policy_version="sensitivity-sweep",
            initial_mastery_probability=args.initial_mastery,
            prerequisite_mastery_threshold=threshold,
        )
        controller = build_controller(settings)
        result = run_personas(
            controller,
            seed=args.sweep_seed,
            persona_count=args.sweep_personas,
            scenario_tag=f"thr{threshold}",
            skills_by_id=skills_by_id,
            correct_answers=correct_answers,
            bank_size=len(bank_items),
            max_attempts=args.max_attempts,
        )
        result["threshold"] = threshold
        results.append(result)
    return results


def run_parameter_sweep(args, bank_items, skills, skills_by_id, correct_answers, work_dir: Path) -> list[dict]:
    results = []
    for variant_name, parameters in PARAMETER_VARIANTS.items():
        bkt_model = build_pybkt_variant_model(bank_items, skills, args.sweep_seed, parameters)
        controller = build_manual_controller(
            bank_items,
            skills,
            work_dir / f"params-{variant_name}.sqlite3",
            threshold=0.75,
            initial_mastery=args.initial_mastery,
            bkt_model=bkt_model,
        )
        result = run_personas(
            controller,
            seed=args.sweep_seed,
            persona_count=args.sweep_personas,
            scenario_tag=f"param-{variant_name}",
            skills_by_id=skills_by_id,
            correct_answers=correct_answers,
            bank_size=len(bank_items),
            max_attempts=args.max_attempts,
        )
        result["parameter_variant"] = variant_name
        result["parameters"] = parameters
        results.append(result)
    return results


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--bank", type=Path, default=Path("outputs/approved_banks/pilot-approved-bank-38-v1.jsonl"))
    root.add_argument("--model", type=Path, default=Path("outputs/bkt_dev_model_v4.pkl"))
    root.add_argument("--model-version", default="bkt-synthetic-v4")
    root.add_argument("--initial-mastery", type=float, default=0.20)
    root.add_argument("--max-attempts", type=int, default=45)
    root.add_argument("--scale-seeds", type=int, nargs="+", default=[1, 2])
    root.add_argument("--scale-personas-per-seed", type=int, default=120)
    root.add_argument("--thresholds", type=float, nargs="+", default=[0.60, 0.65, 0.70, 0.75, 0.80])
    root.add_argument("--sweep-seed", type=int, default=1)
    root.add_argument("--sweep-personas", type=int, default=50)
    root.add_argument(
        "--stages",
        nargs="+",
        choices=["scale", "threshold", "parameters"],
        default=["scale", "threshold", "parameters"],
    )
    root.add_argument("--output", type=Path, default=Path("outputs/pilot_sensitivity_summary.json"))
    return root


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    skills_path, references_path = course_paths("ai")
    args.skills_path = skills_path
    args.references_path = references_path

    bank_items = load_approved_bank(args.bank)
    catalogue = load_skills(skills_path, references_path)
    skills = catalogue.skills
    skills_by_id = {skill.skill_id: skill for skill in skills}
    correct_answers = {
        item.item_id: item.question.correct_answer for item in bank_items if item.item_id
    }

    summary: dict = {
        "bank": str(args.bank),
        "model_version": args.model_version,
        "bank_size": len(bank_items),
    }
    with tempfile.TemporaryDirectory(prefix="pilot-sensitivity-") as tmp:
        work_dir = Path(tmp)
        if "scale" in args.stages:
            print("Running scale sweep...")
            summary["scale_sweep"] = run_scale_sweep(
                args, bank_items, skills, skills_by_id, correct_answers, work_dir
            )
        if "threshold" in args.stages:
            print("Running threshold sweep...")
            summary["threshold_sweep"] = run_threshold_sweep(
                args, bank_items, skills, skills_by_id, correct_answers, work_dir
            )
        if "parameters" in args.stages:
            print("Running BKT parameter sweep...")
            summary["parameter_sweep"] = run_parameter_sweep(
                args, bank_items, skills, skills_by_id, correct_answers, work_dir
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
