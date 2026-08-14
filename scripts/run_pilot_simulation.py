"""Drive synthetic learner personas through the real application controller
to seed the pilot database with attempt data, since real learners are not
yet available. Exercises the same recommend/submit path as the Streamlit app.
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

from api.bank import BankItem
from api.presentation import option_id
from app.bootstrap import AppSettings, build_controller, load_approved_bank
from app.controller import ApplicationController, ApplicationError, ContentGapError
from taxonomy.loader import course_paths


@dataclass(frozen=True)
class Persona:
    learner_id: str
    accuracy: float


PERSONAS = [
    Persona("pilot-learner-01", 0.90),
    Persona("pilot-learner-02", 0.75),
    Persona("pilot-learner-03", 0.60),
    Persona("pilot-learner-04", 0.45),
    Persona("pilot-learner-05", 0.30),
    Persona("pilot-learner-06", 0.55),
]


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--database", type=Path, required=True)
    root.add_argument("--bank", type=Path, required=True)
    root.add_argument("--model", type=Path, required=True)
    root.add_argument("--model-version", required=True)
    root.add_argument("--initial-mastery", type=float, default=0.20)
    root.add_argument("--max-attempts-per-learner", type=int, default=40)
    root.add_argument("--seed", type=int, default=20260812)
    root.add_argument("--output", type=Path, default=Path("outputs/pilot_run_summary.json"))
    return root


def run_persona(
    controller: ApplicationController,
    persona: Persona,
    *,
    correct_answers: dict[str, str],
    max_attempts: int,
    rng: random.Random,
) -> dict:
    learner_id = controller.start_learner_session(persona.learner_id)
    excluded_item_ids = list(controller.answered_item_ids(learner_id))
    attempts: list[dict] = []
    stop_reason = "max_attempts_reached"

    for _ in range(max_attempts):
        try:
            question = controller.recommend_question(learner_id, excluded_item_ids)
        except ContentGapError as exc:
            stop_reason = f"content_gap:{exc.content_gap.newly_unlocked_skill_id}"
            break
        except ApplicationError as exc:
            stop_reason = type(exc).__name__
            break

        correct_option = option_id(question.item_id, correct_answers[question.item_id])
        option_ids = [choice.option_id for choice in question.options]
        if rng.random() < persona.accuracy:
            selected = correct_option
        else:
            wrong_options = [oid for oid in option_ids if oid != correct_option]
            selected = rng.choice(wrong_options) if wrong_options else correct_option

        result = controller.submit_answer(learner_id, question.presentation_id, selected)
        excluded_item_ids.append(question.item_id)
        attempts.append(
            {
                "item_id": question.item_id,
                "skill_id": question.skill_id,
                "difficulty": question.difficulty,
                "correct": result.correct,
                "updated_mastery": result.updated_mastery,
            }
        )
    else:
        stop_reason = "max_attempts_reached"

    mastery_by_skill: dict[str, float] = {}
    for attempt in attempts:
        mastery_by_skill[attempt["skill_id"]] = attempt["updated_mastery"]

    return {
        "learner_id": learner_id,
        "accuracy_target": persona.accuracy,
        "attempt_count": len(attempts),
        "correct_count": sum(1 for a in attempts if a["correct"]),
        "stop_reason": stop_reason,
        "final_mastery_by_skill": mastery_by_skill,
        "attempts": attempts,
    }


def main(argv=None) -> int:
    arguments = parser().parse_args(argv)
    skills_path, references_path = course_paths("ai")
    settings = AppSettings(
        database_path=arguments.database,
        approved_bank_path=arguments.bank,
        bkt_model_path=arguments.model,
        skills_path=skills_path,
        references_path=references_path,
        model_version=arguments.model_version,
        policy_version="recommendation-policy-v1",
        initial_mastery_probability=arguments.initial_mastery,
    )
    controller = build_controller(settings)

    bank_items: list[BankItem] = load_approved_bank(arguments.bank)
    correct_answers = {
        item.item_id: item.question.correct_answer
        for item in bank_items
        if item.item_id is not None
    }

    rng = random.Random(arguments.seed)
    results = [
        run_persona(
            controller,
            persona,
            correct_answers=correct_answers,
            max_attempts=arguments.max_attempts_per_learner,
            rng=rng,
        )
        for persona in PERSONAS
    ]

    summary = {
        "database": str(arguments.database),
        "bank": str(arguments.bank),
        "model_version": arguments.model_version,
        "seed": arguments.seed,
        "learners": results,
        "total_attempts": sum(r["attempt_count"] for r in results),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "learners"}, indent=2))
    for result in results:
        print(
            f"{result['learner_id']}: {result['attempt_count']} attempts, "
            f"{result['correct_count']} correct, stopped: {result['stop_reason']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
