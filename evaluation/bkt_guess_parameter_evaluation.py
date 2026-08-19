"""Offline synthetic behavioral evaluation: BKT guesses parameter 0.40 -> 0.25
(commit 7fa7bed, "Correct BKT guess parameter to match true chance rate").

This is behavioral validation, not proof that guesses=0.25 is statistically optimal --
it compares how the currently-committed "prior" (guesses=0.40) and "candidate"
(guesses=0.25) BKT artifacts respond to identical, fixed synthetic attempt sequences.

Model pins, per course (see MODEL_PAIRS below):
- intro-ai: prior=outputs/bkt_dev_model_v4.pkl (git-committed), candidate=
  outputs/bkt_dev_model_v6.pkl (git-committed). Both share seed=20260811 and the same
  6 original skills; v6 additionally covers AI-FND-03/AI-FND-04 (added after v4 was
  built), reported separately with no prior counterpart.
- dsa/linear-algebra/database-systems: prior=...*_v1.pkl (present on disk, NOT
  git-tracked as of commit 7fa7bed -- "Superseded versions (v5, and each course's v1)
  are untracked now"), candidate=...*_v2.pkl (git-committed). v1 was trained with a
  different seed than v2 (see each course's *_v1.metadata.json), but since
  "moderated-pilot" training uses fixed=True (coefficients pinned directly, never
  re-estimated from the training frame), the actual fitted prior/learns/slips/forgets
  are byte-identical between v1 and v2 for every course -- verified against the
  committed/on-disk metadata JSON before relying on it. guesses is the only
  parameter that differs. This is documented as a limitation in the report regardless.

Uses only disposable artifacts: every DB is a fresh SQLite/PostgreSQL scratch file
this script creates and the caller is responsible for discarding; nothing here writes
to Supabase, active-bank pointers, course state, production BKT config, or any
replenishment branch.

    python -m evaluation.bkt_guess_parameter_evaluation --output <path>.json
"""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from pyBKT.models import Model as PyBKTModel
from pyBKT.models import Roster

from api.bank import BankItem
from api.presentation import present_bank_item
from app.bootstrap import load_approved_bank, load_fitted_bkt_model
from authoring.replenishment.manifest import active_bank_path, load_course_manifest
from bkt import AttemptEvent, BKTModel, BKTService, SQLiteBKTRepository
from recommendation import RecommendationPolicyConfig, RecommendationRequest, RecommendationService
from recommendation.sqlite_repository import SQLiteRecommendationRepository
from taxonomy.loader import load_skills

REPO_ROOT = Path(__file__).resolve().parent.parent

# RecommendationPolicyConfig defaults -- the thresholds "threshold-crossing attempt"
# below is measured against.
INTRODUCTORY_THRESHOLD = 0.40
ADVANCED_THRESHOLD = 0.75

SEQUENCE_LENGTH = 10
FIXED_CLOCK_START = datetime(2026, 8, 17, 12, 0, tzinfo=timezone.utc)
PERSONA_SEEDS = {"improving": 90211, "declining": 90212}
PERSONAS = [
    "consistently_correct",
    "consistently_incorrect",
    "improving",
    "declining",
    "alternating",
    "lucky_guesser",
]

MODEL_PAIRS: dict[str, dict[str, dict[str, object]]] = {
    "intro-ai": {
        "prior": {"path": "outputs/bkt_dev_model_v4.pkl", "version": "bkt-synthetic-v4", "committed": True},
        "candidate": {"path": "outputs/bkt_dev_model_v6.pkl", "version": "bkt-synthetic-v6", "committed": True},
    },
    "dsa": {
        "prior": {"path": "outputs/bkt_dsa_model_v1.pkl", "version": "bkt-dsa-v1", "committed": False},
        "candidate": {"path": "outputs/bkt_dsa_model_v2.pkl", "version": "bkt-dsa-v2", "committed": True},
    },
    "linear-algebra": {
        "prior": {"path": "outputs/bkt_linear_algebra_model_v1.pkl", "version": "bkt-linear-algebra-v1", "committed": False},
        "candidate": {"path": "outputs/bkt_linear_algebra_model_v2.pkl", "version": "bkt-linear-algebra-v2", "committed": True},
    },
    "database-systems": {
        "prior": {"path": "outputs/bkt_database_systems_model_v1.pkl", "version": "bkt-database-systems-v1", "committed": False},
        "candidate": {"path": "outputs/bkt_database_systems_model_v2.pkl", "version": "bkt-database-systems-v2", "committed": True},
    },
}


def persona_outcomes(persona: str) -> list[bool]:
    if persona == "consistently_correct":
        return [True] * SEQUENCE_LENGTH
    if persona == "consistently_incorrect":
        return [False] * SEQUENCE_LENGTH
    if persona == "alternating":
        return [index % 2 == 0 for index in range(SEQUENCE_LENGTH)]
    if persona == "lucky_guesser":
        # Mostly wrong, with two isolated correct answers -- simulates guessing right
        # by chance rather than genuine mastery. guesses=0.25 (vs 0.40) means the model
        # should attribute an unexpected correct answer *more* to real learning, not
        # less, since correct-by-guessing is a priori less likely under a lower guess
        # rate -- this persona is the most direct behavioral probe of the actual change.
        outcomes = [False] * SEQUENCE_LENGTH
        outcomes[2] = True
        outcomes[6] = True
        return outcomes
    if persona in PERSONA_SEEDS:
        rng = np.random.default_rng(PERSONA_SEEDS[persona])
        outcomes = []
        for index in range(SEQUENCE_LENGTH):
            progress = index / (SEQUENCE_LENGTH - 1)
            probability = 0.1 + 0.8 * progress if persona == "improving" else 0.9 - 0.8 * progress
            outcomes.append(bool(rng.random() < probability))
        outcomes[0] = persona == "declining"
        outcomes[-1] = persona == "improving"
        return outcomes
    raise ValueError(f"unknown persona {persona!r}")


def load_pybkt_model(path: Path) -> PyBKTModel:
    model = PyBKTModel(parallel=False)
    model.load(str(path))
    return model


def mastery_trajectory(model: PyBKTModel, skill_id: str, outcomes: list[bool], *, tag: str) -> list[float]:
    learner_id = f"eval-{tag}"
    roster = Roster(students=[learner_id], skills=skill_id, model=model)
    trajectory = [float(roster.get_mastery_prob(skill_id, learner_id))]
    for correct in outcomes:
        roster.update_state(skill_id, learner_id, int(correct))
        trajectory.append(float(roster.get_mastery_prob(skill_id, learner_id)))
    return trajectory


def threshold_crossing(trajectory: list[float], threshold: float) -> int | None:
    for attempt_index, mastery in enumerate(trajectory):
        if mastery >= threshold:
            return attempt_index
    return None


@dataclass
class TrajectoryResult:
    course_id: str
    skill_id: str
    persona: str
    model_role: str  # "prior" | "candidate"
    model_version: str
    trajectory: list[float]
    final_mastery: float
    introductory_crossing_attempt: int | None
    advanced_crossing_attempt: int | None
    deterministic: bool


def run_trajectory_comparison() -> list[TrajectoryResult]:
    results: list[TrajectoryResult] = []
    for course_id, roles in MODEL_PAIRS.items():
        loaded = {}
        included_skills = {}
        for role, info in roles.items():
            path = REPO_ROOT / info["path"]
            metadata_path = path.with_suffix("") if False else Path(str(path)[: -len(".pkl")] + ".metadata.json")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            loaded[role] = load_pybkt_model(path)
            included_skills[role] = set(metadata["included_skill_ids"])

        shared_skills = sorted(included_skills["prior"] & included_skills["candidate"])
        candidate_only_skills = sorted(included_skills["candidate"] - included_skills["prior"])

        for skill_id in shared_skills + candidate_only_skills:
            roles_to_run = roles.keys() if skill_id in shared_skills else ["candidate"]
            for persona in PERSONAS:
                outcomes = persona_outcomes(persona)
                for role in roles_to_run:
                    tag = f"{course_id}-{skill_id}-{persona}-{role}"
                    first = mastery_trajectory(loaded[role], skill_id, outcomes, tag=f"{tag}-run1")
                    second = mastery_trajectory(loaded[role], skill_id, outcomes, tag=f"{tag}-run2")
                    results.append(
                        TrajectoryResult(
                            course_id=course_id,
                            skill_id=skill_id,
                            persona=persona,
                            model_role=role,
                            model_version=str(roles[role]["version"]),
                            trajectory=first,
                            final_mastery=first[-1],
                            introductory_crossing_attempt=threshold_crossing(first, INTRODUCTORY_THRESHOLD),
                            advanced_crossing_attempt=threshold_crossing(first, ADVANCED_THRESHOLD),
                            deterministic=(first == second),
                        )
                    )
    return results


# --- Part B: recommendation ordering, fallback behavior, and course isolation ------


@dataclass
class RecommendationStep:
    order: int
    skill_id: str
    item_id: str
    difficulty: str
    mastery_probability: float
    reason: str


def _build_course_repository(
    database_path: Path, course_id: str, model_role_paths: dict[str, object]
) -> tuple[SQLiteRecommendationRepository, BankItem]:
    manifest = load_course_manifest(course_id)
    skills = load_skills(manifest.taxonomy_path / "skills.csv", manifest.taxonomy_path / "references.csv").skills
    items = load_approved_bank(active_bank_path(manifest))
    repository = SQLiteRecommendationRepository(database_path, course_id=course_id, skills=skills, items=items)
    repository.initialize_schema()
    representative_item = next(item for item in items if item.skill_id == skills[0].skill_id)
    return repository, representative_item


def _feed_persona(
    repository: SQLiteBKTRepository,
    model_path: Path,
    model_version: str,
    course_id: str,
    learner_id: str,
    item: BankItem,
    persona: str,
) -> None:
    engine = load_fitted_bkt_model(model_path, model_version=model_version, course_id=course_id)
    service = BKTService(engine, repository)
    for attempt_index, correct in enumerate(persona_outcomes(persona), start=1):
        attempt_id = f"{learner_id}-{item.skill_id}-{attempt_index}"
        presentation = present_bank_item(item, learner_id=learner_id, attempt_id=attempt_id)
        correct_option = next(
            option for option in presentation.presented_options if option.value == item.question.correct_answer
        )
        wrong_option = next(
            option for option in presentation.presented_options if option.value != item.question.correct_answer
        )
        chosen = correct_option if correct else wrong_option
        attempt = AttemptEvent(
            attempt_id=attempt_id,
            course_id=course_id,
            presentation_id=presentation.presentation_id,
            learner_id=learner_id,
            item_id=item.item_id,
            skill_id=item.skill_id,
            selected_option_id=chosen.option_id,
            correct=correct,  # overwritten by BKTService._score_attempt's own scoring
            attempt_order=attempt_index,
            occurred_at=FIXED_CLOCK_START + timedelta(seconds=attempt_index),
        )
        service.process_attempt(attempt, item=item, presentation=presentation)


def run_recommendation_ordering_and_isolation(database_path: Path) -> dict[str, object]:
    shared_learner_id = "cross-course-shared-learner"
    ordering_by_course: dict[str, dict[str, list[dict]]] = {}
    isolation_findings: list[dict] = []
    course_repositories: dict[str, tuple[SQLiteRecommendationRepository, BankItem]] = {}

    for course_id, roles in MODEL_PAIRS.items():
        repository, representative_item = _build_course_repository(database_path, course_id, roles)
        course_repositories[course_id] = (repository, representative_item)
        ordering_by_course[course_id] = {}

        for role in ("prior", "candidate"):
            info = roles[role]
            if role == "prior" and not bool(info["committed"]):
                # Still run it -- the file exists on disk -- but the ordering result is
                # labeled accordingly in the report.
                pass
            model_path = REPO_ROOT / info["path"]
            if not model_path.is_file():
                continue
            learner_id = f"{course_id}-{role}-learner"
            _feed_persona(
                repository, model_path, str(info["version"]), course_id, learner_id,
                representative_item, "improving",
            )
            service = RecommendationService(
                repository, course_id=course_id, model_version=str(info["version"]),
                config=RecommendationPolicyConfig(),
            )
            steps = []
            excluded: list[str] = []
            available_skill_ids = [skill.skill_id for skill in repository.list_skills()]
            for order in range(1, min(6, len(available_skill_ids) + 1)):
                try:
                    result = service.recommend(
                        RecommendationRequest(
                            learner_id=learner_id,
                            available_skill_ids=available_skill_ids,
                            excluded_item_ids=excluded,
                        )
                    )
                except Exception as exc:  # RecommendationUnavailable once exhausted
                    steps.append({"order": order, "exhausted": True, "error": str(exc)})
                    break
                steps.append(
                    asdict(
                        RecommendationStep(
                            order=order, skill_id=result.skill_id, item_id=result.item_id,
                            difficulty=result.difficulty, mastery_probability=result.mastery_probability,
                            reason=result.reason,
                        )
                    )
                )
                excluded.append(result.item_id)
            ordering_by_course[course_id][role] = steps

    # Course isolation, real positive + negative controls: feed one real attempt for
    # `shared_learner_id` into exactly one course (intro-ai, candidate model), then
    # confirm every OTHER course-scoped repository -- backed by the same physical
    # SQLite file -- sees nothing for that learner_id (negative control), while
    # intro-ai's own repository does (positive control, proving the check itself is
    # capable of detecting a leak rather than trivially passing).
    origin_course_id = "intro-ai"
    origin_repository, origin_item = course_repositories[origin_course_id]
    origin_model_path = REPO_ROOT / MODEL_PAIRS[origin_course_id]["candidate"]["path"]
    _feed_persona(
        origin_repository, origin_model_path, str(MODEL_PAIRS[origin_course_id]["candidate"]["version"]),
        origin_course_id, shared_learner_id, origin_item, "consistently_correct",
    )
    for course_id, (repository, representative_item) in course_repositories.items():
        attempts_here = repository.list_attempts(learner_id=shared_learner_id, skill_id=origin_item.skill_id)
        mastery_here = repository.get_mastery(shared_learner_id, origin_item.skill_id)
        isolation_findings.append(
            {
                "course_id": course_id,
                "is_origin_course": course_id == origin_course_id,
                "shared_learner_id_has_attempts_here": bool(attempts_here),
                "shared_learner_id_has_mastery_here": mastery_here is not None,
            }
        )

    return {"recommendation_ordering": ordering_by_course, "cross_course_isolation": isolation_findings}


def _fixed_coefficient_model(skill_id: str, *, guesses: float, seed: int) -> PyBKTModel:
    """Mirrors tests/test_multi_course_runtime.py's _fixed_coefficient_model helper --
    a tiny moderated-pilot-profile model fitted only for the given synthetic skill, so
    the collision check below can use a model that actually knows the collision
    skill_id (a real course's committed model never trained on a fabricated skill)."""
    from bkt.adapter import PyBKTAdapter
    from bkt.train_dev_model import MODERATED_PILOT_PARAMETERS, generate_synthetic_attempts

    attempts = generate_synthetic_attempts([skill_id], seed=seed)
    training_frame = PyBKTAdapter().to_dataframe(attempts)
    model = PyBKTModel(seed=seed, num_fits=1, parallel=False)
    model.coef_ = {
        skill_id: {
            "prior": MODERATED_PILOT_PARAMETERS["prior"],
            "learns": np.array([MODERATED_PILOT_PARAMETERS["learns"]]),
            "guesses": np.array([guesses]),
            "slips": np.array([MODERATED_PILOT_PARAMETERS["slips"]]),
            "forgets": np.array([MODERATED_PILOT_PARAMETERS["forgets"]]),
        }
    }
    model.fit(data=training_frame, fixed=True)
    return model


def run_colliding_skill_id_check(database_path: Path) -> dict[str, object]:
    """Deliberately reuses the exact same skill_id string across two synthetic
    course_ids in the same physical database -- the adversarial boundary case --
    while ALSO using it to test prior (guesses=0.40) vs candidate (guesses=0.25)
    behavior at that boundary: course A gets the prior guess rate, course B gets the
    candidate rate, both trained on the same synthetic data otherwise, both fed the
    identical attempt outcome."""
    from taxonomy.schemas import SkillDefinition

    collision_skill_id = "AA-CLSN-01"
    skill = SkillDefinition(
        skill_id=collision_skill_id, topic="t", subtopic="t", name="n",
        learning_objective="o", cognitive_process="remember", generation_strategy="hand_authored",
    )
    item = BankItem.model_validate(
        {
            "item_id": "collision-item-1", "provenance": "generated", "skill_id": collision_skill_id,
            "question": {
                "question": "collision?", "options": ["A", "B", "C", "D"],
                "correct_answer": "A", "explanation": "because", "concept": "c", "difficulty": "introductory",
            },
        }
    )
    findings = {}
    for course_id, guesses, correct_first in (
        ("collision-course-a-prior-guess", 0.40, True),
        ("collision-course-b-candidate-guess", 0.25, True),
    ):
        repository = SQLiteRecommendationRepository(database_path, course_id=course_id, skills=[skill], items=[item])
        repository.initialize_schema()
        pybkt_model = _fixed_coefficient_model(collision_skill_id, guesses=guesses, seed=1)
        engine = BKTModel(pybkt_model, course_id=course_id, model_version=f"collision-check-guesses-{guesses}", fitted=True)
        service = BKTService(engine, repository)
        learner_id = "collision-learner"
        presentation = present_bank_item(item, learner_id=learner_id, attempt_id=f"{course_id}-1")
        correct_option = next(
            option for option in presentation.presented_options if option.value == item.question.correct_answer
        )
        wrong_option = next(
            option for option in presentation.presented_options if option.value != item.question.correct_answer
        )
        attempt = AttemptEvent(
            attempt_id=f"{course_id}-1", course_id=course_id, presentation_id=presentation.presentation_id,
            learner_id=learner_id, item_id=item.item_id, skill_id=collision_skill_id,
            selected_option_id=(correct_option if correct_first else wrong_option).option_id,
            correct=correct_first, attempt_order=1, occurred_at=FIXED_CLOCK_START,
        )
        snapshot = service.process_attempt(attempt, item=item, presentation=presentation)
        findings[course_id] = {
            "guesses": guesses,
            "mastery_after_attempt": snapshot.mastery_probability,
            "attempt_correct": correct_first,
        }

    course_a_id, course_b_id = "collision-course-a-prior-guess", "collision-course-b-candidate-guess"
    repo_a = SQLiteRecommendationRepository(database_path, course_id=course_a_id, skills=[skill], items=[item])
    findings["course_a_sees_course_b_attempts"] = bool(
        [a for a in repo_a.list_attempts(learner_id="collision-learner", skill_id=collision_skill_id) if a.course_id != course_a_id]
    )
    findings["masteries_differ_by_guess_parameter_despite_identical_skill_id_and_outcome"] = (
        findings[course_a_id]["mastery_after_attempt"] != findings[course_b_id]["mastery_after_attempt"]
    )
    return findings


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    trajectory_results = run_trajectory_comparison()

    with tempfile.TemporaryDirectory(prefix="bkt_eval_") as tmp_dir:
        database_path = Path(tmp_dir) / "disposable.sqlite3"
        recommendation_findings = run_recommendation_ordering_and_isolation(database_path)
        collision_findings = run_colliding_skill_id_check(database_path)

    report = {
        "model_pairs": MODEL_PAIRS,
        "thresholds": {"introductory": INTRODUCTORY_THRESHOLD, "advanced": ADVANCED_THRESHOLD},
        "trajectories": [asdict(result) for result in trajectory_results],
        "recommendation_and_isolation": recommendation_findings,
        "colliding_skill_id_check": collision_findings,
    }

    non_deterministic = [r for r in trajectory_results if not r.deterministic]
    print(f"Trajectories computed: {len(trajectory_results)}")
    print(f"Non-deterministic trajectories: {len(non_deterministic)}")
    print(f"Cross-course isolation findings: {recommendation_findings['cross_course_isolation']}")
    print(f"Colliding skill_id check: {collision_findings}")

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        print(f"\nFull report written to {arguments.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
