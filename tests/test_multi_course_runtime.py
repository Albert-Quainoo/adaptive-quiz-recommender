"""Proves two things end to end, using the real lazy CourseCatalogController
(not a bypass): (1) course isolation holds even when two courses
deliberately reuse the exact same skill_id/item_id strings -- the boundary
is course_id, never skill-id/item-id naming; (2) course runtimes are loaded
lazily, cached per course_id, and a preparing course is never built.
"""

import time
from pathlib import Path

import numpy as np
import pytest
from pyBKT.models import Model as PyBKTModel

import app.multi_course as multi_course_module
from app.bootstrap import AppSettings
from app.flow import activate_course, activate_learner
from app.multi_course import ActiveCourse, UnrecognizedCourse, build_course_catalog
from app.session import get_session_state, retain_question
from authoring.replenishment.manifest import CourseManifest
from bkt.adapter import PyBKTAdapter
from bkt.train_dev_model import MODERATED_PILOT_PARAMETERS, generate_synthetic_attempts

COLLIDING_SKILL_ID = "AA-PLC-01"
COLLIDING_ITEM_ID = "shared-item-1"


def _fixed_coefficient_model(skill_ids: list[str], *, seed: int) -> PyBKTModel:
    attempts = generate_synthetic_attempts(skill_ids, seed=seed)
    training_frame = PyBKTAdapter().to_dataframe(attempts)
    model = PyBKTModel(seed=seed, num_fits=1, parallel=False)
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
    return model


def _build_active_course(
    tmp_path: Path,
    *,
    course_id: str,
    skill_id: str,
    item_id: str,
    seed: int,
    aliases: tuple[str, ...] = (),
) -> CourseManifest:
    course_dir = tmp_path / course_id
    taxonomy_dir = course_dir / "taxonomy"
    taxonomy_dir.mkdir(parents=True)
    (taxonomy_dir / "skills.csv").write_text(
        "skill_id,topic,subtopic,name,learning_objective,cognitive_process,"
        "generation_strategy,template_id,prerequisite_skill_ids\n"
        f"{skill_id},Topic,Subtopic,Name,Objective,remember,hand_authored,,\n",
        encoding="utf-8",
    )
    (taxonomy_dir / "references.csv").write_text("skill_id,reference_material\n", encoding="utf-8")

    bank_path = course_dir / "bank.jsonl"
    item = {
        "item_id": item_id,
        "provenance": "generated",
        "skill_id": skill_id,
        "question": {
            "question": f"What is {course_id}?",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "explanation": "Because.",
            "concept": course_id,
            "difficulty": "introductory",
        },
    }
    bank_path.write_text(__import__("json").dumps(item) + "\n", encoding="utf-8")

    model_path = course_dir / "model.pkl"
    _fixed_coefficient_model([skill_id], seed=seed).save(str(model_path))

    return CourseManifest(
        course_id=course_id,
        title=course_id.upper(),
        version="1",
        taxonomy_path=taxonomy_dir,
        approved_bank_path=bank_path,
        bkt_model_path=model_path,
        candidate_store_path=course_dir / "candidates.json",
        review_store_path=course_dir / "reviews",
        allowed_domains=("example.edu",),
        low_supply_threshold=3,
        target_supply=6,
        default_bkt_model_version=f"{course_id}-v1",
        status="active",
        aliases=aliases or (course_id.upper(),),
    )


def _patch_manifests(monkeypatch, manifests):
    monkeypatch.setattr(
        multi_course_module,
        "load_active_manifests",
        lambda: [m for m in manifests if m.status == "active"],
    )
    monkeypatch.setattr(multi_course_module, "load_all_manifests", lambda: manifests)

    def _load_one(course_id):
        try:
            return next(m for m in manifests if m.course_id == course_id)
        except StopIteration:
            from authoring.replenishment.manifest import ManifestError

            raise ManifestError(f"no course manifest for {course_id!r}") from None

    monkeypatch.setattr(multi_course_module, "load_course_manifest", _load_one)


def _settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        database_path=tmp_path / "shared.sqlite3",
        approved_bank_path=Path("unused"),
        bkt_model_path=Path("unused"),
        skills_path=Path("unused"),
        references_path=Path("unused"),
        model_version="unused",
        policy_version="test-policy-v1",
    )


@pytest.fixture
def colliding_catalogue(tmp_path, monkeypatch):
    """Two courses that deliberately share the exact same skill_id AND
    item_id string -- if isolation ever relied on those strings being
    unique, this fixture would immediately expose it."""
    manifest_a = _build_active_course(
        tmp_path,
        course_id="course-a",
        skill_id=COLLIDING_SKILL_ID,
        item_id=COLLIDING_ITEM_ID,
        seed=1,
    )
    manifest_b = _build_active_course(
        tmp_path,
        course_id="course-b",
        skill_id=COLLIDING_SKILL_ID,
        item_id=COLLIDING_ITEM_ID,
        seed=2,
    )
    _patch_manifests(monkeypatch, [manifest_a, manifest_b])
    return build_course_catalog(_settings(tmp_path))


def _answer_one_question(controller, session):
    question = controller.recommend_question(session.learner_id, session.seen_item_ids)
    retain_question(session, question)
    return controller.submit_answer(
        session.learner_id, question.presentation_id, question.options[0].option_id
    )


def test_colliding_skill_and_item_ids_still_isolate_attempts_and_mastery(
    colliding_catalogue,
):
    controller_a = colliding_catalogue.resolve_active("course-a")
    controller_b = colliding_catalogue.resolve_active("course-b")
    assert controller_a.course_id == "course-a"
    assert controller_b.course_id == "course-b"

    session = get_session_state({})
    activate_learner(controller_a, session, "shared-learner")
    activate_course(controller_a, session, "course-a")
    _answer_one_question(controller_a, session)

    # Raw repository rows for the *same* skill_id/item_id string exist under
    # both course_id values -- this is the setup that would corrupt data
    # under naming-convention-based isolation.
    a_attempts = controller_a.repository.list_attempts(
        learner_id="shared-learner", skill_id=COLLIDING_SKILL_ID
    )
    assert len(a_attempts) == 1
    assert a_attempts[0].course_id == "course-a"

    b_attempts = controller_b.repository.list_attempts(
        learner_id="shared-learner", skill_id=COLLIDING_SKILL_ID
    )
    assert b_attempts == []

    a_mastery = controller_a.repository.get_mastery("shared-learner", COLLIDING_SKILL_ID)
    assert a_mastery is not None
    assert a_mastery.course_id == "course-a"

    b_mastery = controller_b.repository.get_mastery("shared-learner", COLLIDING_SKILL_ID)
    assert b_mastery is None

    assert controller_a.answered_item_ids("shared-learner") == [COLLIDING_ITEM_ID]
    assert controller_b.answered_item_ids("shared-learner") == []


def test_colliding_courses_have_independent_session_rows(colliding_catalogue):
    controller_a = colliding_catalogue.resolve_active("course-a")
    controller_b = colliding_catalogue.resolve_active("course-b")

    session = get_session_state({})
    activate_learner(controller_a, session, "shared-learner")
    activate_course(controller_a, session, "course-a")

    assert controller_a.repository.learner_exists("shared-learner")
    assert not controller_b.repository.learner_exists("shared-learner")

    activate_course(controller_b, session, "course-b")
    assert controller_b.repository.learner_exists("shared-learner")


def test_now_answering_in_course_b_does_not_touch_course_a(colliding_catalogue):
    controller_a = colliding_catalogue.resolve_active("course-a")
    controller_b = colliding_catalogue.resolve_active("course-b")

    session = get_session_state({})
    activate_learner(controller_a, session, "shared-learner")
    activate_course(controller_b, session, "course-b")
    _answer_one_question(controller_b, session)

    assert controller_b.repository.list_attempts(
        learner_id="shared-learner", skill_id=COLLIDING_SKILL_ID
    )
    assert (
        controller_a.repository.list_attempts(
            learner_id="shared-learner", skill_id=COLLIDING_SKILL_ID
        )
        == []
    )
    assert controller_a.repository.get_mastery("shared-learner", COLLIDING_SKILL_ID) is None


def test_resolve_for_learner_returns_unrecognized_for_an_unknown_query(colliding_catalogue):
    assert isinstance(colliding_catalogue.resolve_for_learner("Astrophysics"), UnrecognizedCourse)


def test_resolve_for_learner_returns_active_by_alias(colliding_catalogue):
    result = colliding_catalogue.resolve_for_learner("COURSE-A")
    assert isinstance(result, ActiveCourse)
    assert result.course_id == "course-a"


# --- Lazy loading -----------------------------------------------------


def test_catalog_construction_builds_no_controller(tmp_path, monkeypatch):
    manifest = _build_active_course(
        tmp_path, course_id="course-a", skill_id="AA-PLC-01", item_id="item-1", seed=1
    )
    _patch_manifests(monkeypatch, [manifest])
    catalogue = build_course_catalog(_settings(tmp_path))
    assert catalogue._controllers == {}


def test_listing_courses_builds_no_controller(colliding_catalogue):
    colliding_catalogue.list_active_courses()
    colliding_catalogue.list_preparing_courses()
    assert colliding_catalogue._controllers == {}


def test_resolve_active_caches_the_built_controller(colliding_catalogue):
    first = colliding_catalogue.resolve_active("course-a")
    second = colliding_catalogue.resolve_active("course-a")
    assert first is second
    assert list(colliding_catalogue._controllers) == ["course-a"]


def test_touching_one_course_never_builds_the_other(colliding_catalogue):
    colliding_catalogue.resolve_active("course-a")
    assert "course-b" not in colliding_catalogue._controllers


def test_a_preparing_course_is_never_built(tmp_path, monkeypatch):
    active = _build_active_course(
        tmp_path, course_id="course-a", skill_id="AA-PLC-01", item_id="item-1", seed=1
    )
    preparing = active.model_copy(
        update={"course_id": "course-b", "status": "preparing", "aliases": ("COURSE-B",)}
    )
    _patch_manifests(monkeypatch, [active, preparing])
    catalogue = build_course_catalog(_settings(tmp_path))

    result = catalogue.resolve_for_learner("COURSE-B")
    assert result.__class__.__name__ == "UnavailableCourse"
    assert "course-b" not in catalogue._controllers

    with pytest.raises(KeyError):
        catalogue.resolve_active("course-b")
    assert "course-b" not in catalogue._controllers


def test_switching_courses_does_not_leak_questions_answered_ids_or_mastery(
    colliding_catalogue,
):
    controller_a = colliding_catalogue.resolve_active("course-a")
    controller_b = colliding_catalogue.resolve_active("course-b")

    session = get_session_state({})
    activate_learner(controller_a, session, "switcher")
    activate_course(controller_a, session, "course-a")
    _answer_one_question(controller_a, session)
    assert session.question is not None
    assert controller_a.answered_item_ids("switcher") == [COLLIDING_ITEM_ID]

    activate_course(controller_b, session, "course-b")
    # Switching courses resets the transient quiz state and recomputes
    # seen_item_ids from *this* course's controller only -- course-b has no
    # history for this learner yet, even though course-a's answered item
    # has the exact same item_id string.
    assert session.question is None
    assert session.seen_item_ids == []
    assert session.course_id == "course-b"
    assert controller_b.answered_item_ids("switcher") == []


# --- Startup timing: 1 active course vs 4 active courses --------------


def test_startup_cost_is_independent_of_active_course_count(tmp_path, monkeypatch):
    one_course = [
        _build_active_course(
            tmp_path / "one", course_id="solo", skill_id="SO-PLC-01", item_id="i1", seed=1
        )
    ]
    four_courses = [
        _build_active_course(
            tmp_path / "four",
            course_id=f"course-{i}",
            skill_id=f"C{i}-PLC-01",
            item_id=f"item-{i}",
            seed=i,
        )
        for i in range(1, 5)
    ]

    _patch_manifests(monkeypatch, one_course)
    t0 = time.perf_counter()
    build_course_catalog(_settings(tmp_path / "one"))
    one_course_ms = (time.perf_counter() - t0) * 1000

    _patch_manifests(monkeypatch, four_courses)
    t0 = time.perf_counter()
    build_course_catalog(_settings(tmp_path / "four"))
    four_course_ms = (time.perf_counter() - t0) * 1000

    # Both are metadata-only reads; four courses registered should not cost
    # meaningfully more than one, because none of the four had a controller
    # built. A pre-lazy-loading design (eagerly building every active
    # course's bank/taxonomy/BKT model at startup) would scale ~4x here.
    assert four_course_ms < one_course_ms * 3
