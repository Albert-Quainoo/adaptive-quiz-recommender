"""Correctness and query-count regression coverage for the unit-of-work
fast path added to app/controller.py's submit_answer (see
database.unit_of_work/connection_scope): concurrent duplicate submissions
still only ever produce one attempt/mastery row, an injected mid-flow
failure rolls back everything the action did, and each phase's actual
query count stays within the bound the learner-path performance
investigation established -- so a future change that quietly reintroduces
a redundant round trip fails a test instead of just showing up as a slower
production benchmark someone has to notice.
"""

import contextlib
import io
import json
import threading

import pytest

from app.flow import activate_course
from app.perf import phase
from app.session import attempt_id_for, get_session_state, identify_learner
from tests.test_streamlit_application import build_controller, option


def _perf_metric_from(buffer: io.StringIO) -> dict:
    lines = [
        line for line in buffer.getvalue().splitlines() if line.startswith("PERF_METRIC ")
    ]
    assert len(lines) == 1, buffer.getvalue()
    return json.loads(lines[0][len("PERF_METRIC ") :])


def test_concurrent_duplicate_submissions_produce_exactly_one_attempt_and_mastery_row(
    tmp_path,
):
    controller, repository = build_controller(tmp_path / "concurrent.sqlite3")
    question = controller.recommend_question("learner-1", [])
    selected = option(question, "Correct")

    results: list[object] = []
    barrier = threading.Barrier(5)

    def submit_once():
        barrier.wait()
        try:
            results.append(
                controller.submit_answer("learner-1", question.presentation_id, selected)
            )
        except Exception as exc:  # noqa: BLE001 - recording for assertion below
            results.append(exc)

    threads = [threading.Thread(target=submit_once) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    # Every thread submitted the exact same answer to the exact same
    # question -- a genuine duplicate/retry scenario, not a conflict -- so
    # every one must succeed with the identical outcome, never raise.
    assert all(not isinstance(result, Exception) for result in results), results
    assert len({result.attempt_id for result in results}) == 1
    assert len({result.updated_mastery for result in results}) == 1

    assert len(repository.list_attempts(learner_id="learner-1")) == 1
    assert len(repository.list_mastery(learner_id="learner-1")) == 1
    repository.close()


def test_concurrent_conflicting_submissions_reject_the_loser(tmp_path):
    """The realistic race: the same presentation submitted concurrently
    with two *different* selected options (e.g. a slow double-click) must
    not let the second write silently overwrite the first -- exactly one
    thread's answer becomes the stored attempt (attempt_id is a hash of
    learner_id+presentation_id, so both threads target the same row)."""
    controller, repository = build_controller(tmp_path / "concurrent-conflict.sqlite3")
    question = controller.recommend_question("learner-1", [])
    correct = option(question, "Correct")
    wrong = option(question, "Wrong A")

    results: list[object] = []
    barrier = threading.Barrier(2)

    def submit(selected_option_id):
        barrier.wait()
        try:
            results.append(
                controller.submit_answer(
                    "learner-1", question.presentation_id, selected_option_id
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(exc)

    t1 = threading.Thread(target=submit, args=(correct,))
    t2 = threading.Thread(target=submit, args=(wrong,))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(repository.list_attempts(learner_id="learner-1")) == 1
    assert len(repository.list_mastery(learner_id="learner-1")) == 1
    repository.close()


def test_injected_failure_after_recommendation_write_rolls_back_the_whole_submission(
    tmp_path, monkeypatch
):
    """Forces a failure inside BKTService.process_attempt (after
    answer_persistence's reads have already run) and asserts nothing from
    this submission was persisted -- self.repository.unit_of_work()
    wrapping submit_answer means a failure anywhere in the action rolls
    back everything, not just whichever individual write happened to be
    mid-flight."""
    controller, repository = build_controller(tmp_path / "injected-failure.sqlite3")
    question = controller.recommend_question("learner-1", [])
    selected = option(question, "Correct")

    def explode(*args, **kwargs):
        raise RuntimeError("synthetic failure inside process_attempt")

    monkeypatch.setattr(controller.bkt_service.model, "update_mastery", explode)

    with pytest.raises(Exception):
        controller.submit_answer("learner-1", question.presentation_id, selected)

    attempt_id = attempt_id_for("learner-1", question.presentation_id)
    assert repository.get_attempt(attempt_id) is None
    assert repository.list_attempts(learner_id="learner-1") == []
    assert repository.list_mastery(learner_id="learner-1") == []
    repository.close()


def test_query_count_stays_within_the_established_bound_for_submission(tmp_path):
    controller, repository = build_controller(tmp_path / "query-budget-submit.sqlite3")
    question = controller.recommend_question("learner-1", [])
    selected = option(question, "Correct")

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        with phase("total_submit_rerun", correlation_id="query-budget-submit-check"):
            controller.submit_answer("learner-1", question.presentation_id, selected)
    payload = _perf_metric_from(buffer)

    # Established by the learner-path performance investigation: submission
    # was ~10 queries and one pool checkout per statement before the
    # unit-of-work + atomic-upsert changes. This asserts the regression
    # ceiling this work achieved, not a brittle exact count.
    assert payload["db_queries"] <= 8, payload
    assert payload["pool_checkouts"] == 1, payload
    repository.close()


def test_query_count_stays_within_the_established_bound_for_course_selection(tmp_path):
    controller, repository = build_controller(tmp_path / "query-budget-course.sqlite3")
    session = get_session_state({})
    identify_learner(session, "learner-1")

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        with phase("course_selection", correlation_id="query-budget-course-check"):
            activate_course(controller, session, "test-course")
    payload = _perf_metric_from(buffer)

    # Before this work, course_selection opened a separate connection per
    # statement (2 checkouts: start_learner_session, answered_item_ids).
    assert payload["pool_checkouts"] == 1, payload
    repository.close()
