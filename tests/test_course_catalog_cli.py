"""CLI subcommands driven end to end, mirroring
tests/test_replenishment_cli.py's CLI-fake pattern.
"""

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

import authoring.course_catalog.cli as cli
import authoring.replenishment.manifest as manifest_module
from authoring.course_catalog import readiness as readiness_module
from authoring.course_catalog.repository import SQLiteCourseApprovalRepository
from authoring.grounded_review import CurationItem, GroundedReview


def _write_manifest(directory, course_id, status, **overrides):
    fields = dict(
        course_id=course_id,
        title="X",
        version="1",
        taxonomy_path=str(directory / "taxonomy"),
        approved_bank_path=str(directory / "bank.jsonl"),
        bkt_model_path=str(directory / "model.pkl"),
        candidate_store_path=str(directory / "candidates.json"),
        review_store_path=str(directory / "reviews"),
        allowed_domains=["example.edu"],
        low_supply_threshold=3,
        target_supply=6,
        default_bkt_model_version="v1",
        status=status,
        aliases=[],
        auto_activate_when_ready=True,
    )
    fields.update(overrides)
    (directory / f"{course_id}.json").write_text(json.dumps(fields), encoding="utf-8")


@pytest.fixture
def isolated_manifests(tmp_path, monkeypatch):
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    monkeypatch.setattr(manifest_module, "MANIFEST_DIRECTORY", manifest_dir)
    monkeypatch.setattr(cli, "MANIFEST_DIRECTORY", manifest_dir)
    return manifest_dir


def test_inspect_course_readiness_on_unready_course_reports_blockers_and_exits_zero(
    tmp_path, isolated_manifests, capsys
):
    _write_manifest(
        isolated_manifests, "x", "awaiting_content_approval"
    )
    database = tmp_path / "db.sqlite3"
    exit_code = cli.main(["--database", str(database), "inspect-course-readiness", "x"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "not ready" in output
    saved = json.loads((isolated_manifests / "x.json").read_text())
    assert saved["status"] == "awaiting_content_approval"


def test_approve_course_on_non_proposed_course_fails_safely_with_no_partial_state(
    tmp_path, isolated_manifests, capsys
):
    _write_manifest(isolated_manifests, "x", "preparing")
    database = tmp_path / "db.sqlite3"
    exit_code = cli.main(
        ["--database", str(database), "approve-course", "x", "--approver", "op"]
    )
    assert exit_code != 0
    saved = json.loads((isolated_manifests / "x.json").read_text())
    assert saved["status"] == "preparing"

    repository = SQLiteCourseApprovalRepository(database)
    repository.initialize_schema()
    assert repository.list_for_course("x") == []


def test_approve_course_on_proposed_course_transitions_and_writes_manifest(
    tmp_path, isolated_manifests, capsys
):
    _write_manifest(isolated_manifests, "x", "proposed")
    database = tmp_path / "db.sqlite3"
    exit_code = cli.main(
        ["--database", str(database), "approve-course", "x", "--approver", "op"]
    )
    assert exit_code == 0
    saved = json.loads((isolated_manifests / "x.json").read_text())
    assert saved["status"] == "approved_for_preparation"

    repository = SQLiteCourseApprovalRepository(database)
    repository.initialize_schema()
    records = repository.list_for_course("x")
    assert len(records) == 1
    assert records[0].decision == "approved"


def test_reject_course_appends_a_record_without_changing_manifest_status(
    tmp_path, isolated_manifests, capsys
):
    _write_manifest(isolated_manifests, "x", "proposed")
    database = tmp_path / "db.sqlite3"
    exit_code = cli.main(
        ["--database", str(database), "reject-course", "x", "--reason", "missing sources"]
    )
    assert exit_code == 0
    saved = json.loads((isolated_manifests / "x.json").read_text())
    assert saved["status"] == "proposed"

    repository = SQLiteCourseApprovalRepository(database)
    repository.initialize_schema()
    records = repository.list_for_course("x")
    assert len(records) == 1
    assert records[0].decision == "rejected"


def test_archive_course_on_non_active_course_fails_safely(
    tmp_path, isolated_manifests, capsys
):
    _write_manifest(isolated_manifests, "x", "preparing")
    database = tmp_path / "db.sqlite3"
    exit_code = cli.main(
        ["--database", str(database), "archive-course", "x", "--approver", "op"]
    )
    assert exit_code != 0
    saved = json.loads((isolated_manifests / "x.json").read_text())
    assert saved["status"] == "preparing"


def test_archive_course_on_active_course_removes_it_from_active_manifests(
    tmp_path, isolated_manifests, capsys
):
    _write_manifest(isolated_manifests, "x", "active")
    database = tmp_path / "db.sqlite3"
    exit_code = cli.main(
        ["--database", str(database), "archive-course", "x", "--approver", "op"]
    )
    assert exit_code == 0
    saved = json.loads((isolated_manifests / "x.json").read_text())
    assert saved["status"] == "archived"
    assert manifest_module.load_active_manifests() == []


_READY_SKILL_ID = "XX-PLC-01"


class _FakeFittedModel:
    def __init__(self, skill_id: str, prior: float) -> None:
        index = pd.MultiIndex.from_tuples(
            [(skill_id, "prior"), (skill_id, "learns")], names=["skill", "param"]
        )
        self._frame = pd.DataFrame({"value": [prior, 0.3]}, index=index)

    def get_parameters(self) -> pd.DataFrame:
        return self._frame


def _write_ready_manifest(directory, tmp_path, course_id, *, monkeypatch, **overrides):
    """A course genuinely satisfying every readiness check (real taxonomy,
    bank, approved review, and a patched fitted BKT model), at
    awaiting_content_approval -- so activation tests exercise the real
    is_ready=True path, not just the "not ready" no-op."""
    taxonomy_dir = tmp_path / f"{course_id}-taxonomy"
    taxonomy_dir.mkdir(exist_ok=True)
    (taxonomy_dir / "skills.csv").write_text(
        "skill_id,topic,subtopic,name,learning_objective,cognitive_process,"
        "generation_strategy,template_id,prerequisite_skill_ids\n"
        f"{_READY_SKILL_ID},Topic,Subtopic,Name,Objective,remember,hand_authored,,\n",
        encoding="utf-8",
    )
    (taxonomy_dir / "references.csv").write_text(
        "skill_id,reference_material\n", encoding="utf-8"
    )

    bank_path = tmp_path / f"{course_id}-bank.jsonl"
    item = {
        "item_id": "x-item-1",
        "provenance": "generated",
        "skill_id": _READY_SKILL_ID,
        "question": {
            "question": "What is placeholder?",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "A",
            "explanation": "Because.",
            "concept": "Placeholder",
            "difficulty": "introductory",
        },
    }
    bank_path.write_text(json.dumps(item) + "\n", encoding="utf-8")

    review_store_path = tmp_path / f"{course_id}-reviews"
    review_store_path.mkdir(exist_ok=True)
    review = GroundedReview(
        batch_id="b1",
        source_hashes={},
        items=[
            CurationItem(
                original_question_id="q1",
                skill_id=_READY_SKILL_ID,
                intent_id="i1",
                recommendation="approve_as_written",
                recommendation_reason="looks correct",
                final_review_status="approved",
                reviewed_by="reviewer-1",
                reviewed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
            )
        ],
    )
    (review_store_path / "b1__x.json").write_text(review.model_dump_json(), encoding="utf-8")

    model_path = tmp_path / f"{course_id}-model.pkl"
    model_path.touch()

    monkeypatch.setattr(
        readiness_module,
        "load_fitted_bkt_model",
        lambda path, *, model_version, course_id: _FakeFittedModel(_READY_SKILL_ID, 0.20),
    )

    _write_manifest(
        directory,
        course_id,
        "awaiting_content_approval",
        taxonomy_path=str(taxonomy_dir),
        approved_bank_path=str(bank_path),
        bkt_model_path=str(model_path),
        review_store_path=str(review_store_path),
        **overrides,
    )


def test_inspect_course_readiness_writes_nothing(
    tmp_path, isolated_manifests, monkeypatch, capsys
):
    """Proves the read-only claim directly: even for a fully ready course,
    inspect-course-readiness must never open the admin database (the file
    must not even be created) and must never change the manifest."""
    _write_ready_manifest(isolated_manifests, tmp_path, "x", monkeypatch=monkeypatch)
    database = tmp_path / "db.sqlite3"

    exit_code = cli.main(["--database", str(database), "inspect-course-readiness", "x"])

    assert exit_code == 0
    assert "x: ready" in capsys.readouterr().out
    assert not database.exists()
    saved = json.loads((isolated_manifests / "x.json").read_text())
    assert saved["status"] == "awaiting_content_approval"


def test_activate_course_without_confirm_reports_readiness_and_writes_nothing(
    tmp_path, isolated_manifests, monkeypatch, capsys
):
    _write_ready_manifest(isolated_manifests, tmp_path, "x", monkeypatch=monkeypatch)
    database = tmp_path / "db.sqlite3"

    exit_code = cli.main(
        ["--database", str(database), "activate-course", "x", "--approver", "op"]
    )

    assert exit_code != 0
    saved = json.loads((isolated_manifests / "x.json").read_text())
    assert saved["status"] == "awaiting_content_approval"
    if database.exists():
        repository = SQLiteCourseApprovalRepository(database)
        repository.initialize_schema()
        assert repository.list_for_course("x") == []


def test_activate_course_on_not_ready_course_fails_even_with_confirm(
    tmp_path, isolated_manifests, capsys
):
    _write_manifest(isolated_manifests, "x", "awaiting_content_approval")
    database = tmp_path / "db.sqlite3"

    exit_code = cli.main(
        [
            "--database", str(database),
            "activate-course", "x",
            "--approver", "op",
            "--confirm",
        ]
    )

    assert exit_code != 0
    saved = json.loads((isolated_manifests / "x.json").read_text())
    assert saved["status"] == "awaiting_content_approval"


def test_activate_course_with_confirm_activates_a_ready_course_and_records_the_real_approver(
    tmp_path, isolated_manifests, monkeypatch, capsys
):
    _write_ready_manifest(isolated_manifests, tmp_path, "x", monkeypatch=monkeypatch)
    database = tmp_path / "db.sqlite3"

    exit_code = cli.main(
        [
            "--database", str(database),
            "activate-course", "x",
            "--approver", "albert",
            "--confirm",
        ]
    )

    assert exit_code == 0
    saved = json.loads((isolated_manifests / "x.json").read_text())
    assert saved["status"] == "active"

    repository = SQLiteCourseApprovalRepository(database)
    repository.initialize_schema()
    records = repository.list_for_course("x")
    assert len(records) == 1
    assert records[0].decision == "activated"
    assert records[0].approver_identity == "albert"
