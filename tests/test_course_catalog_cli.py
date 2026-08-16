"""CLI subcommands driven end to end, mirroring
tests/test_replenishment_cli.py's CLI-fake pattern.
"""

import json

import pytest

import authoring.course_catalog.cli as cli
import authoring.replenishment.manifest as manifest_module
from authoring.course_catalog.repository import SQLiteCourseApprovalRepository


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
