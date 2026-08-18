"""Unit tests for scripts/stage_run_artifacts.py in isolation from the
replenishment pipeline: given a manifest naming a subset of paths under a
source tree, stage_run_artifacts() must copy exactly that subset into the
destination, leave every other (stale, pre-existing) file under source
untouched, and never delete or rewrite anything in source.
"""

import json
from pathlib import Path

import pytest

from scripts.stage_run_artifacts import stage_run_artifacts


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_stages_only_manifest_listed_files_and_directories(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"

    _write(source / "_reports" / "latest.md", "# report\n")
    _write(source / "_reports" / "latest.json", "{}")
    _write(source / "intro-ai" / "this-run-job" / "job.json", '{"job_id": "this-run-job"}')

    # Stale artifacts already sitting under source from an earlier run,
    # checked out from the content-ops branch -- not part of this run.
    _write(source / "dsa" / "stale-job" / "job.json", '{"job_id": "stale-job"}')
    _write(source / "intro-ai" / "another-stale-job" / "job.json", '{"job_id": "another-stale-job"}')

    manifest_path = source / "_reports" / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "12345",
                "dry_run": False,
                "paths": [
                    "_reports/latest.md",
                    "_reports/latest.json",
                    "intro-ai/this-run-job",
                ],
            }
        ),
        encoding="utf-8",
    )

    staged = stage_run_artifacts(manifest_path, source, dest)

    assert (dest / "_reports" / "latest.md").read_text(encoding="utf-8") == "# report\n"
    assert (dest / "_reports" / "latest.json").is_file()
    assert (dest / "_reports" / "run_manifest.json").is_file()
    assert (dest / "intro-ai" / "this-run-job" / "job.json").is_file()

    # Excluded: neither stale job directory was copied into the staged output.
    assert not (dest / "dsa").exists()
    assert not (dest / "intro-ai" / "another-stale-job").exists()

    assert dest / "intro-ai" / "this-run-job" in staged


def test_source_tree_is_never_modified(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"

    _write(source / "_reports" / "latest.md", "# report\n")
    _write(source / "dsa" / "stale-job" / "job.json", '{"job_id": "stale-job"}')

    manifest_path = source / "_reports" / "run_manifest.json"
    manifest_path.write_text(
        json.dumps({"run_id": "1", "dry_run": True, "paths": ["_reports/latest.md"]}),
        encoding="utf-8",
    )

    stage_run_artifacts(manifest_path, source, dest)

    # Every file that existed under source before staging is still there,
    # byte for byte -- staging only ever reads from source and writes to dest.
    assert (source / "_reports" / "latest.md").read_text(encoding="utf-8") == "# report\n"
    assert (source / "dsa" / "stale-job" / "job.json").read_text(
        encoding="utf-8"
    ) == '{"job_id": "stale-job"}'


def test_missing_manifest_listed_path_raises(tmp_path):
    source = tmp_path / "source"
    dest = tmp_path / "dest"
    manifest_path = source / "_reports" / "run_manifest.json"
    _write(
        manifest_path,
        json.dumps({"run_id": "1", "dry_run": True, "paths": ["_reports/latest.md"]}),
    )
    # latest.md was never written -- a manifest naming a path that doesn't
    # exist is a bug in the caller, not something to silently skip.
    with pytest.raises(FileNotFoundError):
        stage_run_artifacts(manifest_path, source, dest)
