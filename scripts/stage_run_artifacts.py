"""Copy only the paths a single replenishment run's manifest names out of
outputs/replenishment/ into a run-scoped staging directory, so the GitHub
Actions artifact upload (.github/workflows/replenishment.yml's "Upload
review artifacts" step) contains exactly this run's report, snapshot,
ledger, and job-scoped artifacts -- never a stale job directory or report
left over from an earlier run and checked out again via the content-ops
branch.

The manifest itself (run_replenishment_cycle.py's _write_run_manifest())
lives at outputs/replenishment/_reports/run_manifest.json and is always
copied first, so the staged directory is self-describing.

This only copies -- it never deletes or modifies anything under --source,
so the content-ops branch's historical artifacts are untouched.

Run with:
    python -m scripts.stage_run_artifacts \
        --manifest outputs/replenishment/_reports/run_manifest.json \
        --source outputs/replenishment \
        --dest outputs/replenishment_run_artifact
"""

import argparse
import json
import shutil
from pathlib import Path


def stage_run_artifacts(manifest_path: Path, source_root: Path, dest_root: Path) -> list[Path]:
    """Copy the manifest file plus every path it lists from source_root into
    dest_root, preserving relative structure. Returns the destination paths
    written. Raises FileNotFoundError if a manifest-listed path is missing
    from source_root -- a manifest naming a path that was never written is a
    bug in the caller, not something to silently skip."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_relative = manifest_path.relative_to(source_root)

    staged: list[Path] = []

    def _copy(relative: Path) -> Path:
        src = source_root / relative
        dst = dest_root / relative
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
        return dst

    staged.append(_copy(manifest_relative))
    for entry in manifest["paths"]:
        staged.append(_copy(Path(entry)))
    return staged


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.stage_run_artifacts",
        description=(
            "Copy this run's manifest-listed report/snapshot/job artifacts "
            "into a run-scoped directory, excluding stale pre-existing files."
        ),
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--dest", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    staged = stage_run_artifacts(args.manifest, args.source, args.dest)
    for path in staged:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
