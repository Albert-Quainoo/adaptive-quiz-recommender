"""Merge approved BankItem JSONL files without modifying their sources."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from app.bootstrap import load_approved_bank


def merge_approved_banks(bank_paths: Sequence[Path], output: Path) -> None:
    items = [item for path in bank_paths for item in load_approved_bank(path)]
    item_ids = [item.item_id for item in items]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("approved banks contain duplicate item IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
            for item in items
        ),
        encoding="utf-8",
    )
    temporary.replace(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args(argv)
    merge_approved_banks(arguments.bank, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
