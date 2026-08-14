"""Create pending curated revisions for grounded-pilot-20260811-v3."""

import argparse
from pathlib import Path

from authoring.pilot_curation_v3 import write_artifacts


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--batch", type=Path, required=True)
    root.add_argument("--output", type=Path, required=True)
    return root


def main(argv=None) -> int:
    arguments = parser().parse_args(argv)
    write_artifacts(arguments.batch, arguments.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
