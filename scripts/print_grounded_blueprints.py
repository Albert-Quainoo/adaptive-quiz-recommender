"""Print a versioned grounded-question intent blueprint for review."""

import argparse

from authoring.question_intents import (
    PILOT_BATCH_ID,
    format_blueprint_review,
    load_blueprint_for_batch,
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", default=PILOT_BATCH_ID)
    arguments = parser.parse_args(argv)
    print(format_blueprint_review(load_blueprint_for_batch(arguments.batch_id)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
