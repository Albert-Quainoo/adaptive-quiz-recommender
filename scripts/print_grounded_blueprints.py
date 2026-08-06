"""Print the proposed grounded-pilot v2 intent blueprints for review."""

from authoring.question_intents import format_blueprint_review, load_pilot_blueprint


def main() -> int:
    print(format_blueprint_review(load_pilot_blueprint()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
