import csv
from pathlib import Path

from pydantic import ValidationError

from taxonomy.schemas import SkillCatalogue, SkillDefinition

SKILL_COLUMNS = (
    "skill_id",
    "topic",
    "subtopic",
    "name",
    "learning_objective",
    "cognitive_process",
    "generation_strategy",
    "template_id",
    "prerequisite_skill_ids",
)

REFERENCE_COLUMNS = (
    "skill_id",
    "reference_material",
)

LIST_SEPARATOR = ";"

DATA_DIR = Path(__file__).resolve().parent / "data"


def course_paths(course: str) -> tuple[Path, Path]:
    directory = DATA_DIR / course.lower()

    return directory / "skills.csv", directory / "references.csv"


class TaxonomyError(ValueError):
    pass


def split_list_cell(value: str) -> list[str]:
    return [item.strip() for item in value.split(LIST_SEPARATOR) if item.strip()]


def validate_header(columns: list[str] | None, expected: tuple[str, ...]) -> None:
    if not columns:
        raise TaxonomyError("the file has no header row.")

    problems = []

    duplicates = sorted({column for column in columns if columns.count(column) > 1})
    if duplicates:
        problems.append(f"duplicate columns: {', '.join(duplicates)}")

    missing = sorted(set(expected) - set(columns))
    if missing:
        problems.append(f"missing columns: {', '.join(missing)}")

    unexpected = sorted(set(columns) - set(expected))
    if unexpected:
        problems.append(f"unexpected columns: {', '.join(unexpected)}")

    if problems:
        raise TaxonomyError("; ".join(problems))


def describe_row_shape(label: str, row: dict) -> str | None:
    if row.get(None):
        return f"{label}: row has {len(row[None])} unexpected extra value(s)"

    empty_cells = sorted(
        column for column, value in row.items()
        if column is not None and value is None
    )

    if empty_cells:
        return f"{label}: row is missing cells for {', '.join(empty_cells)}"

    return None


def describe_row_errors(label: str, error: ValidationError) -> list[str]:
    return [
        f"{label}: "
        f"{'.'.join(str(part) for part in detail['loc']) or 'row'} - {detail['msg']}"
        for detail in error.errors()
    ]


def read_rows(
    path: Path, expected: tuple[str, ...]
) -> tuple[list[tuple[str, dict]], list[str]]:
    rows: list[tuple[str, dict]] = []
    errors: list[str] = []

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)

        if reader.fieldnames:
            reader.fieldnames = [column.strip() for column in reader.fieldnames]

        try:
            validate_header(reader.fieldnames, expected)
        except TaxonomyError as error:
            raise TaxonomyError(f"{path.name}: {error}") from error

        for row in reader:
            label = f"  {path.name} line {reader.line_num}"
            shape_problem = describe_row_shape(label, row)

            if shape_problem:
                errors.append(shape_problem)
                continue

            rows.append((label, {column: value.strip() for column, value in row.items()}))

    return rows, errors


def load_skills(skills_path: Path, references_path: Path) -> SkillCatalogue:
    skill_rows, errors = read_rows(skills_path, SKILL_COLUMNS)
    reference_rows, reference_errors = read_rows(references_path, REFERENCE_COLUMNS)
    errors.extend(reference_errors)

    references: dict[str, list[str]] = {}

    for label, fields in reference_rows:
        if not fields["reference_material"]:
            errors.append(f"{label}: reference_material is empty")
            continue

        skill_id = fields["skill_id"].upper()
        reference = fields["reference_material"]

        if reference in references.get(skill_id, []):
            errors.append(f"{label}: duplicate reference for {skill_id}")
            continue

        references.setdefault(skill_id, []).append(reference)

    skills: list[SkillDefinition] = []

    for label, fields in skill_rows:
        skill_id = fields["skill_id"].upper()

        fields["prerequisite_skill_ids"] = split_list_cell(
            fields["prerequisite_skill_ids"]
        )
        fields["reference_material"] = references.pop(skill_id, [])

        try:
            skills.append(SkillDefinition(**fields))
        except ValidationError as error:
            errors.extend(describe_row_errors(label, error))

    for skill_id in sorted(references):
        errors.append(
            f"  {references_path.name}: reference for unknown skill id {skill_id}"
        )

    courses = sorted({skill.skill_id.split("-")[0] for skill in skills})

    if len(courses) > 1:
        errors.append(
            f"  {skills_path.name}: one file must hold one course, "
            f"found {', '.join(courses)}"
        )

    if errors:
        raise TaxonomyError("the taxonomy has problems:\n" + "\n".join(errors))

    try:
        return SkillCatalogue(skills=skills)
    except ValidationError as error:
        raise TaxonomyError(f"{skills_path.name}: {error}") from error
