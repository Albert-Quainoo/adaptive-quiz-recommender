import csv
from pathlib import Path

from pydantic import ValidationError

from taxonomy.schemas import SkillCatalogue, SkillDefinition

LIST_COLUMNS = ("reference_material", "prerequisite_skill_ids")
LIST_SEPARATOR = ";"


class TaxonomyError(ValueError):
    pass


def split_list_cell(value: str) -> list[str]:
    return [item.strip() for item in value.split(LIST_SEPARATOR) if item.strip()]


def describe_row_errors(line_number: int, error: ValidationError) -> list[str]:
    return [
        f"  line {line_number}: "
        f"{'.'.join(str(part) for part in detail['loc']) or 'row'} - {detail['msg']}"
        for detail in error.errors()
    ]


def load_skills(path: Path) -> SkillCatalogue:
    skills: list[SkillDefinition] = []
    errors: list[str] = []

    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)

        for row in reader:
            fields = {
                column: value.strip()
                for column, value in row.items()
                if column is not None and value is not None
            }

            for column in LIST_COLUMNS:
                if column in fields:
                    fields[column] = split_list_cell(fields[column])

            try:
                skills.append(SkillDefinition(**fields))
            except ValidationError as error:
                errors.extend(describe_row_errors(reader.line_num, error))

    if errors:
        raise TaxonomyError(
            f"{path.name} has invalid rows:\n" + "\n".join(errors)
        )

    try:
        return SkillCatalogue(skills=skills)
    except ValidationError as error:
        raise TaxonomyError(f"{path.name}: {error}") from error
