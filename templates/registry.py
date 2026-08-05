from collections.abc import Callable

from api.schemas import QuizQuestion, difficulty_level
from taxonomy.schemas import SkillCatalogue, SkillDefinition

Template = Callable[[SkillDefinition, difficulty_level, int | None], QuizQuestion]

# Template ids that the taxonomy already names but no template implements yet.
# They are listed here so the router can say "not built yet" instead of
# "unknown", and never so that generation can quietly fall back to the model.
RESERVED_TEMPLATE_IDS = (
    "search.bfs_trace",
    "search.dfs_trace",
    "search.ucs_trace",
    "search.greedy_trace",
    "nn.forward_trace",
)

TEMPLATES: dict[str, Template] = {}


class TemplateError(ValueError):
    pass


def declared_template_ids(catalogue: SkillCatalogue) -> list[str]:
    return sorted(
        {skill.template_id for skill in catalogue.skills if skill.template_id}
    )


def implemented_template_ids(catalogue: SkillCatalogue) -> list[str]:
    return [
        template_id
        for template_id in declared_template_ids(catalogue)
        if template_id in TEMPLATES
    ]


def unimplemented_template_ids(catalogue: SkillCatalogue) -> list[str]:
    return [
        template_id
        for template_id in declared_template_ids(catalogue)
        if template_id not in TEMPLATES
    ]


def register(template_id: str) -> Callable[[Template], Template]:
    def decorator(template: Template) -> Template:
        TEMPLATES[template_id] = template
        return template

    return decorator


def generate_templated_question(
    skill: SkillDefinition,
    difficulty: difficulty_level,
    seed: int | None = None,
) -> QuizQuestion:
    if skill.generation_strategy != "templated":
        raise TemplateError(
            f"{skill.skill_id} is {skill.generation_strategy}, not templated."
        )

    if not skill.template_id:
        raise TemplateError(f"{skill.skill_id} is templated but names no template.")

    template = TEMPLATES.get(skill.template_id)

    if template is None:
        if skill.template_id in RESERVED_TEMPLATE_IDS:
            raise TemplateError(
                f"{skill.template_id} is reserved in the taxonomy "
                f"but has no implementation yet."
            )

        raise TemplateError(f"{skill.template_id} is not a registered template.")

    return template(skill, difficulty, seed)
