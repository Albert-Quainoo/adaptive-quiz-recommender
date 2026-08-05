from collections.abc import Callable

from api.bank import BankItem
from api.schemas import QuizGenerationRequest, QuizResponse, difficulty_level
from taxonomy.schemas import SkillDefinition
from templates.registry import generate_templated_question

QuizGenerator = Callable[[QuizGenerationRequest], QuizResponse]


class AuthoringError(ValueError):
    pass


def build_request(
    skill: SkillDefinition,
    difficulty: difficulty_level,
    question_count: int = 1,
) -> QuizGenerationRequest:
    """Turn a skill into a generation request grounded in its own references.

    A generated skill with no reference material is refused rather than sent to
    the model ungrounded, which is the failure the reference column exists to
    prevent. find_skills_missing_reference_material names them in advance.
    """
    if not skill.reference_material:
        raise AuthoringError(
            f"{skill.skill_id} has no reference material to generate from."
        )

    return QuizGenerationRequest(
        topic=skill.name,
        difficulty=difficulty,
        learning_objective=skill.learning_objective,
        question_count=question_count,
        reference_material=skill.reference_material,
    )


def build_item(
    skill: SkillDefinition,
    difficulty: difficulty_level,
    seed: int | None = None,
    generator: QuizGenerator | None = None,
) -> BankItem:
    """Author one bank item for a skill, by whichever route the skill declares.

    The model-backed generator is imported only when a generated skill actually
    needs it, so authoring a templated item never loads torch.
    """
    if skill.generation_strategy == "templated":
        question = generate_templated_question(skill, difficulty, seed)
    elif skill.generation_strategy == "generated":
        request = build_request(skill, difficulty)

        if generator is None:
            from api.quiz_generator import generate_quiz

            generator = generate_quiz

        question = generator(request).questions[0]
    else:
        raise AuthoringError(
            f"{skill.skill_id} is hand_authored, so a person writes its items."
        )

    return BankItem(
        question=question,
        provenance=skill.generation_strategy,
        skill_id=skill.skill_id,
    )
