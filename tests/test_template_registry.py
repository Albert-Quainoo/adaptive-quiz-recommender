import pytest

from api.schemas import QuizQuestion
from taxonomy.schemas import SkillDefinition
from templates import registry
from templates.registry import TemplateError, generate_templated_question


def skill(**overrides) -> SkillDefinition:
    fields = {
        "skill_id": "AI-SRC-10",
        "topic": "Search and Problem Solving",
        "subtopic": "Informed search",
        "name": "A-star Search",
        "learning_objective": "Trace A-star Search using (f(n)=g(n)+h(n)).",
        "cognitive_process": "apply",
        "generation_strategy": "templated",
        "template_id": "search.astar_trace",
    }
    fields.update(overrides)
    return SkillDefinition(**fields)


def test_registered_template_is_routed():
    question = generate_templated_question(skill(), "intermediate", seed=1)

    assert isinstance(question, QuizQuestion)
    assert question.concept == "A-star Search"


def test_unknown_template_id_is_rejected():
    with pytest.raises(TemplateError, match="search.nope is not a registered template"):
        generate_templated_question(
            skill(template_id="search.nope"), "intermediate", seed=1
        )


def test_reserved_template_id_reports_that_it_is_unimplemented(monkeypatch):
    # Every declared template is implemented today, so reserve one to prove the
    # router still distinguishes "declared but not built" from "unknown".
    monkeypatch.setattr(registry, "RESERVED_TEMPLATE_IDS", ("search.idastar_trace",))

    with pytest.raises(TemplateError, match="reserved in the taxonomy"):
        generate_templated_question(
            skill(template_id="search.idastar_trace"), "intermediate", seed=1
        )


def test_untemplated_skill_is_rejected():
    generated = skill(generation_strategy="generated", template_id=None)

    with pytest.raises(TemplateError, match="is generated, not templated"):
        generate_templated_question(generated, "intermediate", seed=1)


def test_templated_skill_without_a_template_id_is_rejected():
    # The schema forbids this combination, so it can only be reached by
    # bypassing validation - the router still refuses rather than guessing.
    unvalidated = SkillDefinition.model_construct(
        skill_id="AI-SRC-10",
        generation_strategy="templated",
        template_id=None,
    )

    with pytest.raises(TemplateError, match="names no template"):
        generate_templated_question(unvalidated, "intermediate", seed=1)
