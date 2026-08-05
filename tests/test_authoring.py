import pytest

from api.schemas import QuizGenerationRequest, QuizQuestion, QuizResponse
from authoring.builder import AuthoringError, build_item, build_request
from taxonomy.schemas import SkillDefinition
from templates.registry import TemplateError

REFERENCE = "A heuristic estimates the remaining cost from a state to a goal."


def skill(**overrides) -> SkillDefinition:
    fields = {
        "skill_id": "AI-SRC-08",
        "topic": "Search and Problem Solving",
        "subtopic": "Informed search",
        "name": "Heuristic function",
        "learning_objective": (
            "Explain how a heuristic estimates the remaining cost "
            "from a state to the goal."
        ),
        "cognitive_process": "understand",
        "generation_strategy": "generated",
        "reference_material": [REFERENCE],
    }
    fields.update(overrides)
    return SkillDefinition(**fields)


def astar_skill() -> SkillDefinition:
    return SkillDefinition(
        skill_id="AI-SRC-10",
        topic="Search and Problem Solving",
        subtopic="Informed search",
        name="A-star Search",
        learning_objective="Trace A-star Search using (f(n)=g(n)+h(n)).",
        cognitive_process="apply",
        generation_strategy="templated",
        template_id="search.astar_trace",
    )


class RecordingGenerator:
    """Stands in for the model so the seam can be tested without loading one."""

    def __init__(self):
        self.requests: list[QuizGenerationRequest] = []

    def __call__(self, request: QuizGenerationRequest) -> QuizResponse:
        self.requests.append(request)

        return QuizResponse(
            questions=[
                QuizQuestion(
                    question="Which value does a heuristic estimate?",
                    options=[
                        "The remaining cost to the goal",
                        "The cost already paid",
                        "The number of expanded nodes",
                        "The branching factor",
                    ],
                    correct_answer="The remaining cost to the goal",
                    explanation="A heuristic estimates the cost still to be paid.",
                    concept="Heuristic function",
                    difficulty=request.difficulty,
                )
            ]
        )


def test_request_is_built_from_the_skill():
    request = build_request(skill(), "intermediate")

    assert request.topic == "Heuristic function"
    assert request.difficulty == "intermediate"
    assert request.learning_objective.startswith("Explain how a heuristic")
    assert request.reference_material == [REFERENCE]
    assert request.question_count == 1


def test_request_can_ask_for_several_questions():
    assert build_request(skill(), "advanced", question_count=3).question_count == 3


def test_generated_skill_without_references_is_refused():
    with pytest.raises(AuthoringError, match="no reference material"):
        build_request(skill(reference_material=[]), "intermediate")


def test_generated_item_uses_the_grounded_request():
    generator = RecordingGenerator()

    item = build_item(skill(), "introductory", generator=generator)

    assert item.provenance == "generated"
    assert item.skill_id == "AI-SRC-08"
    assert item.question.difficulty == "introductory"
    assert generator.requests[0].reference_material == [REFERENCE]


def test_generated_item_is_refused_without_references():
    generator = RecordingGenerator()

    with pytest.raises(AuthoringError, match="no reference material"):
        build_item(skill(reference_material=[]), "introductory", generator=generator)

    assert generator.requests == []


def test_templated_item_comes_from_the_template():
    item = build_item(astar_skill(), "intermediate", seed=7)

    assert item.provenance == "templated"
    assert item.skill_id == "AI-SRC-10"
    assert item.question.correct_answer == "S -> A -> C -> B -> D -> G"


def test_templated_item_never_calls_the_model():
    generator = RecordingGenerator()

    build_item(astar_skill(), "intermediate", seed=7, generator=generator)

    assert generator.requests == []


def test_unimplemented_template_does_not_fall_back_to_the_model():
    generator = RecordingGenerator()
    reserved = astar_skill().model_copy(
        update={"skill_id": "AI-NN-05", "template_id": "nn.forward_trace"}
    )

    with pytest.raises(TemplateError, match="no implementation yet"):
        build_item(reserved, "intermediate", seed=7, generator=generator)

    assert generator.requests == []


def test_hand_authored_skill_is_refused():
    with pytest.raises(AuthoringError, match="a person writes its items"):
        build_item(
            skill(skill_id="AI-ETH-01", generation_strategy="hand_authored"),
            "intermediate",
        )


def test_provenance_always_matches_the_declared_strategy():
    generator = RecordingGenerator()

    assert build_item(skill(), "intermediate", generator=generator).provenance == (
        "generated"
    )
    assert build_item(astar_skill(), "intermediate", seed=1).provenance == "templated"
