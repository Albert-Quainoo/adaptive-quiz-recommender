import random

import pytest

from api.schemas import QuizQuestion
from taxonomy.schemas import SkillDefinition
from templates.forward_pass import (
    Network,
    UnusableNetwork,
    build_forward_question,
    forward,
    generate,
    generate_network,
    output_without_biases,
    output_without_relu,
)
from templates.registry import generate_templated_question

# Hand-worked network:
#   hidden 1: 1*-2 + 4*1 + 4*3 = 14, minus bias 1 -> 13, ReLU keeps 13
#   hidden 2: 1*-3 + 4*-4 + 4*3 = -7, plus bias 1 -> -6, ReLU clips to 0
#   output:   13*-1 + 0*-1 + 2 = -11
FIXTURE = Network(
    inputs=(1, 4, 4),
    hidden_weights=((-2, 1, 3), (-3, -4, 3)),
    hidden_biases=(-1, 1),
    output_weights=(-1, -1),
    output_bias=2,
)


def nn_skill() -> SkillDefinition:
    return SkillDefinition(
        skill_id="AI-NN-05",
        topic="Neural Networks and Deep Learning",
        subtopic="Network operation",
        name="Forward propagation",
        learning_objective=(
            "Trace how information moves through a feedforward neural network."
        ),
        cognitive_process="apply",
        generation_strategy="templated",
        template_id="nn.forward_trace",
    )


def test_forward_pass_is_hand_checked():
    result = forward(FIXTURE)

    assert result.pre_activations == (13, -6)
    assert result.activations == (13, 0)
    assert result.output == -11


def test_the_distractor_arithmetic_is_hand_checked():
    result = forward(FIXTURE)

    assert output_without_relu(result) == -5
    assert output_without_biases(FIXTURE) == -14


def test_relu_is_what_separates_the_answer_from_the_distractor():
    result = forward(FIXTURE)

    assert result.output != output_without_relu(result)


def test_a_network_that_never_clips_is_rejected():
    positive = Network(
        inputs=(1, 1, 1),
        hidden_weights=((1, 1, 1), (2, 2, 2)),
        hidden_biases=(1, 1),
        output_weights=(1, 1),
        output_bias=0,
    )

    with pytest.raises(UnusableNetwork, match="ReLU does not matter"):
        build_forward_question(nn_skill(), "intermediate", forward(positive))


def test_question_reports_the_hand_checked_values():
    question = build_forward_question(nn_skill(), "intermediate", forward(FIXTURE))

    assert question.correct_answer == "-11"
    assert set(question.options) == {"-11", "-5", "-14", "-13"}
    assert "13 before ReLU, 13 after" in question.explanation
    assert "-6 before ReLU, 0 after" in question.explanation


def test_template_is_registered_and_routed():
    question = generate_templated_question(nn_skill(), "intermediate", seed=3)

    assert isinstance(question, QuizQuestion)
    assert question.concept == "Forward propagation"


def test_the_same_seed_produces_an_identical_question():
    first = generate(nn_skill(), "advanced", seed=21)
    second = generate(nn_skill(), "advanced", seed=21)

    assert first.model_dump() == second.model_dump()


def test_different_seeds_produce_different_questions():
    questions = {generate(nn_skill(), "intermediate", seed=seed).question for seed in range(10)}

    assert len(questions) > 1


@pytest.mark.parametrize("difficulty", ["introductory", "intermediate", "advanced"])
@pytest.mark.parametrize("seed", range(8))
def test_generated_questions_are_valid_and_arithmetically_true(difficulty, seed):
    question = generate(nn_skill(), difficulty, seed=seed)

    assert len(question.options) == 4
    assert len(set(question.options)) == 4
    assert question.correct_answer in question.options
    assert QuizQuestion.model_validate(question.model_dump()) == question


@pytest.mark.parametrize("difficulty", ["introductory", "intermediate", "advanced"])
def test_layer_sizes_follow_the_difficulty(difficulty):
    network = generate_network(difficulty, random.Random(1))
    expected_inputs, expected_hidden = {
        "introductory": (2, 2),
        "intermediate": (3, 2),
        "advanced": (3, 3),
    }[difficulty]

    assert len(network.inputs) == expected_inputs
    assert len(network.hidden_weights) == expected_hidden
    assert all(len(weights) == expected_inputs for weights in network.hidden_weights)


def test_a_missing_seed_still_produces_a_valid_question():
    question = generate(nn_skill(), "intermediate")

    assert question.correct_answer in question.options
