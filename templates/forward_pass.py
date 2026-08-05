"""Forward-propagation template for AI-NN-05.

A small feedforward network with integer weights, so every value in the trace
is exact and the distractors can be the specific arithmetic a learner gets
wrong: dropping the biases, or forgetting that ReLU clips negatives.
"""

import random
from dataclasses import dataclass

from api.schemas import QuizQuestion, difficulty_level
from taxonomy.schemas import SkillDefinition
from templates.registry import TemplateError, register

TEMPLATE_ID = "nn.forward_trace"

# inputs, hidden units
LAYER_SIZES = {
    "introductory": (2, 2),
    "intermediate": (3, 2),
    "advanced": (3, 3),
}

WEIGHT_RANGE = (-4, 4)
INPUT_RANGE = (0, 5)
BIAS_RANGE = (-3, 3)

MAX_ATTEMPTS = 200


class UnusableNetwork(ValueError):
    pass


@dataclass(frozen=True)
class Network:
    inputs: tuple[int, ...]
    hidden_weights: tuple[tuple[int, ...], ...]
    hidden_biases: tuple[int, ...]
    output_weights: tuple[int, ...]
    output_bias: int


@dataclass(frozen=True)
class ForwardPass:
    network: Network
    pre_activations: tuple[int, ...]
    activations: tuple[int, ...]
    output: int


def forward(network: Network) -> ForwardPass:
    pre_activations = tuple(
        sum(value * weight for value, weight in zip(network.inputs, weights))
        + bias
        for weights, bias in zip(network.hidden_weights, network.hidden_biases)
    )
    activations = tuple(max(0, value) for value in pre_activations)
    output = (
        sum(
            activation * weight
            for activation, weight in zip(activations, network.output_weights)
        )
        + network.output_bias
    )

    return ForwardPass(
        network=network,
        pre_activations=pre_activations,
        activations=activations,
        output=output,
    )


def output_without_biases(network: Network) -> int:
    """What a learner gets if they drop every bias term."""
    activations = [
        max(
            0,
            sum(value * weight for value, weight in zip(network.inputs, weights)),
        )
        for weights in network.hidden_weights
    ]

    return sum(
        activation * weight
        for activation, weight in zip(activations, network.output_weights)
    )


def output_without_relu(pass_: ForwardPass) -> int:
    """What a learner gets if they carry negative pre-activations through."""
    return (
        sum(
            value * weight
            for value, weight in zip(
                pass_.pre_activations, pass_.network.output_weights
            )
        )
        + pass_.network.output_bias
    )


def generate_network(difficulty: difficulty_level, rng: random.Random) -> Network:
    input_count, hidden_count = LAYER_SIZES[difficulty]

    return Network(
        inputs=tuple(rng.randint(*INPUT_RANGE) for _ in range(input_count)),
        hidden_weights=tuple(
            tuple(rng.randint(*WEIGHT_RANGE) for _ in range(input_count))
            for _ in range(hidden_count)
        ),
        hidden_biases=tuple(rng.randint(*BIAS_RANGE) for _ in range(hidden_count)),
        output_weights=tuple(rng.randint(*WEIGHT_RANGE) for _ in range(hidden_count)),
        output_bias=rng.randint(*BIAS_RANGE),
    )


def describe_network(network: Network) -> str:
    hidden = ". ".join(
        f"Hidden unit {index + 1} has weights {list(weights)} and bias {bias}"
        for index, (weights, bias) in enumerate(
            zip(network.hidden_weights, network.hidden_biases)
        )
    )

    return (
        f"A feedforward network takes the input vector {list(network.inputs)}. "
        f"{hidden}. "
        f"Each hidden unit applies ReLU. "
        f"The output unit has weights {list(network.output_weights)}, "
        f"bias {network.output_bias}, and no activation function."
    )


def build_forward_question(
    skill: SkillDefinition,
    difficulty: difficulty_level,
    pass_: ForwardPass,
) -> QuizQuestion:
    network = pass_.network

    if all(value >= 0 for value in pass_.pre_activations):
        raise UnusableNetwork("no hidden unit is clipped, so ReLU does not matter.")

    options = [
        str(pass_.output),
        str(output_without_relu(pass_)),
        str(output_without_biases(network)),
        str(pass_.output - network.output_bias),
    ]

    if len(set(options)) != len(options):
        raise UnusableNetwork("two answer options are identical.")

    working = ". ".join(
        f"Hidden unit {index + 1}: {pre} before ReLU, {activation} after"
        for index, (pre, activation) in enumerate(
            zip(pass_.pre_activations, pass_.activations)
        )
    )

    return QuizQuestion(
        question=f"{describe_network(network)} What value does the output unit produce?",
        options=options,
        correct_answer=str(pass_.output),
        explanation=(
            f"Each hidden unit sums its weighted inputs, adds its bias, then "
            f"applies ReLU. {working}. The output unit then computes "
            f"{list(pass_.activations)} against weights "
            f"{list(network.output_weights)} plus bias {network.output_bias}, "
            f"giving {pass_.output}."
        ),
        concept=skill.name,
        difficulty=difficulty,
    )


@register(TEMPLATE_ID)
def generate(
    skill: SkillDefinition,
    difficulty: difficulty_level,
    seed: int | None = None,
) -> QuizQuestion:
    if seed is None:
        seed = random.randrange(2**32)

    rng = random.Random(seed)

    for _ in range(MAX_ATTEMPTS):
        try:
            network = generate_network(difficulty, rng)

            return build_forward_question(skill, difficulty, forward(network))
        except UnusableNetwork:
            continue

    raise TemplateError(
        f"{TEMPLATE_ID} could not build a usable {difficulty} network "
        f"from seed {seed}."
    )
