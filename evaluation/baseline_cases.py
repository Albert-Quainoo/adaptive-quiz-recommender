from api.schemas import QuizGenerationRequest
from evaluation.schemas import EvaluationCase

BASELINE_CASES: list[EvaluationCase] = [
    EvaluationCase(
        case_id="BASE_AGENT_001",
        request=QuizGenerationRequest(
            topic="Intelligent agents",
            difficulty="introductory",
            learning_objective="Identify the defining characteristics of an intelligent rational agent.",
            question_count=1,
            reference_material=[
                "An agent perceives its environment through sensors and acts upon that environment through actuators.",
                "A rational agent selects the action expected to maximise its performance measure, given its percept sequence and built-in knowledge.",
                "Rationality is not omniscience: it depends on the performance measure, prior knowledge, available actions, and the percepts received so far.",
            ],
        ),
    ),
    EvaluationCase(
        case_id="BASE_AGENT_002",
        request=QuizGenerationRequest(
            topic="PEAS and environment types",
            difficulty="intermediate",
            learning_objective="Identify PEAS components and classify task environments from given scenarios.",
            question_count=3,
            reference_material=[
                "PEAS stands for Performance measure, Environment, Actuators, Sensors.",
                "For an automated taxi driver: performance measure is safety, speed, legality and comfort; environment is roads, traffic and pedestrians; actuators are steering, accelerator, brake and indicator; sensors are cameras, GPS and speedometer.",
                "Task environments are classified along these dimensions: fully observable or partially observable; single-agent or multi-agent; deterministic or stochastic; episodic or sequential; static or dynamic; discrete or continuous; known or unknown.",
                "These environment types describe the task an agent operates in. They are not categories of business, physical or economic environment.",
            ],
        ),
    ),
    EvaluationCase(
        case_id="BASE_SEARCH_001",
        request=QuizGenerationRequest(
            topic="BFS and DFS",
            difficulty="intermediate",
            learning_objective="Distinguish BFS from DFS using expansion order, frontier behaviour, and search strategy.",
            question_count=3,
            reference_material=[
                "Breadth-first search expands the shallowest unexpanded node and stores its frontier in a FIFO queue.",
                "Depth-first search expands the deepest unexpanded node and stores its frontier in a LIFO stack.",
                "Breadth-first search is complete, and optimal when every step cost is equal; its main weakness is memory, since the frontier grows exponentially with depth.",
                "Depth-first search needs memory linear in the depth of the tree, but it is not optimal and is not complete on infinite-depth search spaces.",
            ],
        ),
    ),
    EvaluationCase(
        case_id="BASE_SEARCH_002",
        request=QuizGenerationRequest(
            topic="Uniform-cost search",
            difficulty="intermediate",
            learning_objective="Calculate cumulative path costs and select the next frontier node using uniform-cost search.",
            question_count=3,
            reference_material=[
                "Uniform-cost search expands the frontier node with the lowest cumulative path cost g(n).",
                "The cumulative cost of a child equals the cumulative cost of its parent plus the step cost from parent to child.",
                "The frontier is a priority queue ordered by g(n), and expanding a node removes it from the frontier and adds its children.",
                "Uniform-cost search is optimal whenever every step cost is non-negative, and it behaves like breadth-first search when all step costs are equal.",
            ],
        ),
    ),
    EvaluationCase(
        case_id="BASE_SEARCH_003",
        request=QuizGenerationRequest(
            topic="A-star search",
            difficulty="advanced",
            learning_objective="Calculate g(n), h(n), and f(n) values, select the most promising node, and evaluate how heuristic quality affects optimality and search efficiency.",
            question_count=5,
            reference_material=[
                "In A-star search, f(n) equals g(n) plus h(n), where g(n) is the cost of the path from the start node to n and h(n) estimates the remaining cost from n to the goal.",
                "A-star search expands the frontier node with the lowest f(n) value.",
                "A heuristic is admissible when it never overestimates the true remaining cost. A-star with an admissible heuristic returns an optimal solution in tree search.",
                "A heuristic is consistent when the estimate at a node never exceeds the step cost to a neighbour plus the estimate at that neighbour. Consistency guarantees optimality in graph search.",
                "A more informed heuristic, one that gives higher estimates while remaining admissible, expands fewer nodes and so searches more efficiently. Improving heuristic quality does not reduce optimality.",
            ],
        ),
    ),
    EvaluationCase(
        case_id="BASE_ML_001",
        request=QuizGenerationRequest(
            topic="Supervised learning",
            difficulty="introductory",
            learning_objective="Identify supervised-learning problems by recognising labelled data and target outputs.",
            question_count=3,
            reference_material=[
                "Supervised learning trains a model on labelled examples, where each input is paired with its target output.",
                "The goal is to learn a mapping from inputs to outputs that generalises to unseen inputs.",
                "Unsupervised learning uses unlabelled data and discovers structure such as clusters. Reinforcement learning learns from reward signals rather than labelled targets.",
            ],
        ),
    ),
    EvaluationCase(
        case_id="BASE_ML_002",
        request=QuizGenerationRequest(
            topic="Classification and regression",
            difficulty="intermediate",
            learning_objective="Distinguish classification from regression based on the type of target being predicted.",
            question_count=3,
            reference_material=[
                "Classification predicts a discrete class label. Regression predicts a continuous numeric value.",
                "Predicting a house price or a temperature is regression. Predicting whether an email is spam is classification.",
                "A model that outputs the probability of an event is still performing classification, because the target being predicted is a class label rather than a continuous quantity.",
            ],
        ),
    ),
    EvaluationCase(
        case_id="BASE_ML_003",
        request=QuizGenerationRequest(
            topic="K-means clustering",
            difficulty="intermediate",
            learning_objective="Apply the assignment and centroid-update steps of the K-means clustering algorithm.",
            question_count=3,
            reference_material=[
                "K-means alternates two steps. In the assignment step each data point is assigned to the cluster of its nearest centroid, measured by Euclidean distance.",
                "In the update step each centroid is recomputed as the mean of all points currently assigned to it.",
                "The algorithm repeats these two steps until the cluster assignments stop changing.",
            ],
        ),
    ),
    EvaluationCase(
        case_id="BASE_NN_001",
        request=QuizGenerationRequest(
            topic="Neural networks",
            difficulty="advanced",
            learning_objective="Analyse a multi-layer neural-network forward pass and explain how weights, biases, and nonlinear activation functions affect its output.",
            question_count=5,
            reference_material=[
                "A neuron computes a weighted sum of its inputs plus a bias term, then applies an activation function to that sum.",
                "In a forward pass the activations of each layer become the inputs of the next layer.",
                "Without a nonlinear activation function, any stack of layers collapses into a single equivalent linear transformation.",
                "ReLU outputs zero for negative inputs and the input itself otherwise. The sigmoid function squashes any input into the range between zero and one.",
                "The bias shifts the weighted sum before the activation is applied, changing the input value at which the activation function changes behaviour. It does not change how nonlinear the activation function is.",
            ],
        ),
    ),
]
