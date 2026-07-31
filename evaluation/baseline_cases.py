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
        ),
    ),
    EvaluationCase(
        case_id="BASE_AGENT_002",
        request=QuizGenerationRequest(
            topic="PEAS and environment types",
            difficulty="intermediate",
            learning_objective="Identify PEAS components and classify task environments from given scenarios.",
            question_count=3,
        ),
    ),
    EvaluationCase(
        case_id="BASE_SEARCH_001",
        request=QuizGenerationRequest(
            topic="BFS and DFS",
            difficulty="intermediate",
            learning_objective="Distinguish BFS from DFS using expansion order, frontier behaviour, and search strategy.",
            question_count=3,
        ),
    ),
    EvaluationCase(
        case_id="BASE_SEARCH_002",
        request=QuizGenerationRequest(
            topic="Uniform-cost search",
            difficulty="intermediate",
            learning_objective="Calculate cumulative path costs and select the next frontier node using uniform-cost search.",
            question_count=3,
        ),
    ),
    EvaluationCase(
        case_id="BASE_SEARCH_003",
        request=QuizGenerationRequest(
            topic="A-star search",
            difficulty="advanced",
            learning_objective="Calculate g(n), h(n), and f(n) values, select the most promising node, and evaluate how heuristic quality affects optimality and search efficiency.",
            question_count=5,
        ),
    ),
    EvaluationCase(
        case_id="BASE_ML_001",
        request=QuizGenerationRequest(
            topic="Supervised learning",
            difficulty="introductory",
            learning_objective="Identify supervised-learning problems by recognising labelled data and target outputs.",
            question_count=3,
        ),
    ),
    EvaluationCase(
        case_id="BASE_ML_002",
        request=QuizGenerationRequest(
            topic="Classification and regression",
            difficulty="intermediate",
            learning_objective="Distinguish classification from regression based on the type of target being predicted.",
            question_count=3,
        ),
    ),
    EvaluationCase(
        case_id="BASE_ML_003",
        request=QuizGenerationRequest(
            topic="K-means clustering",
            difficulty="intermediate",
            learning_objective="Apply the assignment and centroid-update steps of the K-means clustering algorithm.",
            question_count=3,
        ),
    ),
    EvaluationCase(
        case_id="BASE_NN_001",
        request=QuizGenerationRequest(
            topic="Neural networks",
            difficulty="advanced",
            learning_objective="Analyse a multi-layer neural-network forward pass and explain how weights, biases, and nonlinear activation functions affect its output.",
            question_count=5,
        ),
    ),
]
