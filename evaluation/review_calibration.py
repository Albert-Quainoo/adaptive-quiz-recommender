"""Offline evaluation and threshold-calibration harness for the automated review
layer -- never a training procedure. It measures a reviewer's precision/recall and
(most importantly) its false-low-risk rate against a labeled positive/negative set,
so review_policy_version and confidence thresholds can be chosen and re-checked
deliberately, not by trusting the reviewer's own self-report.

Positive set: the 38 existing human-approved bank items, plus the corrected
heuristic h(n)/g(n) revision. Negative set: the original inaccurate heuristic item,
plus synthetic mutations of approved items covering the seven mutation types this
milestone calls out (swap answer, unsupported explanation, duplicate distractor,
objective mismatch, wrong difficulty, missing grounding reference, and the
heuristic-wording conflation itself).

Optimize for a near-zero false_low_risk_rate, not maximum automatic acceptance.
"""

import json
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from api.schemas import QuizQuestion
from authoring.grounded_batch import IntentQuestion, PendingQuestion
from authoring.grounded_review import question_content_hash
from authoring.question_intents import QuestionIntent
from authoring.review.config import ReviewPolicyConfig
from authoring.review.models import (
    AnswerAssessment,
    DifficultyAssessment,
    DuplicateAssessment,
    GroundingAssessment,
    ObjectiveAssessment,
    SemanticReviewResult,
)
from authoring.review.reports import AutomatedReviewReportStore
from authoring.review.response_parser import ReviewerOutputError
from authoring.review.reviewer import ContentReviewer, FakeContentReviewer, ReviewerUnavailableError
from authoring.review.service import review_candidate
from evaluation.review_calibration_schemas import (
    CalibrationCase,
    CalibrationCaseResult,
    CalibrationReport,
)
from taxonomy.loader import course_paths, load_skills
from taxonomy.schemas import ReferenceProvenance, SkillDefinition

REPO_ROOT = Path(__file__).resolve().parents[1]
APPROVED_BANK_PATH = REPO_ROOT / "outputs/approved_banks/pilot-approved-bank-38-v1.jsonl"
COST_PER_REVIEWER_CALL_USD = 0.01
FIXED_TIME = datetime(2026, 8, 11, 12, 0, tzinfo=timezone.utc)

CRITICAL_MUTATION_TYPES = frozenset(
    {"swap_correct_answer", "remove_grounding_reference", "heuristic_wording_conflation"}
)


def _load_bank_questions() -> list[tuple[str, str, QuizQuestion]]:
    """(item_id, skill_id, question) for every item in the static 38-item bank."""
    entries = []
    for line in APPROVED_BANK_PATH.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        entries.append(
            (record["item_id"], record["skill_id"], QuizQuestion.model_validate(record["question"]))
        )
    return entries


def mutate_swap_correct_answer(question: QuizQuestion) -> QuizQuestion:
    other = next(option for option in question.options if option != question.correct_answer)
    return question.model_copy(update={"correct_answer": other})


def mutate_unsupported_explanation(question: QuizQuestion) -> QuizQuestion:
    return question.model_copy(
        update={
            "explanation": question.explanation
            + " This is also true because of an unrelated, unverifiable claim."
        }
    )


def mutate_duplicate_distractor(question: QuizQuestion) -> QuizQuestion:
    options = list(question.options)
    target = next(index for index, option in enumerate(options) if option != question.correct_answer)
    source = next(
        index
        for index, option in enumerate(options)
        if index != target and option != question.correct_answer
    )
    options[target] = options[source] + ", restated"
    return question.model_copy(update={"options": options})


def mutate_objective_mismatch(question: QuizQuestion) -> QuizQuestion:
    return question.model_copy(update={"concept": "a topic outside the declared skill's objective"})


def mutate_wrong_difficulty(question: QuizQuestion) -> QuizQuestion:
    order = ["introductory", "intermediate", "advanced"]
    return question.model_copy(
        update={"difficulty": order[(order.index(question.difficulty) + 1) % len(order)]}
    )


MUTATIONS: dict[str, Callable[[QuizQuestion], QuizQuestion]] = {
    "swap_correct_answer": mutate_swap_correct_answer,
    "unsupported_explanation": mutate_unsupported_explanation,
    "duplicate_distractor": mutate_duplicate_distractor,
    "objective_mismatch": mutate_objective_mismatch,
    "wrong_difficulty": mutate_wrong_difficulty,
}

HEURISTIC_ORIGINAL = IntentQuestion(
    intent_id="AI-SRC-08-INT-CALIB",
    question=(
        "State the heuristic value h(n) from the following description: we have "
        "already travelled 3 units and our estimated remaining distance to the goal "
        "is 5 units. What is the heuristic value h(n)?"
    ),
    options=[
        "The cost of the cheapest path from the initial state to the goal state",
        "The accumulated cost from the start, which is 3 units",
        "The total cost from start to goal, which is 8 units",
        "The path cost, which is the sum of the accumulated cost and the estimated remaining cost",
    ],
    correct_answer="The cost of the cheapest path from the initial state to the goal state",
    explanation=(
        "The heuristic h(n) is the cost of the cheapest path from the initial state to "
        "the goal state."
    ),
    concept="heuristic value h(n)",
    difficulty="introductory",
)

HEURISTIC_CORRECTED = HEURISTIC_ORIGINAL.model_copy(
    update={
        "options": [
            "The remaining cost from the current state n to a goal state",
            *HEURISTIC_ORIGINAL.options[1:],
        ],
        "correct_answer": "The remaining cost from the current state n to a goal state",
        "explanation": (
            "The heuristic h(n) estimates the remaining cost from the current state n to "
            "a goal state; the 3 units already travelled are g(n), not h(n)."
        ),
    }
)


def _tag_stem(question: QuizQuestion, case_id: str) -> QuizQuestion:
    """Make a case's full content (and therefore its content_hash, and the stem
    text FakeContentReviewer keys on) unique to that case.

    Several calibration cases deliberately share almost everything with another
    case -- a mutation changes one field but leaves the stem alone, and
    remove_grounding_reference reuses a positive case's question unchanged. Without
    this, review_candidate's content-hash cache and FakeContentReviewer's
    stem-keyed lookup would silently collapse those distinct cases onto one
    another's report and reviewer result.
    """
    # Prefixed, not suffixed: a suffix would corrupt a stem's trailing "?" or
    # imperative structure, which authoring.question_quality.generic_quality_issues'
    # non_question_stem check (reused by authoring/review/deterministic.py) requires.
    return question.model_copy(update={"question": f"[{case_id}] {question.question}"})


def build_positive_cases() -> list[CalibrationCase]:
    cases = [
        CalibrationCase(
            case_id=item_id,
            label="positive",
            skill_id=skill_id,
            intent_id=f"{skill_id}-INT-CALIB",
            question=_tag_stem(question, item_id),
        )
        for item_id, skill_id, question in _load_bank_questions()
    ]
    cases.append(
        CalibrationCase(
            case_id="heuristic-corrected-revision",
            label="positive",
            skill_id="AI-SRC-08",
            intent_id="AI-SRC-08-INT-CALIB",
            question=_tag_stem(
                QuizQuestion.model_validate(HEURISTIC_CORRECTED.model_dump()),
                "heuristic-corrected-revision",
            ),
        )
    )
    return cases


def build_negative_cases(*, mutation_sample_size: int = 7) -> list[CalibrationCase]:
    bank = _load_bank_questions()
    cases = [
        CalibrationCase(
            case_id="heuristic-original-regression",
            label="negative",
            skill_id="AI-SRC-08",
            intent_id="AI-SRC-08-INT-CALIB",
            question=_tag_stem(
                QuizQuestion.model_validate(HEURISTIC_ORIGINAL.model_dump()),
                "heuristic-original-regression",
            ),
            mutation_type="heuristic_wording_conflation",
        )
    ]
    mutation_names = list(MUTATIONS)
    for index in range(min(mutation_sample_size, len(mutation_names), len(bank))):
        item_id, skill_id, question = bank[index]
        mutation_type = mutation_names[index % len(mutation_names)]
        case_id = f"{item_id}-{mutation_type}"
        cases.append(
            CalibrationCase(
                case_id=case_id,
                label="negative",
                skill_id=skill_id,
                intent_id=f"{skill_id}-INT-CALIB",
                question=_tag_stem(MUTATIONS[mutation_type](question), case_id),
                mutation_type=mutation_type,
            )
        )
    removal_item_id, removal_skill_id, removal_question = bank[-1]
    removal_case_id = f"{removal_item_id}-remove_grounding_reference"
    cases.append(
        CalibrationCase(
            case_id=removal_case_id,
            label="negative",
            skill_id=removal_skill_id,
            intent_id=f"{removal_skill_id}-INT-CALIB",
            question=_tag_stem(removal_question, removal_case_id),
            mutation_type="remove_grounding_reference",
        )
    )
    return cases


def _clean_result(correct_answer: str) -> SemanticReviewResult:
    return SemanticReviewResult(
        grounding_assessment=GroundingAssessment(
            grounded=True, independently_supported_answer=True, grounding_confidence=0.95
        ),
        answer_assessment=AnswerAssessment(
            selected_option_text=correct_answer,
            matches_declared_answer=True,
            multiple_defensible_answers=False,
            obviously_signalled_answer=False,
            answer_confidence=0.95,
        ),
        objective_assessment=ObjectiveAssessment(
            measures_declared_skill=True,
            satisfies_intent_blueprint=True,
            matches_objective_verb=True,
            cognitive_demand="understand",
            duplicates_another_intent=False,
        ),
        difficulty_assessment=DifficultyAssessment(
            difficulty_justified=True,
            explanation_depth_matches_difficulty=True,
            is_definition_recall_only=False,
        ),
        duplicate_assessment=DuplicateAssessment(),
        reviewer_model_id="calibration-reference-reviewer",
        reviewer_model_revision="v1",
        reviewer_prompt_version="review-v1",
        reviewer_prompt_template_hash="e" * 64,
        rendered_review_request_hash="e" * 64,
    )


def reference_result_for(case: CalibrationCase) -> SemanticReviewResult:
    """The judgment an accurate reviewer should reach for a labeled case. This is
    what makes a fakes-only calibration run meaningful: it exercises the exact same
    scoring path a real Modal-backed reviewer's output would, with a known-correct
    input standing in for the model call."""
    if case.label == "positive":
        return _clean_result(case.question.correct_answer)

    clean = _clean_result(case.question.correct_answer)
    if case.mutation_type == "swap_correct_answer":
        original = next(
            option for option in case.question.options if option != case.question.correct_answer
        )
        return clean.model_copy(
            update={
                "answer_assessment": clean.answer_assessment.model_copy(
                    update={
                        "selected_option_text": original,
                        "matches_declared_answer": False,
                    }
                )
            }
        )
    if case.mutation_type == "heuristic_wording_conflation":
        return clean.model_copy(
            update={
                "grounding_assessment": clean.grounding_assessment.model_copy(
                    update={
                        "grounded": False,
                        "independently_supported_answer": False,
                        "contradictions": [
                            "declared answer describes path cost, not h(n)'s remaining-cost estimate"
                        ],
                        "grounding_confidence": 0.2,
                    }
                ),
                "answer_assessment": clean.answer_assessment.model_copy(
                    update={
                        "selected_option_text": (
                            "The remaining cost from the current state n to a goal state"
                        ),
                        "matches_declared_answer": False,
                    }
                ),
            }
        )
    if case.mutation_type == "unsupported_explanation":
        return clean.model_copy(
            update={
                "grounding_assessment": clean.grounding_assessment.model_copy(
                    update={"unsupported_claims": ["the added claim has no supporting reference"]}
                )
            }
        )
    if case.mutation_type == "duplicate_distractor":
        return clean.model_copy(
            update={
                "answer_assessment": clean.answer_assessment.model_copy(
                    update={"duplicate_or_rephrased_distractors": [case.question.options[1]]}
                )
            }
        )
    if case.mutation_type == "objective_mismatch":
        return clean.model_copy(
            update={
                "objective_assessment": clean.objective_assessment.model_copy(
                    update={"measures_declared_skill": False}
                )
            }
        )
    if case.mutation_type == "wrong_difficulty":
        return clean.model_copy(
            update={
                "difficulty_assessment": clean.difficulty_assessment.model_copy(
                    update={"difficulty_justified": False}
                )
            }
        )
    # remove_grounding_reference is caught deterministically before any reviewer
    # call, so its reference result is never actually consulted; return clean as a
    # harmless default in case a future mutation type reaches this branch.
    return clean


def build_reference_reviewer(cases: list[CalibrationCase]) -> FakeContentReviewer:
    return FakeContentReviewer(
        {case.question.question: reference_result_for(case) for case in cases}
    )


def _build_candidate(case: CalibrationCase, reference: ReferenceProvenance) -> PendingQuestion:
    # reference_ids always names a real reference (PendingQuestion requires at least
    # one); "remove_grounding_reference" is instead simulated in run_calibration by
    # passing an empty approved_references list, so the deterministic
    # approved_references_only check fails on a reference id that isn't approved --
    # not by violating this schema's own non-empty invariant.
    return PendingQuestion(
        batch_id="calibration",
        question_id=case.case_id,
        skill_id=case.skill_id,
        question_index=0,
        intent_id=case.intent_id,
        seed=1,
        reference_ids=[reference.reference_id],
        prompt_version="v3.3",
        prompt_hash="f" * 64,
        model_id="calibration-generator",
        model_revision="v1",
        generation_parameters={},
        generated_at=FIXED_TIME,
        git_commit="0" * 40,
        raw_response="{}",
        question=IntentQuestion.model_validate(
            {**case.question.model_dump(), "intent_id": case.intent_id}
        ),
    )


def run_calibration(
    cases: list[CalibrationCase],
    *,
    reviewer_factory: Callable[[], ContentReviewer],
    config: ReviewPolicyConfig | None = None,
    report_store_path: Path,
) -> CalibrationReport:
    config = config or ReviewPolicyConfig()
    catalogue_skills = {
        skill.skill_id: skill for skill in load_skills(*course_paths("ai")).skills
    }
    reviewer = reviewer_factory()
    report_store = AutomatedReviewReportStore(report_store_path)
    results: list[CalibrationCaseResult] = []
    total_calls = 0

    for case in cases:
        skill = catalogue_skills.get(case.skill_id) or SkillDefinition(
            skill_id=case.skill_id,
            topic="Calibration",
            subtopic="Calibration",
            name=case.skill_id,
            learning_objective="Calibration placeholder objective.",
            cognitive_process="understand",
            generation_strategy="generated",
        )
        intent = QuestionIntent(
            intent_id=case.intent_id,
            skill_id=case.skill_id,
            assessment_focus="Calibration case",
            question_archetype="calibration",
            preferred_reference_ids=[f"{case.skill_id}-CALIB-REF"],
            required_concepts=["calibration"],
            prohibited_conflations=["none"],
        )
        reference = ReferenceProvenance(
            reference_id=f"{case.skill_id}-CALIB-REF",
            skill_id=case.skill_id,
            reference_material="Calibration placeholder reference material.",
            title="Calibration reference",
            source_url="https://example.edu/calibration",
            source_domain="example.edu",
            content_hash="a" * 64,
            retrieved_at=FIXED_TIME,
            reviewer_id="calibration",
            reviewed_at=FIXED_TIME,
        )
        candidate = _build_candidate(case, reference)
        # "remove_grounding_reference" cases cite a reference id that simply isn't in
        # the approved set -- that's what makes approved_references_only block them.
        approved_references = (
            [] if case.mutation_type == "remove_grounding_reference" else [reference]
        )

        parser_failure = False
        try:
            report = review_candidate(
                candidate,
                skill,
                intent,
                approved_references,
                reviewer=reviewer,
                config=config,
                report_store=report_store,
                clock=lambda: FIXED_TIME,
            )
        except ReviewerUnavailableError:
            report = None
        if report is None:
            results.append(
                CalibrationCaseResult(
                    case_id=case.case_id,
                    label=case.label,
                    mutation_type=case.mutation_type,
                    recommendation="require_full_human_review",
                    risk_level="critical",
                    reviewer_calls=0,
                    parser_failure=True,
                    disagreement=False,
                )
            )
            continue

        if report.reviewer_model_id != "n/a":
            total_calls += 1
        results.append(
            CalibrationCaseResult(
                case_id=case.case_id,
                label=case.label,
                mutation_type=case.mutation_type,
                recommendation=report.recommendation,
                risk_level=report.risk_level,
                reviewer_calls=1 if report.reviewer_model_id != "n/a" else 0,
                parser_failure=parser_failure,
                disagreement=False,
            )
        )

    return _summarize(results, reviewer_calls=total_calls)


def _summarize(results: list[CalibrationCaseResult], *, reviewer_calls: int) -> CalibrationReport:
    positives = [result for result in results if result.label == "positive"]
    negatives = [result for result in results if result.label == "negative"]

    true_positive = sum(1 for r in positives if r.recommendation == "recommend_human_approval")
    false_positive = sum(1 for r in negatives if r.recommendation == "recommend_human_approval")
    false_negative = len(positives) - true_positive

    precision = (
        true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 1.0
    )
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 1.0

    critical_negatives = [r for r in negatives if r.mutation_type in CRITICAL_MUTATION_TYPES]
    critical_detected = sum(1 for r in critical_negatives if r.risk_level == "critical")
    critical_error_detection_rate = (
        critical_detected / len(critical_negatives) if critical_negatives else 1.0
    )

    false_low_risk = sum(1 for r in negatives if r.risk_level == "low")
    false_low_risk_rate = false_low_risk / len(negatives) if negatives else 0.0

    disagreement_rate = (
        sum(1 for r in results if r.disagreement) / len(results) if results else 0.0
    )
    parser_failure_rate = (
        sum(1 for r in results if r.parser_failure) / len(results) if results else 0.0
    )

    return CalibrationReport(
        total_cases=len(results),
        positive_cases=len(positives),
        negative_cases=len(negatives),
        approval_precision=precision,
        approval_recall=recall,
        critical_error_detection_rate=critical_error_detection_rate,
        false_low_risk_rate=false_low_risk_rate,
        disagreement_rate=disagreement_rate,
        parser_failure_rate=parser_failure_rate,
        reviewer_calls=reviewer_calls,
        estimated_cost_usd=reviewer_calls * COST_PER_REVIEWER_CALL_USD,
        results=results,
    )
