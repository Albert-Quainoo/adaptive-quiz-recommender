"""Offline calibration batch: a small, fully offline dry run of the real
automated-review pipeline against known-good and deliberately-flawed candidates
spanning several distinct failure types, ahead of turning on replenishment shadow
mode for real.

Naming note: "shadow mode" is reserved for real replenishment candidates reviewed
live without allowing automatic revision, approval, or promotion. This script makes
zero live/network calls -- every reviewer response is a canned, synthetic payload --
so it is calibration tooling, not shadow mode itself.

Unlike evaluation/review_calibration.py's full precision/recall harness (which uses
FakeContentReviewer and never touches raw JSON), every case here goes through
ModelBackedContentReviewer wrapped around a fake, JSON-emitting BatchModel -- so the
actual prompt-building, strict parsing, schema/semantic validation, and
derive_assessments() mapping this milestone shipped are all exercised end to end,
with zero network calls and zero cost. Reuses evaluation/review_calibration.py's bank
loading and question-mutation helpers rather than re-deriving flawed candidates from
scratch.

Never calls propose_automated_revision, approve_revision, approve_as_written, or any
promotion path -- only authoring.review.service.review_candidate(), which can only
ever produce a recommendation and risk assessment for a human to act on.

    python -m scripts.run_offline_calibration_batch
    python -m scripts.run_offline_calibration_batch --output outputs/offline_calibration_batch.json
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from api.schemas import QuizQuestion
from authoring.grounded_batch import GenerationOutcome
from authoring.grounded_review import question_content_hash
from authoring.question_intents import QuestionIntent
from authoring.review.config import ReviewPolicyConfig
from authoring.review.reports import AutomatedReviewReportStore
from authoring.review.reviewer import ModelBackedContentReviewer
from authoring.review.service import review_candidate
from evaluation.review_calibration import MUTATIONS, _build_candidate, _load_bank_questions, _tag_stem
from taxonomy.schemas import ReferenceProvenance, SkillDefinition

FIXED_TIME = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)


class _FakeJsonBatchModel:
    """A BatchModel that always returns one canned compact-JSON response, with
    GenerationOutcome metadata attached -- makes zero network calls. Distinct from
    evaluation.review_calibration.FakeContentReviewer: this stands in one level lower,
    at the raw-model layer, so ModelBackedContentReviewer's real prompt-building and
    parse_reviewer_output/derive_assessments still run against it."""

    model_id = "offline-calibration-reference-model"
    model_revision = "offline-calibration-v1"

    def __init__(self, raw_response: str) -> None:
        self._raw_response = raw_response
        self.calls = 0

    def generate_with_metadata(self, messages, seed, generation_parameters) -> GenerationOutcome:
        self.calls += 1
        return GenerationOutcome(
            text=self._raw_response,
            finish_reason="stop",
            input_tokens=len(str(messages)) // 4,
            output_tokens=len(self._raw_response) // 4,
            max_new_tokens=generation_parameters.get("max_new_tokens", 1000),
        )


def _compact_payload(
    question: QuizQuestion,
    *,
    grounded: bool = True,
    consulted_reference_ids: list[str],
    supporting_reference_ids: list[str] | None = None,
    selected_option_index: int | None = 0,
    no_defensible_option: bool = False,
    independent_answer_text: str | None = None,
    declared_answer_matches: bool = True,
    multiple_defensible_answers: bool = False,
    unsupported_claims: list[str] | None = None,
    contradictions: list[str] | None = None,
    objective_aligned: bool = True,
    intent_aligned: bool = True,
    difficulty_appropriate: bool = True,
    duplicate_option_pairs: list[tuple[int, int]] | None = None,
    option_assessments: list[tuple[int, str]] | None = None,
    confidence: float = 0.9,
) -> dict:
    if supporting_reference_ids is None:
        supporting_reference_ids = list(consulted_reference_ids) if grounded else []
    if independent_answer_text is None:
        independent_answer_text = (
            question.options[selected_option_index]
            if selected_option_index is not None
            else question.correct_answer
        )
    if option_assessments is None:
        # Every case below only ever names its one selected/correct option -- the
        # rest default to "incorrect" (or all "incorrect" for no_defensible_option).
        option_assessments = [
            (index, "incorrect" if no_defensible_option else (
                "correct" if index == selected_option_index else "incorrect"
            ))
            for index in range(len(question.options))
        ]
    return {
        "grounded": grounded,
        "consulted_reference_ids": consulted_reference_ids,
        "supporting_reference_ids": supporting_reference_ids,
        "selected_option_index": selected_option_index,
        "independent_answer_text": independent_answer_text,
        "no_defensible_option": no_defensible_option,
        "declared_answer_matches": declared_answer_matches,
        "multiple_defensible_answers": multiple_defensible_answers,
        "option_assessments": [list(pair) for pair in option_assessments],
        "unsupported_claims": unsupported_claims or [],
        "contradictions": contradictions or [],
        "objective_aligned": objective_aligned,
        "intent_aligned": intent_aligned,
        "difficulty_appropriate": difficulty_appropriate,
        "duplicate_option_pairs": [list(pair) for pair in (duplicate_option_pairs or [])],
        "confidence": confidence,
        "blocking_reasons": [],
        "warnings": [],
    }


class CalibrationBatchCase:
    def __init__(
        self,
        case_id: str,
        *,
        known_label: str,
        failure_type: str,
        question: QuizQuestion,
        skill_id: str,
        reference_payload: dict,
    ) -> None:
        self.case_id = case_id
        self.known_label = known_label
        self.failure_type = failure_type
        self.question = _tag_stem(question, case_id)
        self.skill_id = skill_id
        self.reference_payload = reference_payload


def build_cases() -> list[CalibrationBatchCase]:
    bank = _load_bank_questions()
    good_1_id, good_1_skill, good_1_question = bank[0]
    good_2_id, good_2_skill, good_2_question = bank[1]

    swap_id, swap_skill, swap_question = bank[2]
    swapped = MUTATIONS["swap_correct_answer"](swap_question)
    original_correct_index = swapped.options.index(swap_question.correct_answer)

    unsupported_id, unsupported_skill, unsupported_question = bank[3]
    unsupported = MUTATIONS["unsupported_explanation"](unsupported_question)

    duplicate_id, duplicate_skill, duplicate_question = bank[4]
    duplicated = MUTATIONS["duplicate_distractor"](duplicate_question)
    # mutate_duplicate_distractor's own selection logic (evaluation/review_calibration.
    # py), recomputed here since the mutation only returns the mutated question, not
    # which two indices it duplicated:
    _duplicate_target = next(
        index for index, option in enumerate(duplicate_question.options)
        if option != duplicate_question.correct_answer
    )
    _duplicate_source = next(
        index
        for index, option in enumerate(duplicate_question.options)
        if index != _duplicate_target and option != duplicate_question.correct_answer
    )

    mismatch_id, mismatch_skill, mismatch_question = bank[5]
    mismatched = MUTATIONS["objective_mismatch"](mismatch_question)

    wrong_difficulty_id, wrong_difficulty_skill, wrong_difficulty_question = bank[6]
    wrong_difficulty = MUTATIONS["wrong_difficulty"](wrong_difficulty_question)

    no_option_id, no_option_skill, no_option_question = bank[7]

    return [
        CalibrationBatchCase(
            f"{good_1_id}-known-good",
            known_label="good",
            failure_type="none",
            question=good_1_question,
            skill_id=good_1_skill,
            reference_payload=_compact_payload(
                good_1_question,
                consulted_reference_ids=["CALIB-REF"],
                selected_option_index=good_1_question.options.index(good_1_question.correct_answer),
            ),
        ),
        CalibrationBatchCase(
            f"{good_2_id}-known-good",
            known_label="good",
            failure_type="none",
            question=good_2_question,
            skill_id=good_2_skill,
            reference_payload=_compact_payload(
                good_2_question,
                consulted_reference_ids=["CALIB-REF"],
                selected_option_index=good_2_question.options.index(good_2_question.correct_answer),
            ),
        ),
        CalibrationBatchCase(
            f"{swap_id}-swapped-answer",
            known_label="flawed",
            failure_type="swap_correct_answer",
            question=swapped,
            skill_id=swap_skill,
            reference_payload=_compact_payload(
                swapped,
                consulted_reference_ids=["CALIB-REF"],
                selected_option_index=original_correct_index,
                declared_answer_matches=False,
            ),
        ),
        CalibrationBatchCase(
            f"{unsupported_id}-unsupported-explanation",
            known_label="flawed",
            failure_type="unsupported_explanation",
            question=unsupported,
            skill_id=unsupported_skill,
            reference_payload=_compact_payload(
                unsupported,
                consulted_reference_ids=["CALIB-REF"],
                selected_option_index=unsupported.options.index(unsupported.correct_answer),
                unsupported_claims=["the added claim has no supporting reference"],
            ),
        ),
        CalibrationBatchCase(
            f"{duplicate_id}-duplicate-distractor",
            known_label="flawed",
            failure_type="duplicate_distractor",
            question=duplicated,
            skill_id=duplicate_skill,
            reference_payload=_compact_payload(
                duplicated,
                consulted_reference_ids=["CALIB-REF"],
                selected_option_index=duplicated.options.index(duplicated.correct_answer),
                # The regression this whole field exists to catch: two options that
                # say the same thing in different words (deterministic exact-text
                # matching cannot catch this -- see authoring/review/models.py's
                # CompactReviewResult.duplicate_option_pairs).
                duplicate_option_pairs=[(_duplicate_target, _duplicate_source)],
            ),
        ),
        CalibrationBatchCase(
            f"{mismatch_id}-objective-mismatch",
            known_label="flawed",
            failure_type="objective_mismatch",
            question=mismatched,
            skill_id=mismatch_skill,
            reference_payload=_compact_payload(
                mismatched,
                consulted_reference_ids=["CALIB-REF"],
                selected_option_index=mismatched.options.index(mismatched.correct_answer),
                objective_aligned=False,
                intent_aligned=False,
            ),
        ),
        CalibrationBatchCase(
            f"{wrong_difficulty_id}-wrong-difficulty",
            known_label="flawed",
            failure_type="wrong_difficulty",
            question=wrong_difficulty,
            skill_id=wrong_difficulty_skill,
            reference_payload=_compact_payload(
                wrong_difficulty,
                consulted_reference_ids=["CALIB-REF"],
                selected_option_index=wrong_difficulty.options.index(wrong_difficulty.correct_answer),
                difficulty_appropriate=False,
            ),
        ),
        CalibrationBatchCase(
            f"{no_option_id}-no-defensible-option",
            known_label="flawed",
            failure_type="no_defensible_option",
            question=no_option_question,
            skill_id=no_option_skill,
            reference_payload=_compact_payload(
                no_option_question,
                grounded=False,
                consulted_reference_ids=["CALIB-REF"],
                supporting_reference_ids=[],
                selected_option_index=None,
                no_defensible_option=True,
                independent_answer_text="an answer not defensibly among the four options",
                declared_answer_matches=False,
            ),
        ),
    ]


def run_offline_calibration(cases: list[CalibrationBatchCase], *, report_store_path: Path) -> list[dict]:
    config = ReviewPolicyConfig()
    report_store = AutomatedReviewReportStore(report_store_path)
    results = []

    for case in cases:
        skill = SkillDefinition(
            skill_id=case.skill_id,
            topic="Offline calibration",
            subtopic="Offline calibration",
            name=case.skill_id,
            learning_objective="Offline calibration placeholder objective.",
            cognitive_process="understand",
            generation_strategy="generated",
        )
        intent = QuestionIntent(
            intent_id=f"{case.skill_id}-INT-CALIB",
            skill_id=case.skill_id,
            assessment_focus="Offline calibration case",
            question_archetype="offline-calibration",
            preferred_reference_ids=["CALIB-REF"],
            required_concepts=["offline-calibration"],
            prohibited_conflations=["none"],
            cognitive_demand="understand",
        )
        reference = ReferenceProvenance(
            reference_id="CALIB-REF",
            skill_id=case.skill_id,
            reference_material="Offline calibration placeholder reference material.",
            title="Offline calibration reference",
            source_url="https://example.edu/offline-calibration",
            source_domain="example.edu",
            content_hash="b" * 64,
            retrieved_at=FIXED_TIME,
            reviewer_id="offline-calibration",
            reviewed_at=FIXED_TIME,
        )

        class _Case:
            question = case.question
            case_id = case.case_id
            skill_id = case.skill_id
            intent_id = intent.intent_id

        candidate = _build_candidate(_Case(), reference)
        raw_response = json.dumps(case.reference_payload)
        model = _FakeJsonBatchModel(raw_response)
        reviewer = ModelBackedContentReviewer(model, max_new_tokens=config.reviewer_max_new_tokens)

        report = review_candidate(
            candidate,
            skill,
            intent,
            [reference],
            reviewer=reviewer,
            config=config,
            report_store=report_store,
            clock=lambda: FIXED_TIME,
        )

        results.append(
            {
                "case_id": case.case_id,
                "known_label": case.known_label,
                "failure_type": case.failure_type,
                "content_hash": question_content_hash(case.question),
                "reviewer_request_count": report.reviewer_request_count,
                "fake_reviewer_calls_made": model.calls,
                "recommendation": report.recommendation,
                "risk_level": report.risk_level,
                "risk_score": report.risk_score,
                "blocking_reasons": report.blocking_reasons,
                "warnings": report.warnings,
                "consulted_reference_ids": report.consulted_reference_ids,
                "supporting_reference_ids": report.supporting_reference_ids,
                "matches_human_expectation": (report.recommendation == "recommend_human_approval")
                == (case.known_label == "good"),
            }
        )

    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--report-store",
        type=Path,
        default=Path("outputs/offline_calibration_batch_reports.json"),
        help="Scratch AutomatedReviewReport store for this run (overwritten each run).",
    )
    arguments = parser.parse_args(argv)

    cases = build_cases()
    arguments.report_store.parent.mkdir(parents=True, exist_ok=True)
    arguments.report_store.unlink(missing_ok=True)
    results = run_offline_calibration(cases, report_store_path=arguments.report_store)

    total_fake_calls = sum(result["fake_reviewer_calls_made"] for result in results)
    print(f"Offline calibration cases: {len(results)}")
    print(
        f"Fake reviewer calls made: {total_fake_calls}  "
        "(all synthetic -- zero real network calls, zero cost)"
    )
    print()
    header = f"{'case_id':<38} {'label':<7} {'failure_type':<22} {'recommendation':<24} {'risk':<9} {'match':<5}"
    print(header)
    print("-" * len(header))
    mismatches = []
    for result in results:
        match = "yes" if result["matches_human_expectation"] else "NO"
        if not result["matches_human_expectation"]:
            mismatches.append(result)
        print(
            f"{result['case_id']:<38} {result['known_label']:<7} {result['failure_type']:<22} "
            f"{result['recommendation']:<24} {result['risk_level']:<9} {match:<5}"
        )
        if result["blocking_reasons"]:
            for reason in result["blocking_reasons"]:
                print(f"    blocking: {reason}")
        if result["warnings"]:
            for warning in result["warnings"]:
                print(f"    warning: {warning}")

    if mismatches:
        print(
            f"\n{len(mismatches)} case(s) disagreed with the known label -- fix before "
            "starting replenishment shadow mode."
        )
    else:
        print(f"\n{len(results)}/{len(results)} cases matched their known label.")

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(
            json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"\nFull results written to {arguments.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
