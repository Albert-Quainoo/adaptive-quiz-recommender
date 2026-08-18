"""Negative-case proof for the generation/review/promotion pilot
(scripts/pilot_capacity_generation_review_promotion.py).

Per the pilot's task instructions, every required negative case must be proven,
but an existing test that already proves the same invariant is reused rather than
duplicated. Only genuinely uncovered cases get a new test here; the rest are
reused from their existing home:

  * malformed generation JSON
      -> tests/test_grounded_batch.py::test_invalid_raw_output_is_retained_and_retried
  * incorrect correct_answer
      -> NEW: test_incorrect_correct_answer_is_rejected_and_retried (below)
  * duplicate or near-duplicate question
      -> tests/test_grounded_batch.py::test_duplicate_questions_are_rejected_and_bounded
      -> tests/test_review_deterministic.py::test_exact_duplicate_question_text_is_blocking
  * wrong skill or intent
      -> tests/test_review_deterministic.py::test_mismatched_skill_is_blocking
      -> tests/test_review_risk.py::test_objective_mismatch_raises_risk
  * unsupported or invalid reference provenance
      -> NEW: test_invalid_preferred_reference_fails_closed_before_any_model_call (below)
  * reviewer output with an answer not present in the candidate options
      -> tests/test_review_response_parser.py::test_out_of_range_selected_option_index_fails_schema_validation
  * reviewer schema failure after allowed repair attempts
      -> tests/test_review_service.py::test_no_promotion_after_any_parse_failure
      -> tests/test_review_response_parser.py (truncation/schema-failure suite)
  * promotion attempted without the required review state
      -> tests/test_replenishment_bank_promotion.py::test_no_approved_items_is_a_permanent_failure
      -> tests/test_grounded_review.py::test_pending_and_rejected_items_are_excluded_from_export
"""

import json
from datetime import datetime, timezone

import pytest

from api.prompt_builder import PROMPT_VERSION
from authoring.grounded_batch import BatchGenerationError, generate_batch
from authoring.question_intents import PilotBlueprint, QuestionIntent
from taxonomy.loader import course_paths, course_provenance_path

from tests.test_grounded_batch import DeterministicFakeModel, config, raw_question

SKILLS_PATH, REFERENCES_PATH = course_paths("ai")
PROVENANCE_PATH = course_provenance_path("ai")
FIXED_TIME = datetime(2026, 8, 18, 9, 0, tzinfo=timezone.utc)
GIT_COMMIT = "b" * 40


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_incorrect_correct_answer_is_rejected_and_retried(tmp_path):
    """A generated candidate whose correct_answer does not match any of its own
    options is a deterministic validation failure (validate_question), never an
    accepted question -- the generator must reject it and retry, exactly like any
    other invalid attempt."""
    bad_answer = lambda seed: raw_question(seed, correct_answer="Not one of the listed options")
    model = DeterministicFakeModel([bad_answer, lambda seed: raw_question(seed)])
    output = tmp_path / "batch"
    generate_batch(
        config(skill_ids=["AI-SRC-08"], questions_per_skill=1),
        model,
        output,
        skills_path=SKILLS_PATH,
        references_path=REFERENCES_PATH,
        provenance_path=PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
    )
    attempts = jsonl(output / "audit.jsonl")

    assert attempts[0]["validation_status"] == "invalid"
    assert "correct answer is not in options" in attempts[0]["validation_error"].lower()
    assert attempts[1]["validation_status"] == "accepted"


def test_invalid_preferred_reference_fails_closed_before_any_model_call(tmp_path, monkeypatch):
    """A blueprint intent naming a preferred_reference_id that does not exist (or
    belongs to a different skill) in the taxonomy's reviewed reference provenance is
    a deterministic configuration defect: generate_batch must fail before ever
    calling the model, not after a wasted generation attempt."""
    import authoring.question_intents as question_intents

    blueprint_dir = tmp_path / "blueprints"
    blueprint_dir.mkdir()
    bad_intent = QuestionIntent(
        intent_id="AI-SRC-08-INT-99",
        skill_id="AI-SRC-08",
        assessment_focus="A focus that cites a reference the taxonomy never approved",
        question_archetype="test-only invalid-reference probe",
        preferred_reference_ids=["AI-SRC-08-does-not-exist"],
        required_concepts=["placeholder"],
        prohibited_conflations=["placeholder"],
        difficulty="intermediate",
    )
    blueprint = PilotBlueprint(
        batch_id="test-invalid-reference-batch",
        prompt_version=PROMPT_VERSION,
        review_status="blueprint-approved",
        intents=[bad_intent],
    )
    (blueprint_dir / "test-invalid-reference-batch.json").write_text(
        json.dumps(blueprint.model_dump(mode="json")), encoding="utf-8"
    )
    monkeypatch.setattr(question_intents, "BLUEPRINT_DIRECTORY", blueprint_dir)

    model = DeterministicFakeModel()
    with pytest.raises(BatchGenerationError, match="invalid preferred references"):
        generate_batch(
            config(
                batch_id="test-invalid-reference-batch",
                skill_ids=["AI-SRC-08"],
                questions_per_skill=1,
            ),
            model,
            tmp_path / "output",
            skills_path=SKILLS_PATH,
            references_path=REFERENCES_PATH,
            provenance_path=PROVENANCE_PATH,
            clock=lambda: FIXED_TIME,
            git_commit=GIT_COMMIT,
        )
    assert model.calls == []
