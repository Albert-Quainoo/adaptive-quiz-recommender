import json
from datetime import datetime, timezone

import pytest

from api.prompt_builder import build_grounded_quiz_messages
from api.schemas import QuizGenerationRequest, QuizQuestion
from authoring.grounded_review import (
    CurationItem,
    GroundedReview,
    GroundedReviewStore,
    RevisionProvenance,
    approve_revision,
    approved_item_provenance,
    assert_immutable_source,
    export_approved_bank_items,
    generic_quality_issues,
    inspect_question,
    propose_revision,
    reject_item,
    source_hashes,
    write_approved_bank,
)
from authoring.grounding_briefs import grounding_brief
from authoring.pilot_curation import PROPOSED_QUESTIONS, RECOMMENDATIONS
from tests.test_grounded_batch import (
    FIXED_TIME,
    GIT_COMMIT,
    DeterministicFakeModel,
    config,
)
from authoring.grounded_batch import BatchGenerationError, generate_batch, prompt_hash
from authoring.question_intents import intents_by_skill, load_pilot_blueprint
from taxonomy.loader import course_paths, course_provenance_path


SKILLS_PATH, REFERENCES_PATH = course_paths("ai")
PROVENANCE_PATH = course_provenance_path("ai")
REVIEW_TIME = datetime(2026, 8, 6, 14, 0, tzinfo=timezone.utc)


def question(stem="Which statement is correct?", correct="Correct"):
    return QuizQuestion(
        question=stem,
        options=[correct, "Wrong A", "Wrong B", "Wrong C"],
        correct_answer=correct,
        explanation=f"{correct} is the supported choice.",
        concept="Grounded concept",
        difficulty="intermediate",
    )


def item():
    return CurationItem(
        original_question_id="AI-SRC-08-source-id",
        skill_id="AI-SRC-08",
        intent_id="AI-SRC-08-INT-02",
        recommendation="propose_revision",
        recommendation_reason="Needs a concrete calculation.",
    )


def provenance():
    return RevisionProvenance(
        source_batch_id="grounded-pilot-20260805-v2",
        intent_id="AI-SRC-08-INT-02",
        skill_id="AI-SRC-08",
        reference_ids=["AI-SRC-08-reference"],
        model_id="model",
        model_revision="revision",
        prompt_version="v3.3",
        prompt_hash="a" * 64,
    )


def proposed(base=None, edited=None):
    return propose_revision(
        base or item(),
        question(),
        edited or question("Which revised statement is correct?"),
        "editor",
        "Clarified the stem.",
        edited_at=REVIEW_TIME,
        provenance=provenance(),
    )


def review_with(*items):
    return GroundedReview(
        batch_id="grounded-pilot-20260805-v2",
        source_hashes={},
        items=list(items),
    )


def fake_source_batch(tmp_path):
    output = tmp_path / "source"
    generate_batch(
        config(skill_ids=["AI-SRC-08"], questions_per_skill=1),
        DeterministicFakeModel(),
        output,
        skills_path=SKILLS_PATH,
        references_path=REFERENCES_PATH,
        provenance_path=PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
    )
    source = json.loads((output / "pending_questions.jsonl").read_text().splitlines()[0])
    curated = CurationItem(
        original_question_id=source["question_id"],
        skill_id=source["skill_id"],
        intent_id=source["intent_id"],
        recommendation="reject",
        recommendation_reason="Fixture recommendation.",
    )
    return output, GroundedReview(
        batch_id=source["batch_id"],
        source_hashes=source_hashes(output),
        items=[curated],
    )


def test_proposing_revision_keeps_original_generation_record_immutable(tmp_path):
    batch, review = fake_source_batch(tmp_path)
    before = {path.name: path.read_bytes() for path in batch.iterdir() if path.is_file()}
    inspection = inspect_question(batch, review, review.items[0].original_question_id)

    replacement = propose_revision(
        inspection.curation,
        inspection.source_question.question,
        question("Which edited question should a reviewer consider?"),
        "editor",
        "Proposed wording only.",
        edited_at=REVIEW_TIME,
    )

    assert replacement.revisions[0].original_question_id == inspection.source_question.question_id
    assert inspection.source_question.review_status == "pending"
    assert before == {path.name: path.read_bytes() for path in batch.iterdir() if path.is_file()}
    assert_immutable_source(batch, review)


def test_completed_generated_batch_cannot_be_resumed_or_rewritten(tmp_path):
    batch, _ = fake_source_batch(tmp_path)
    before = {path.name: path.read_bytes() for path in batch.iterdir() if path.is_file()}

    with pytest.raises(BatchGenerationError, match="immutable"):
        generate_batch(
            config(skill_ids=["AI-SRC-08"], questions_per_skill=1),
            DeterministicFakeModel(),
            batch,
            skills_path=SKILLS_PATH,
            references_path=REFERENCES_PATH,
            provenance_path=PROVENANCE_PATH,
            clock=lambda: FIXED_TIME,
            git_commit=GIT_COMMIT,
            resume=True,
        )

    assert before == {path.name: path.read_bytes() for path in batch.iterdir() if path.is_file()}


def test_revision_history_retains_each_proposal_and_changed_fields():
    first = proposed()
    second = propose_revision(
        first,
        question(),
        question("Which second revision is clearer?", correct="Clear"),
        "second-editor",
        "Alternative wording.",
        edited_at=REVIEW_TIME.replace(hour=15),
        provenance=provenance(),
    )

    assert len(second.revisions) == 2
    assert second.revisions[0].revision_id != second.revisions[1].revision_id
    assert second.revisions[0].final_review_status == "pending"
    assert set(second.revisions[1].changed_fields) >= {"question", "options", "correct_answer"}


def test_revision_content_hash_is_bound_to_the_edited_question():
    revision = proposed().revisions[0]
    changed = revision.model_dump()
    changed["question"] = question("A different edited question?").model_dump()

    assert revision.content_hash
    with pytest.raises(ValueError, match="content_hash"):
        revision.__class__.model_validate(changed)


def test_rejection_requires_and_records_a_reason():
    with pytest.raises(ValueError, match="reason"):
        reject_item(item(), "reviewer", "   ", reviewed_at=REVIEW_TIME)

    rejected = reject_item(item(), "reviewer", "Factually incorrect.", reviewed_at=REVIEW_TIME)
    assert rejected.final_review_status == "rejected"
    assert rejected.rejection_reason == "Factually incorrect."


def test_pending_and_rejected_items_are_excluded_from_export():
    second = item().model_copy(
        update={
            "original_question_id": "AI-SRC-08-other-source",
            "intent_id": "AI-SRC-08-INT-03",
        }
    )
    rejected = reject_item(second, "reviewer", "Not salvageable.", reviewed_at=REVIEW_TIME)
    assert export_approved_bank_items(review_with(item(), rejected)) == []


def test_only_explicitly_approved_revision_is_exported():
    candidate = proposed()
    assert export_approved_bank_items(review_with(candidate)) == []

    approved = approve_revision(
        candidate,
        candidate.revisions[0].revision_id,
        "reviewer",
        reviewed_at=REVIEW_TIME,
    )
    exported = export_approved_bank_items(review_with(approved))

    assert len(exported) == 1
    assert exported[0].question == candidate.revisions[0].question
    assert exported[0].provenance == "generated"
    assert exported[0].skill_id == "AI-SRC-08"
    assert set(exported[0].model_dump()) == {
        "item_id",
        "question",
        "provenance",
        "skill_id",
    }
    approved_revision = next(
        revision
        for revision in approved.revisions
        if revision.final_review_status == "approved"
    )
    assert approved_revision.source_batch_id == "grounded-pilot-20260805-v2"
    assert approved_revision.reference_ids == ["AI-SRC-08-reference"]
    assert approved_revision.content_hash
    assert exported[0].item_id == approved_revision.revision_id
    lookup = approved_item_provenance(review_with(approved), exported[0].item_id)
    assert lookup.item_id == exported[0].item_id
    assert lookup.source_batch_id == "grounded-pilot-20260805-v2"
    assert lookup.revision_id == approved_revision.revision_id
    assert lookup.intent_id == approved.intent_id
    assert lookup.skill_id == approved.skill_id
    assert lookup.reference_ids == approved_revision.reference_ids
    assert lookup.reviewer_id == "reviewer"


def test_revision_without_source_provenance_cannot_be_approved():
    candidate = propose_revision(
        item(),
        question(),
        question("Which legacy proposal lacks provenance?"),
        "editor",
        "Legacy proposal.",
        edited_at=REVIEW_TIME,
    )
    with pytest.raises(ValueError, match="complete source provenance"):
        approve_revision(
            candidate,
            candidate.revisions[0].revision_id,
            "reviewer",
            reviewed_at=REVIEW_TIME,
        )


def test_bank_item_ids_are_stable_across_repeated_exports():
    candidate = proposed()
    approved = approve_revision(
        candidate,
        candidate.revisions[0].revision_id,
        "reviewer",
        reviewed_at=REVIEW_TIME,
    )
    review = review_with(approved)

    first = export_approved_bank_items(review)[0]
    second = export_approved_bank_items(review)[0]
    approved_revision = next(
        revision
        for revision in approved.revisions
        if revision.final_review_status == "approved"
    )
    assert first.item_id == second.item_id == approved_revision.revision_id
    assert first.item_id != approved.original_question_id
    assert first.model_dump() == second.model_dump()


def test_repeated_file_exports_have_identical_content(tmp_path):
    candidate = proposed()
    approved = approve_revision(
        candidate,
        candidate.revisions[0].revision_id,
        "reviewer",
        reviewed_at=REVIEW_TIME,
    )
    review = review_with(approved)
    first = tmp_path / "first.jsonl"
    second = tmp_path / "second.jsonl"

    write_approved_bank(first, review)
    write_approved_bank(second, review)
    assert first.read_bytes() == second.read_bytes()


def test_inspection_retains_references_raw_response_and_attempt_provenance(tmp_path):
    batch, review = fake_source_batch(tmp_path)
    inspection = inspect_question(batch, review, review.items[0].original_question_id)

    assert inspection.references
    assert inspection.attempts
    assert inspection.source_question.raw_response
    assert inspection.source_question.prompt_hash
    assert inspection.attempts[-1]["raw_response"]


def test_review_store_preserves_revision_history(tmp_path):
    store = GroundedReviewStore(tmp_path / "review.json")
    candidate = proposed()
    store.save(review_with(candidate))

    reopened = store.load().items[0]
    assert reopened.revisions == candidate.revisions
    assert reopened.revisions[0].original_question_id == candidate.original_question_id


def test_future_manifest_records_canonical_brief_version_and_hash(tmp_path):
    output = tmp_path / "batch"
    generate_batch(
        config(skill_ids=["AI-SRC-08"], questions_per_skill=1),
        DeterministicFakeModel(),
        output,
        skills_path=SKILLS_PATH,
        references_path=REFERENCES_PATH,
        provenance_path=PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
    )
    manifest = json.loads((output / "manifest.json").read_text())
    brief = grounding_brief("AI-SRC-08")

    assert manifest["grounding_briefs"]["AI-SRC-08"] == {
        "version": brief.version,
        "content_hash": brief.content_hash,
    }


def test_canonical_brief_content_is_in_prompt_and_therefore_prompt_hash(tmp_path):
    model = DeterministicFakeModel()
    result = generate_batch(
        config(skill_ids=["AI-SRC-08"], questions_per_skill=1),
        model,
        tmp_path / "batch",
        skills_path=SKILLS_PATH,
        references_path=REFERENCES_PATH,
        provenance_path=PROVENANCE_PATH,
        clock=lambda: FIXED_TIME,
        git_commit=GIT_COMMIT,
    )
    prompt = model.calls[0]["messages"][1]["content"]
    brief = grounding_brief("AI-SRC-08")

    assert brief.version in prompt
    assert all(statement in prompt for statement in brief.statements)
    assert result.attempts[0].prompt_hash == result.questions[0].prompt_hash


def test_changing_canonical_brief_changes_future_prompt_hash():
    request = QuizGenerationRequest(
        topic="Heuristic function",
        difficulty="intermediate",
        learning_objective="Explain remaining-cost estimates.",
        question_count=1,
        reference_material=["A heuristic estimates forward cost."],
    )
    intent = intents_by_skill(load_pilot_blueprint())["AI-SRC-08"][0]
    brief = grounding_brief("AI-SRC-08")
    revised_brief = brief.model_copy(
        update={"statements": [*brief.statements, "A new reviewed constraint."]}
    )

    first = build_grounded_quiz_messages(
        request,
        question_intent=intent,
        grounding_brief=brief,
        avoid_stems=[],
    )
    second = build_grounded_quiz_messages(
        request,
        question_intent=intent,
        grounding_brief=revised_brief,
        avoid_stems=[],
    )
    assert prompt_hash(first) != prompt_hash(second)


def test_preliminary_decisions_cover_every_pilot_intent_without_approval():
    assert len(RECOMMENDATIONS) == 30
    assert len(PROPOSED_QUESTIONS) == 11
    assert sum(value[0] == "reject" for value in RECOMMENDATIONS.values()) == 19
    assert all(reason.strip() for _, reason in RECOMMENDATIONS.values())

    missing = PROPOSED_QUESTIONS["AI-SRC-01-INT-07"]
    assert missing.correct_answer == "Path-cost function"
    assert missing.options[0] == "Path-cost function"
    assert missing.question.startswith("A course formulation defines")
    manhattan = PROPOSED_QUESTIONS["AI-SRC-08-INT-02"]
    assert manhattan.explanation == (
        "Manhattan distance is the sum of the absolute horizontal and vertical "
        "coordinate differences:\n\n|5 - 2| + |5 - 1| = 3 + 4 = 7."
    )
    comparison = PROPOSED_QUESTIONS["AI-SRC-08-INT-08"]
    assert "Uniform-Cost Search/Dijkstra's algorithm" in comparison.question
    assert "Uniform-Cost Search/Dijkstra's algorithm" in comparison.correct_answer


def test_generic_quality_checks_cover_requested_deterministic_failures():
    revealed = question("Which option is Correct?")
    statement = question("A declarative statement.")
    constructed = QuizQuestion.model_construct(
        question="A search problem has four components but the explanation says five components.",
        options=["Same", " same ", "C", "D"],
        correct_answer="Missing",
        explanation="Missing is incorrect; there are five components.",
        concept="quality",
        difficulty="intermediate",
    )

    assert "answer_revealed" in {issue.code for issue in generic_quality_issues(revealed, intent_id="I")}
    assert "non_question_stem" in {issue.code for issue in generic_quality_issues(statement, intent_id="I")}
    codes = {
        issue.code
        for issue in generic_quality_issues(
            constructed,
            intent_id="I",
            accepted_intent_ids={"I"},
        )
    }
    assert codes >= {
        "repeated_intent",
        "correct_answer_match",
        "contradictory_counts",
        "duplicate_options",
        "explanation_disagreement",
    }


def test_answer_revealed_does_not_flag_enumerated_missing_component_question():
    """Regression: the stem legitimately enumerates all five components, including
    the correct answer, then asks which is missing -- that's not a giveaway, since
    the reader still has to reason about which of the first four was named. This is
    the real AI-SRC-01 approved-bank item that previously false-positived."""
    enumerated = QuizQuestion(
        question=(
            "A course formulation defines a search problem using an initial state, "
            "actions, a transition model, a goal test, and a path-cost function. If "
            "the first four have been specified, which component is missing?"
        ),
        options=["Path-cost function", "Start state", "Successor state", "Search-tree depth"],
        correct_answer="Path-cost function",
        explanation=(
            "In this formulation, the initial state, actions, transition model and "
            "goal test are already specified, leaving the path-cost function as the "
            "missing component."
        ),
        concept="Missing search-problem component",
        difficulty="intermediate",
    )
    assert "answer_revealed" not in {
        issue.code for issue in generic_quality_issues(enumerated, intent_id="I")
    }


def test_answer_revealed_still_flags_direct_assertion_alongside_an_enumeration():
    """A stem can both enumerate candidate terms AND separately assert the answer
    outright; the enumeration must not blanket-suppress a genuine reveal elsewhere
    in the stem."""
    both = question(
        "Given components A, B, and Correct, which one is Correct?",
        correct="Correct",
    )
    assert "answer_revealed" in {issue.code for issue in generic_quality_issues(both, intent_id="I")}


def test_old_format_review_json_without_require_full_human_review_still_validates():
    """Recommendation gained a fourth value for automated review; a review file
    written before that change must still load unchanged."""
    old_format = {
        "batch_id": "old-batch",
        "source_hashes": {},
        "items": [
            {
                "original_question_id": "q-1",
                "skill_id": "AI-SRC-08",
                "intent_id": "AI-SRC-08-INT-01",
                "recommendation": "propose_revision",
                "recommendation_reason": "Pending human review of generated candidate.",
            }
        ],
    }
    review = GroundedReview.model_validate(old_format)
    assert review.items[0].recommendation == "propose_revision"


def test_require_full_human_review_is_an_accepted_recommendation():
    item = CurationItem(
        original_question_id="q-1",
        skill_id="AI-SRC-08",
        intent_id="AI-SRC-08-INT-01",
        recommendation="require_full_human_review",
        recommendation_reason="Automated reviewer passes disagreed on grounding.",
    )
    assert item.recommendation == "require_full_human_review"
    assert item.final_review_status == "pending"
