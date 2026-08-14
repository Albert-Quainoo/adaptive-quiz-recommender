"""Curated revision proposals for grounded-pilot-20260811-v3."""

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from api.bank import BankItem
from api.schemas import QuizQuestion
from authoring.grounded_batch import PendingQuestion
from authoring.grounded_review import (
    CurationItem,
    GroundedReview,
    GroundedReviewStore,
    RevisionProvenance,
    approve_as_written,
    approve_revision,
    assert_immutable_source,
    propose_revision,
    source_hashes,
)
from authoring.question_intents import intents_by_skill, load_blueprint_for_batch
from authoring.review.config import ReviewPolicyConfig
from authoring.review.models import AutomatedReviewReport
from authoring.review.reports import AutomatedReviewReportStore
from authoring.review.reviewer import ContentReviewer
from authoring.review.service import review_candidate
from taxonomy.loader import load_reference_provenance, load_skills


BATCH_ID = "grounded-pilot-20260811-v3"
PROPOSAL_EDITOR = "human-review-curation"
TAXONOMY_PATH = Path("taxonomy/data/ai")


def revised(
    stem: str,
    options: list[str],
    correct: str,
    explanation: str,
    concept: str,
    difficulty: str,
) -> QuizQuestion:
    return QuizQuestion(
        question=stem,
        options=options,
        correct_answer=correct,
        explanation=explanation,
        concept=concept,
        difficulty=difficulty,
    )


PROPOSED_QUESTIONS = {
    "AI-FND-01-INT-04": revised(
        "Car A follows a fixed prerecorded route. Car B is explicitly capable of reasoning from current sensor information to choose a safe maneuver around a newly detected obstacle. Which comparison identifies the supported intelligent capability?",
        [
            "Car B demonstrates reasoning and problem solving, while Car A performs a fixed automated procedure.",
            "Car A demonstrates reasoning because every prerecorded route is an intelligent decision.",
            "Both cars demonstrate learning because both can move along a route.",
            "Neither car demonstrates an intelligent capability because driving decisions cannot involve problem solving.",
        ],
        "Car B demonstrates reasoning and problem solving, while Car A performs a fixed automated procedure.",
        "The scenario explicitly gives Car B the capability to reason about new sensor information and choose a maneuver. Car A only executes a fixed route, so automation alone does not establish intelligent behavior.",
        "Intelligent capability versus fixed automation",
        "intermediate",
    ),
    "AI-FND-01-INT-05": revised(
        "Restaurant App A learns from a user's past preferences and adapts later recommendations. App B performs a fixed database lookup that returns restaurants within a chosen distance. What makes App A the clearer AI application in this scenario?",
        [
            "It learns from past preferences and adapts its recommendations.",
            "It stores restaurant records in a database.",
            "It filters records using a fixed location rule.",
            "It displays the lookup results on a screen.",
        ],
        "It learns from past preferences and adapts its recommendations.",
        "The stated learning and adaptation distinguish App A from App B's fixed location filter and database lookup. Storage, filtering, and display alone do not supply that intelligent capability.",
        "Learning application versus fixed lookup",
        "intermediate",
    ),
    "AI-AGT-01-INT-03": revised(
        "A warehouse delivery robot is evaluated by the percentage of packages delivered to the correct destination without a collision. In its PEAS description, what role does this percentage play?",
        [
            "Performance measure",
            "Environment",
            "Actuator",
            "Sensor",
        ],
        "Performance measure",
        "The percentage states how success is evaluated, so it is the performance measure rather than part of the environment, an actuator, or a sensor.",
        "PEAS performance measure",
        "introductory",
    ),
    "AI-AGT-01-INT-04": revised(
        "A car navigator receives its current location from a GPS receiver and changes the car's direction through a steering controller. Which sensor-and-actuator pairing is correct?",
        [
            "Sensor: GPS receiver; actuator: steering controller",
            "Sensor: steering controller; actuator: GPS receiver",
            "Sensor: destination goal; actuator: current location",
            "Sensor: road network; actuator: destination goal",
        ],
        "Sensor: GPS receiver; actuator: steering controller",
        "The GPS receiver supplies location information to the navigator, while the steering controller carries out an action that changes the car's direction.",
        "Navigator sensor-actuator pairing",
        "intermediate",
    ),
    "AI-AGT-01-INT-06": revised(
        "A car navigator has this PEAS description: performance measure—arrive safely at the destination; environment—roads and traffic; sensors—cameras; actuators—GPS receiver, steering, and brakes. Exactly one element is assigned to the wrong role. Which correction repairs the description?",
        [
            "Move the GPS receiver from actuators to sensors.",
            "Move steering from actuators to the performance measure.",
            "Move roads and traffic from the environment to sensors.",
            "Move safe arrival from the performance measure to actuators.",
        ],
        "Move the GPS receiver from actuators to sensors.",
        "A GPS receiver supplies location information, so it is a sensor. Steering and brakes remain actuators, and the performance measure and environment are already assigned correctly.",
        "PEAS role correction",
        "intermediate",
    ),
    "AI-SRC-01-INT-11": revised(
        "Before any move, a delivery robot is in Room A and its goal is to reach Room D. Which element is the initial state of this search problem?",
        [
            "The robot being in Room A before any move",
            "The action of moving from Room A to Room B",
            "The robot being in Room B after one move",
            "The condition that the robot is in Room D",
        ],
        "The robot being in Room A before any move",
        "The initial state is the concrete configuration before any action occurs: the robot is in Room A. Reaching Room D is the goal condition.",
        "Initial state",
        "introductory",
    ),
    "AI-SRC-02-INT-01": revised(
        "Which statement correctly distinguishes a state-space graph from a search tree?",
        [
            "A state-space graph represents underlying states, while each search-tree node represents a particular path to a state.",
            "A state-space graph represents action sequences, while each search-tree node represents only an underlying state.",
            "Both structures represent only paths and omit the underlying states.",
            "Both structures merge every path that reaches the same state into one search-tree node.",
        ],
        "A state-space graph represents underlying states, while each search-tree node represents a particular path to a state.",
        "The state-space graph captures the underlying states and transitions. A search-tree node is path-specific, so different paths to one state can produce different tree nodes.",
        "State-space and search-tree representation",
        "introductory",
    ),
    "AI-SRC-02-INT-02": revised(
        "The paths S → A → C and S → B → C end at the same underlying state C. Why can the search tree contain a distinct node for each path?",
        [
            "Each search-tree node represents its particular path to the resulting state.",
            "The underlying state C becomes two different states when reached by different actions.",
            "A repeated state can appear only when the state-space graph contains a cycle.",
            "Search trees merge paths only after the goal has been found.",
        ],
        "Each search-tree node represents its particular path to the resulting state.",
        "The two nodes remain distinct because they encode different paths, even though both paths end at the same underlying state C.",
        "Repeated states in search-tree nodes",
        "introductory",
    ),
    "AI-SRC-02-INT-03": revised(
        "A search-tree node is reached from the root by the displayed sequence Move east → Move north. What does this root-to-node sequence encode?",
        [
            "An ordered plan from the start state to the state represented by the node",
            "Only the final state, with no information about how it was reached",
            "The complete state-space graph for the problem",
            "The order in which every frontier node must be expanded",
        ],
        "An ordered plan from the start state to the state represented by the node",
        "A root-to-node sequence records the ordered actions used to reach that node, so it encodes a plan rather than only the terminal state or an expansion order.",
        "Root-to-node plan",
        "introductory",
    ),
    "AI-SRC-03-INT-02": revised(
        "A selected node N represents state A, where the available actions lead to states B and C. What does EXPAND(N) produce?",
        [
            "Child nodes for B and C, each recording its resulting state, parent N, and generating action",
            "A Boolean verdict stating whether A is a goal state",
            "A reordered frontier containing every node in the search",
            "A completed solution path from A to the goal",
        ],
        "Child nodes for B and C, each recording its resulting state, parent N, and generating action",
        "EXPAND applies the available actions at N and constructs child nodes. Each child records the resulting state together with parent N and the action that generated it.",
        "Node expansion",
        "introductory",
    ),
    "AI-SRC-03-INT-03": revised(
        "Under the project convention, expanding a parent generates a child whose state is X. The child is admitted to the frontier and X is recorded as reached, but the child has not yet been removed for expansion. Which description is correct now?",
        [
            "The child is in the frontier, X is reached, and X is not yet explored.",
            "The child is not in the frontier, X is reached, and X is explored.",
            "The child is in the frontier, but X is neither reached nor explored.",
            "The child has already been expanded and X is explored.",
        ],
        "The child is in the frontier, X is reached, and X is not yet explored.",
        "Frontier admission records X as reached. Under the stated convention, X becomes explored only after its node is removed from the frontier for expansion.",
        "Reached and explored lifecycle",
        "introductory",
    ),
    "AI-SRC-03-INT-04": revised(
        "A generated child has state X with path cost 6. The reached structure already has a frontier record for X with path cost 4, so the existing record is at least as good. What should the search do?",
        [
            "Keep the existing record and do not insert the duplicate child.",
            "Insert the new child as a second frontier record for X.",
            "Replace the cost-4 record with the new cost-6 record.",
            "Mark X as expanded immediately without selecting its frontier record.",
        ],
        "Keep the existing record and do not insert the duplicate child.",
        "Because X has already been reached through a path that is at least as good, the new child supplies no improvement and should not create a duplicate frontier record.",
        "Duplicate frontier prevention",
        "intermediate",
    ),
    "AI-SRC-03-INT-06": revised(
        "The project convention records a state as reached when its node enters the frontier and as explored only after that node is removed for expansion. A process report says: 'Child Y was generated, admitted to the frontier, and recorded as reached. While Y still waits in the frontier, it is also marked explored.' Which correction repairs the report's single timing error?",
        [
            "Do not mark Y explored until its node is removed from the frontier for expansion.",
            "Delay recording Y as reached until after its node has been expanded.",
            "Add Y to reached a second time when its node is expanded.",
            "Remove Y from the frontier as soon as it is recorded as reached.",
        ],
        "Do not mark Y explored until its node is removed from the frontier for expansion.",
        "The reached timing is already correct because Y entered the frontier. Only the explored label is premature; it applies after removal for expansion, without adding Y to reached again.",
        "Reached-versus-explored timing",
        "intermediate",
    ),
    "AI-SRC-08-INT-02": revised(
        "State the heuristic value h(n) from the following description: We have already travelled 3 units and our estimated remaining distance to the goal is 5 units. What is the heuristic value h(n)?",
        [
            "The accumulated cost from the start, which is 3 units.",
            "The estimated remaining cost to the goal, which is 5 units.",
            "The total cost from start to goal, which is 8 units.",
            "The path cost, which is the sum of the accumulated cost and the estimated remaining cost.",
        ],
        "The estimated remaining cost to the goal, which is 5 units.",
        "The heuristic h(n) is the estimated remaining cost to the goal, so h(n) = 5. The 3 units already travelled are the distinct accumulated cost g(n).",
        "Heuristic value h(n)",
        "introductory",
    ),
}


REVIEW_NOTES = {
    "AI-FND-01-INT-04": "Replaced unsupported learning and machine-learning claims with an explicit reasoning-and-problem-solving capability contrasted against fixed route automation.",
    "AI-FND-01-INT-05": "Made learning from past preferences explicit and contrasted it with a fixed location filter and database lookup.",
    "AI-AGT-01-INT-03": "Replaced duplicate sensor recall with a single-role PEAS scenario that identifies the performance measure.",
    "AI-AGT-01-INT-04": "Reframed the item around one unambiguous GPS-sensor and steering-actuator pairing instead of PEAS-letter permutations.",
    "AI-AGT-01-INT-06": "Introduced exactly one PEAS misclassification and asks the learner to move the GPS receiver from actuators to sensors.",
    "AI-SRC-01-INT-11": "Added a concrete Room A starting configuration and removed the generic initial-state definition that competed with the specific answer.",
    "AI-SRC-02-INT-01": "Corrected the comparison to distinguish underlying state representation from path-specific search-tree nodes without claiming the graph stores only the current state.",
    "AI-SRC-02-INT-02": "Used two explicit paths to the same state and retained only one path-specific explanation as the correct option.",
    "AI-SRC-02-INT-03": "Asked what one displayed root-to-node action sequence encodes and removed overlapping plan-versus-action-sequence options.",
    "AI-SRC-03-INT-02": "Defined EXPAND as constructing child nodes with resulting state, parent, and generating action, with non-overlapping distractors.",
    "AI-SRC-03-INT-03": "Added a child admitted to frontier and reached but not expanded, then tests its exact lifecycle and structure membership.",
    "AI-SRC-03-INT-04": "Restored the duplicate-discovery intent using an existing at-least-as-good frontier record and no duplicate insertion.",
    "AI-SRC-03-INT-06": "Created one explored-timing error under the stated project convention and asks for its sole correction without re-adding reached state.",
    "AI-SRC-08-INT-02": "Corrected directional wording to estimated remaining cost to the goal while preserving h(n) = 5 and distinguishing g(n) = 3.",
}


UNCHANGED_INTENTS = {
    "AI-FND-01-INT-06",
    "AI-AGT-01-INT-01",
    "AI-AGT-01-INT-02",
    "AI-AGT-01-INT-05",
    "AI-SRC-01-INT-12",
    "AI-SRC-01-INT-13",
    "AI-SRC-03-INT-01",
    "AI-SRC-03-INT-05",
    "AI-SRC-08-INT-01",
    "AI-SRC-08-INT-03",
}


def load_sources(batch_path: Path) -> list[PendingQuestion]:
    return [
        PendingQuestion.model_validate_json(line)
        for line in (batch_path / "pending_questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]


def build_review(batch_path: Path, *, edited_at: datetime) -> GroundedReview:
    sources = load_sources(batch_path)
    source_intents = {source.intent_id for source in sources}
    expected_intents = set(PROPOSED_QUESTIONS) | UNCHANGED_INTENTS
    if len(sources) != 24 or source_intents != expected_intents:
        raise ValueError("source batch does not match the reviewed 24-intent inventory")

    items = []
    for source in sources:
        if source.intent_id in UNCHANGED_INTENTS:
            items.append(
                CurationItem(
                    original_question_id=source.question_id,
                    skill_id=source.skill_id,
                    intent_id=source.intent_id,
                    recommendation="approve_as_written",
                    recommendation_reason="Human review found no content change necessary; retain the immutable generated candidate for final approval.",
                )
            )
            continue

        edited = PROPOSED_QUESTIONS[source.intent_id]
        QuizQuestion.model_validate(edited.model_dump())
        item = propose_revision(
            CurationItem(
                original_question_id=source.question_id,
                skill_id=source.skill_id,
                intent_id=source.intent_id,
                recommendation="propose_revision",
                recommendation_reason=REVIEW_NOTES[source.intent_id],
            ),
            source.question,
            edited,
            PROPOSAL_EDITOR,
            REVIEW_NOTES[source.intent_id],
            edited_at=edited_at,
            provenance=RevisionProvenance.from_source(source),
        )
        revision = item.revisions[0]
        BankItem(
            item_id=revision.revision_id,
            skill_id=source.skill_id,
            provenance="generated",
            question=revision.question,
        )
        items.append(item)

    review = GroundedReview(
        batch_id=BATCH_ID,
        source_hashes=source_hashes(batch_path),
        items=items,
    )
    assert_immutable_source(batch_path, review)
    return review


def review_bundle(review: GroundedReview, sources: list[PendingQuestion]) -> str:
    source_by_id = {source.question_id: source for source in sources}
    unchanged = [item for item in review.items if item.recommendation == "approve_as_written"]
    revised_items = [item for item in review.items if item.recommendation == "propose_revision"]
    counts = Counter((source.skill_id, source.question.difficulty) for source in sources)
    lines = [
        f"# Human review bundle: {review.batch_id}",
        "",
        "All candidates and revisions remain pending final human approval. The immutable generated source records are identified by the hashes in `review.json`.",
        "",
        "## Inventory",
        "",
        "- Final candidate count: 24",
        "- Unchanged approval candidates: 10",
        "- Proposed revisions: 14",
        f"- Difficulty distribution: {sum(v for (s, d), v in counts.items() if d == 'introductory')} introductory, {sum(v for (s, d), v in counts.items() if d == 'intermediate')} intermediate",
        "",
        "| Skill | Introductory | Intermediate | Total |",
        "|---|---:|---:|---:|",
    ]
    for skill in ("AI-FND-01", "AI-AGT-01", "AI-SRC-01", "AI-SRC-02", "AI-SRC-03", "AI-SRC-08"):
        intro = counts[(skill, "introductory")]
        intermediate = counts[(skill, "intermediate")]
        lines.append(f"| {skill} | {intro} | {intermediate} | {intro + intermediate} |")

    lines.extend(["", "## Unchanged approval candidates", ""])
    for item in unchanged:
        source = source_by_id[item.original_question_id]
        lines.extend(
            [
                f"### {item.intent_id}",
                "",
                f"- Original question ID: `{item.original_question_id}`",
                f"- Difficulty: {source.question.difficulty}",
                f"- References: {', '.join(f'`{value}`' for value in source.reference_ids)}",
                f"- Question: {source.question.question}",
                f"- Correct answer: {source.question.correct_answer}",
                "- Proposed final status: pending",
                "",
            ]
        )

    lines.extend(["## Proposed revisions", ""])
    for item in revised_items:
        source = source_by_id[item.original_question_id]
        revision = item.revisions[0]
        lines.extend(
            [
                f"### {item.intent_id}",
                "",
                f"- Original question ID: `{item.original_question_id}`",
                f"- Revision ID: `{revision.revision_id}`",
                f"- Content hash: `{revision.content_hash}`",
                f"- References: {', '.join(f'`{value}`' for value in revision.reference_ids)}",
                f"- Changed fields: {', '.join(f'`{value}`' for value in revision.changed_fields)}",
                f"- Review note: {revision.review_note}",
                "- Final review status: pending",
                "",
                "**Original**",
                "",
                f"- Question: {source.question.question}",
                f"- Options: {' | '.join(source.question.options)}",
                f"- Correct answer: {source.question.correct_answer}",
                f"- Explanation: {source.question.explanation}",
                f"- Concept: {source.question.concept}",
                "",
                "**Revised**",
                "",
                f"- Question: {revision.question.question}",
                f"- Options: {' | '.join(revision.question.options)}",
                f"- Correct answer: {revision.question.correct_answer}",
                f"- Explanation: {revision.question.explanation}",
                f"- Concept: {revision.question.concept}",
                f"- Difficulty: {revision.question.difficulty}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def write_artifacts(batch_path: Path, output_path: Path) -> GroundedReview:
    timestamp = datetime.now(timezone.utc)
    review = build_review(batch_path, edited_at=timestamp)
    sources = load_sources(batch_path)
    output_path.mkdir(parents=True, exist_ok=False)
    GroundedReviewStore(output_path / "review.json").save(review)
    proposals = output_path / "proposals"
    proposals.mkdir()
    for item in review.items:
        if item.revisions:
            (proposals / f"{item.intent_id}.json").write_text(
                json.dumps(item.revisions[0].model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
    (output_path / "review_bundle.md").write_text(
        review_bundle(review, sources), encoding="utf-8"
    )
    summary = {
        "batch_id": review.batch_id,
        "source_immutable": True,
        "source_question_count": len(sources),
        "unchanged_approval_candidates": sum(
            item.recommendation == "approve_as_written" for item in review.items
        ),
        "proposed_revisions": sum(
            item.recommendation == "propose_revision" for item in review.items
        ),
        "pending_items": sum(item.final_review_status == "pending" for item in review.items),
        "quiz_question_validations": len(PROPOSED_QUESTIONS),
        "bank_item_validations": len(PROPOSED_QUESTIONS),
    }
    (output_path / "validation_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return review


def run_automated_review(
    batch_path: Path,
    review: GroundedReview,
    *,
    reviewer: ContentReviewer,
    reports_path: Path,
    config: ReviewPolicyConfig,
) -> dict[str, AutomatedReviewReport]:
    """Score every item's current head content -- its revision if one was proposed,
    else the immutable source -- with the automated reviewer.

    Call this before approve_all. Without it, a human-proposed revision can reach
    "approved" without the automated reviewer (grounding checks, duplicate-distractor
    detection, etc.) ever having scored the text actually being approved -- only the
    discarded pre-revision draft, if that.

    Deliberately does not filter revisions.py to final_review_status == "pending" the
    way authoring/replenishment/worker.py's _reviewed_head does: that filter is right
    for the worker's still-pending curation queue, but this function is also meant to
    run *after* approve_all, once the revision's own final_review_status is "approved"
    -- filtering on "pending" there would silently fall back to the stale
    pre-revision source. Every item here has at most one revision (build_review
    creates exactly one propose_revision item per PROPOSED_QUESTIONS entry), so the
    latest revision, regardless of status, is unambiguously the head.
    """
    sources = {source.question_id: source for source in load_sources(batch_path)}
    blueprint = load_blueprint_for_batch(BATCH_ID)
    intents = {
        intent.intent_id: intent
        for skill_intents in intents_by_skill(blueprint).values()
        for intent in skill_intents
    }
    catalogue = load_skills(TAXONOMY_PATH / "skills.csv", TAXONOMY_PATH / "references.csv")
    skills = {skill.skill_id: skill for skill in catalogue.skills}
    provenance = load_reference_provenance(
        TAXONOMY_PATH / "reference_provenance.csv", known_skill_ids=set(skills)
    )
    references_by_skill = defaultdict(list)
    for record in provenance:
        references_by_skill[record.skill_id].append(record)

    report_store = AutomatedReviewReportStore(reports_path)
    reports: dict[str, AutomatedReviewReport] = {}
    for item in review.items:
        source = sources[item.original_question_id]
        candidate = (
            source
            if not item.revisions
            else source.model_copy(update={"question": item.revisions[-1].question})
        )
        reports[item.intent_id] = review_candidate(
            candidate,
            skills[item.skill_id],
            intents[item.intent_id],
            references_by_skill[item.skill_id],
            reviewer=reviewer,
            config=config,
            report_store=report_store,
        )
    return reports


def approve_all(
    review: GroundedReview,
    *,
    reviewer: str,
    reviewed_at: datetime,
) -> GroundedReview:
    items = []
    for item in review.items:
        if item.recommendation == "approve_as_written":
            items.append(
                approve_as_written(item, reviewer, reviewed_at=reviewed_at)
            )
        else:
            if len(item.revisions) != 1:
                raise ValueError(f"{item.intent_id} needs exactly one proposed revision")
            items.append(
                approve_revision(
                    item,
                    item.revisions[0].revision_id,
                    reviewer,
                    reviewed_at=reviewed_at,
                )
            )
    return GroundedReview.model_validate(
        review.model_dump() | {"items": [item.model_dump() for item in items]}
    )


def approved_bank_items(
    review: GroundedReview, sources: list[PendingQuestion]
) -> list[BankItem]:
    source_by_id = {source.question_id: source for source in sources}
    exported = []
    for item in review.items:
        if item.final_review_status != "approved":
            raise ValueError(f"{item.intent_id} is not approved")
        source = source_by_id[item.original_question_id]
        if item.recommendation == "approve_as_written":
            item_id = source.question_id
            question = source.question
        else:
            approved = [
                revision
                for revision in item.revisions
                if revision.final_review_status == "approved"
            ]
            if len(approved) != 1:
                raise ValueError(f"{item.intent_id} has no unique approved revision")
            item_id = approved[0].revision_id
            question = approved[0].question
        exported.append(
            BankItem(
                item_id=item_id,
                skill_id=item.skill_id,
                provenance="generated",
                question=question,
            )
        )
    return exported


def write_bank(path: Path, items: list[BankItem]) -> None:
    if path.exists():
        raise ValueError(f"approved bank already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
            for item in items
        ),
        encoding="utf-8",
    )
