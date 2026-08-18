"""Drives one skill's replenishment job through its pipeline stages.

Every stage below calls an existing, unmodified function from the retrieval,
generation, curation, or bank modules -- this file only decides which stage
runs next and how to record the outcome on the job row. It never approves
anything and never authors pedagogy: generation is hard-gated on a reviewed
intent blueprint that must already exist under authoring/blueprints/.
"""

import hashlib
import json
import os
import subprocess
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from api.bank import BankItem
from api.schemas import QuizQuestion
from app.bootstrap import load_approved_bank
from authoring.grounded_batch import (
    BatchConfig,
    BatchGenerationError,
    BatchModel,
    GenerationOutcome,
    PendingQuestion,
    generate_batch,
)
from authoring.grounded_review import (
    CurationItem,
    GroundedReview,
    GroundedReviewStore,
    RevisionProvenance,
    build_pending_review,
    export_approved_bank_items,
    get_item,
    list_pending,
    load_source_questions,
    question_content_hash,
)
from authoring import question_intents
from authoring.deterministic_templates import DETERMINISTIC_TEMPLATES
from authoring.question_intents import (
    PilotBlueprint,
    QuestionIntent,
    intents_by_skill,
    load_blueprint_for_batch,
)
from authoring.replenishment.demand import (
    compute_blueprint_slot_demand,
    compute_demand_fingerprint,
    deficient_slots,
    load_approved_item_ids,
)
from authoring.replenishment.jobs import ReplenishmentJob, SQLiteReplenishmentJobRepository
from authoring.replenishment.manifest import (
    CourseManifest,
    active_bank_path,
    active_bank_pointer_path,
)
from authoring.retrieval.brave import MissingCredentials
from authoring.retrieval.diagnostics import RetrievalDiagnostics
from authoring.retrieval.models import ReferenceCandidate
from authoring.retrieval.search import (
    PageFetcher,
    RetrievalError,
    SearchProvider,
    known_for,
    retrieve_candidates,
)
from authoring.retrieval.store import CandidateStore
from authoring.review.config import ReviewPolicyConfig
from authoring.review.models import AutomatedReviewReport
from authoring.review.reports import AutomatedReviewReportStore, review_report_path
from authoring.review.reviewer import ContentReviewer, ModelBackedContentReviewer, ReviewerUnavailableError
from authoring.review.revision import (
    generate_revision_candidate,
    is_localized_issue,
    propose_automated_revision,
)
from authoring.review.service import review_candidate
from scripts.import_reference_candidates import import_candidates
from scripts.merge_approved_banks import merge_approved_banks
from scripts.validate_approved_bank import validate_bank
from taxonomy.loader import load_reference_provenance, load_skills


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class ModelUnavailableError(RuntimeError):
    """Raised when the Llama model cannot be loaded or reached."""


class LlamaBatchModel:
    """Adapter from api.model_loader's cached model/tokenizer to BatchModel.

    Torch/transformers are imported inside generate(), not at module import
    time, so anything that only needs job status or inventory (the admin
    view, `cli.py status`) never has to have GPU dependencies installed.
    """

    def __init__(self, *, model_revision: str | None = None) -> None:
        # An explicit pin (constructor arg or MODEL_REVISION env var) is deliberate
        # operator intent and is never silently overridden. Absent a pin, _generate()
        # opportunistically records the exact commit hash transformers resolved for
        # the loaded model (a real, pinned identifier "unknown" then reports later, if
        # available) the first time the model actually loads -- see _generate() below.
        self._pinned_model_revision = model_revision or os.getenv("MODEL_REVISION") or None
        self._model_revision = self._pinned_model_revision or "unknown"

    @property
    def model_id(self) -> str:
        from api.model_loader import get_model_id

        try:
            return get_model_id()
        except RuntimeError as exc:
            raise ModelUnavailableError(str(exc)) from exc

    @property
    def model_revision(self) -> str:
        return self._model_revision

    def generate(
        self, messages: list[dict[str, str]], seed: int, generation_parameters: dict
    ) -> str:
        return self._generate(messages, seed, generation_parameters).text

    def generate_with_metadata(
        self, messages: list[dict[str, str]], seed: int, generation_parameters: dict
    ) -> GenerationOutcome:
        return self._generate(messages, seed, generation_parameters)

    def _generate(
        self, messages: list[dict[str, str]], seed: int, generation_parameters: dict
    ) -> GenerationOutcome:
        try:
            import torch
            from api.model_loader import load_model, load_tokenizer

            tokenizer = load_tokenizer()
            model = load_model()
        except Exception as exc:
            raise ModelUnavailableError(str(exc)) from exc

        if self._pinned_model_revision is None:
            resolved_revision = getattr(model.config, "_commit_hash", None)
            if resolved_revision:
                self._model_revision = resolved_revision

        torch.manual_seed(seed)
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        max_new_tokens = generation_parameters.get("max_new_tokens", 800)
        with torch.inference_mode():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        input_tokens = inputs["input_ids"].shape[1]
        generated_tokens = output[0, input_tokens:]
        output_tokens = generated_tokens.shape[0]
        # See modal_app.py's identical logic for why the token id, not just the
        # count, decides "stop" vs "length".
        if output_tokens > 0 and generated_tokens[-1].item() == tokenizer.eos_token_id:
            finish_reason = "stop"
        elif output_tokens >= max_new_tokens:
            finish_reason = "length"
        else:
            finish_reason = "unknown"
        text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return GenerationOutcome(
            text=text,
            finish_reason=finish_reason,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            max_new_tokens=max_new_tokens,
        )


class AmbiguousBlueprintError(RuntimeError):
    """Raised when a job has no explicit batch_id and more than one reviewed
    blueprint covers its skill -- a real ambiguity that must be resolved by
    enqueuing with an explicit batch_id, never guessed."""


def blueprints_covering_skill(skill_id: str) -> list[PilotBlueprint]:
    """Every reviewed blueprint file that declares at least one intent for this
    skill, in deterministic batch_id order -- never file modification time, which
    is unstable across git checkouts/edits and was the root cause this replaced."""
    blueprint_directory = question_intents.BLUEPRINT_DIRECTORY
    if not blueprint_directory.is_dir():
        return []
    covering: list[PilotBlueprint] = []
    for path in sorted(blueprint_directory.glob("*.json")):
        try:
            blueprint = PilotBlueprint.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if skill_id in intents_by_skill(blueprint):
            covering.append(blueprint)
    return sorted(covering, key=lambda blueprint: blueprint.batch_id)


def resolve_job_blueprint(job: ReplenishmentJob) -> tuple[PilotBlueprint | None, str | None]:
    """Resolve exactly one blueprint for this job's skill, without ever depending on
    blueprint file modification time.

    A job's batch_id, once resolved, is meant to be persisted into job.metadata by
    the caller so every later stage reuses that pinned value instead of
    re-resolving -- otherwise a blueprint file touched between two ticks of the same
    job (a git checkout, an editor save) could silently make retrieve_references and
    generate_questions target two different blueprints for one job. If the job
    already carries a batch_id, that batch is loaded directly (raises ValueError if
    it names no real blueprint, or a blueprint that does not cover this skill).
    Returns (None, None) only when the job has no batch_id and no blueprint covers
    the skill at all -- a skill with no reviewed blueprint yet, which retrieval
    still handles speculatively (never invents intents; generation stays blocked on
    human authoring). Raises AmbiguousBlueprintError when the job has no batch_id
    and more than one blueprint covers the skill.
    """
    batch_id = job.metadata.get("batch_id")
    if batch_id:
        blueprint = load_blueprint_for_batch(batch_id)
        if job.skill_id not in intents_by_skill(blueprint):
            raise ValueError(f"blueprint {batch_id} does not cover skill {job.skill_id}")
        return blueprint, batch_id

    covering = blueprints_covering_skill(job.skill_id)
    if not covering:
        return None, None
    if len(covering) > 1:
        raise AmbiguousBlueprintError(
            f"{len(covering)} blueprints cover {job.skill_id} "
            f"({', '.join(blueprint.batch_id for blueprint in covering)}); "
            "enqueue with an explicit batch_id"
        )
    blueprint = covering[0]
    return blueprint, blueprint.batch_id


def current_commit_allow_dirty() -> str:
    """A commit id for provenance, without the clean-worktree requirement
    that current_git_commit() imposes on the manual pilot-batch workflow --
    a background worker runs against whatever the working tree looks like."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _next_bank_version(manifest: CourseManifest) -> int:
    pointer_path = active_bank_pointer_path(manifest)
    if not pointer_path.is_file():
        return 1
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    return int(pointer["version"]) + 1


def _write_bank_jsonl(path: Path, items: list[BankItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(item.model_dump(mode="json"), sort_keys=True) + "\n"
            for item in items
        ),
        encoding="utf-8",
    )


def _write_active_pointer(
    manifest: CourseManifest, bank_path: Path, version: int, *, clock: Callable[[], datetime]
) -> None:
    pointer_path = active_bank_pointer_path(manifest)
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = pointer_path.with_suffix(pointer_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            {
                "course_id": manifest.course_id,
                "path": str(bank_path),
                "version": version,
                "updated_at": clock().isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(pointer_path)


def _batch_output_dir(manifest: CourseManifest, batch_id: str, skill_id: str) -> Path:
    # Scoped per (batch_id, skill_id): the worker always generates one skill
    # at a time, even when a reviewed blueprint's intents span several
    # skills, so two skills sharing a blueprint never collide on one
    # immutable batch manifest. Nested under the manifest's own
    # review_store_path (not a hardcoded repo-relative path) so a test or
    # demo manifest pointed at a temp directory never writes into the real
    # outputs/ tree.
    return manifest.review_store_path.parent / "batches" / f"{batch_id}__{skill_id}"


def _handle_retrieve_references(
    job: ReplenishmentJob,
    manifest: CourseManifest,
    *,
    job_repository: SQLiteReplenishmentJobRepository,
    search_provider: SearchProvider,
    fetcher: PageFetcher,
    clock: Callable[[], datetime],
    max_attempts: int,
) -> None:
    catalogue = load_skills(manifest.skills_path(), manifest.references_path())
    skill = next((s for s in catalogue.skills if s.skill_id == job.skill_id), None)
    if skill is None:
        job_repository.mark_permanent_failure(
            job.job_id,
            error_code="unknown_skill",
            error_message=f"{job.skill_id} is not in the {manifest.course_id} taxonomy",
        )
        return

    # Resolved once, explicitly, before the first network call this job could make
    # (Brave Search) -- never re-scanned by blueprint file modification time. See
    # resolve_job_blueprint's docstring: a job with no batch_id yet either resolves
    # unambiguously (exactly one blueprint covers the skill) or is left unresolved
    # (no blueprint at all -- retrieval still proceeds speculatively for it, same as
    # before this check existed). Two or more blueprints covering the same skill
    # with no explicit batch_id on the job is a hard failure, not a guess.
    try:
        blueprint_for_demand, resolved_batch_id = resolve_job_blueprint(job)
    except AmbiguousBlueprintError as exc:
        job_repository.mark_permanent_failure(
            job.job_id, error_code="ambiguous_blueprint_selection", error_message=str(exc)
        )
        return
    except ValueError as exc:
        job_repository.mark_permanent_failure(
            job.job_id, error_code="invalid_batch_id", error_message=str(exc)
        )
        return

    # If this is the first tick to resolve a batch_id, pin it into job.metadata so
    # every later stage of this same job reuses the identical blueprint, rather than
    # re-resolving (and risking drift if a blueprint file changes in between ticks).
    batch_metadata = {"batch_id": resolved_batch_id} if resolved_batch_id else {}

    if blueprint_for_demand is not None:
        approved_item_ids = load_approved_item_ids(manifest)
        demand = compute_blueprint_slot_demand(
            blueprint_for_demand, job.skill_id, approved_item_ids
        )
        if demand and not deficient_slots(demand):
            job_repository.mark_no_longer_needed(
                job.job_id,
                error_code="demand_already_satisfied",
                error_message=(
                    f"every intent slot in {blueprint_for_demand.batch_id} for "
                    f"{job.skill_id} is already an approved bank item; nothing to "
                    "retrieve or generate"
                ),
                metadata={
                    "demand_fingerprint": _demand_fingerprint(
                        blueprint_for_demand, job.skill_id, approved_item_ids, manifest
                    ),
                },
            )
            return

    store = CandidateStore(manifest.candidate_store_path)
    held = store.load()
    approved_now = [
        candidate
        for candidate in held
        if candidate.skill_id == job.skill_id and candidate.review_status == "approved"
    ]

    if not approved_now:
        try:
            new_candidates: list[ReferenceCandidate] = retrieve_candidates(
                skill,
                search_provider,
                fetcher,
                manifest.allowed_domains,
                limit=job.requested_count,
                diagnostics=RetrievalDiagnostics(),
                known=known_for(job.skill_id, held),
                clock=clock,
            )
        except MissingCredentials as exc:
            job_repository.mark_retryable_failure(
                job.job_id,
                error_code="missing_brave_credentials",
                error_message=str(exc),
                backoff_seconds=300,
                max_attempts=max_attempts,
                clock=clock,
            )
            return
        except RetrievalError as exc:
            job_repository.mark_retryable_failure(
                job.job_id,
                error_code="retrieval_error",
                error_message=str(exc),
                max_attempts=max_attempts,
                clock=clock,
            )
            return
        store.add(new_candidates)
        held = store.load()
        approved_now = [
            candidate
            for candidate in held
            if candidate.skill_id == job.skill_id and candidate.review_status == "approved"
        ]

    if approved_now:
        job_repository.mark_queued(
            job.job_id, job_type="generate_questions", metadata=batch_metadata
        )
        return

    pending_count = sum(
        1
        for candidate in held
        if candidate.skill_id == job.skill_id and candidate.review_status == "pending"
    )
    job_repository.mark_waiting(
        job.job_id,
        "waiting_for_reference_review",
        job_type="retrieve_references",
        metadata={**batch_metadata, "pending_candidates": pending_count},
    )


def _blueprint_generation_difficulty(skill_id: str, intents: list[QuestionIntent]) -> str:
    """Derive BatchConfig.difficulty from the blueprint's own intents for this skill,
    rather than falling back to BatchConfig's "intermediate" default. The blueprint is
    authoritative: an automated worker has no human present to pick a difficulty the
    blueprint itself leaves ambiguous. Every intent for this skill must both belong to
    skill_id and declare an explicit difficulty -- a missing one is a deterministic
    configuration defect the caller must fail on before generate_batch ever calls the
    model, not something to guess through.

    When every intent shares one explicit difficulty, that value is returned. When
    they declare more than one, "mixed" is returned instead: generate_batch's own
    BatchConfig(difficulty="mixed") support already resolves each question's
    difficulty from its own intent.difficulty in that case (see
    authoring/grounded_batch.py), so a blueprint intentionally spanning tiers for one
    skill -- e.g. two introductory + two intermediate intents -- generates correctly
    rather than being rejected outright just because its tiers aren't uniform."""
    if not intents:
        raise BatchGenerationError(f"{skill_id} has no reviewed question intents")
    declared: set[str | None] = set()
    for intent in intents:
        if intent.skill_id != skill_id:
            raise BatchGenerationError(
                f"{intent.intent_id} belongs to skill {intent.skill_id}, not {skill_id}"
            )
        declared.add(intent.difficulty)
    if None in declared:
        offending = ", ".join(
            intent.intent_id for intent in intents if intent.difficulty is None
        )
        raise BatchGenerationError(
            f"{skill_id} blueprint intent(s) have no explicit difficulty "
            f"(automated replenishment cannot guess): {offending}"
        )
    return declared.pop() if len(declared) == 1 else "mixed"


def _demand_fingerprint(
    blueprint: PilotBlueprint,
    skill_id: str,
    approved_item_ids: set[str],
    manifest: CourseManifest,
) -> str:
    """Best-effort difficulty resolution -- "unknown" rather than raising --
    since this only feeds an audit fingerprint recorded alongside a
    no_longer_needed outcome, never a gate on whether generation proceeds."""
    intents = intents_by_skill(blueprint).get(skill_id, [])
    try:
        difficulty = _blueprint_generation_difficulty(skill_id, intents)
    except BatchGenerationError:
        difficulty = "unknown"
    return compute_demand_fingerprint(
        blueprint, skill_id, approved_item_ids,
        difficulty=difficulty, target_supply=manifest.target_supply,
    )


def _handle_generate_questions(
    job: ReplenishmentJob,
    manifest: CourseManifest,
    *,
    job_repository: SQLiteReplenishmentJobRepository,
    model_factory: Callable[[], BatchModel],
    clock: Callable[[], datetime],
    max_attempts: int,
) -> None:
    catalogue = load_skills(manifest.skills_path(), manifest.references_path())
    skill = next((s for s in catalogue.skills if s.skill_id == job.skill_id), None)
    if skill is None:
        job_repository.mark_permanent_failure(
            job.job_id,
            error_code="unknown_skill",
            error_message=f"{job.skill_id} is not in the {manifest.course_id} taxonomy",
        )
        return

    import_candidates(
        manifest.candidate_store_path,
        manifest.skills_path(),
        manifest.references_path(),
        manifest.provenance_path(),
    )

    # Resolved via the job's own explicit batch_id whenever one is already carried
    # (pinned by _handle_retrieve_references, or set at enqueue time) -- never
    # re-scanned by blueprint file modification time. A job that reaches generation
    # with no batch_id and an ambiguous or missing blueprint pool fails closed here.
    try:
        blueprint, _ = resolve_job_blueprint(job)
    except AmbiguousBlueprintError as exc:
        job_repository.mark_permanent_failure(
            job.job_id, error_code="ambiguous_blueprint_selection", error_message=str(exc)
        )
        return
    except ValueError as exc:
        job_repository.mark_permanent_failure(
            job.job_id, error_code="invalid_batch_id", error_message=str(exc)
        )
        return
    if blueprint is None:
        job_repository.mark_permanent_failure(
            job.job_id,
            error_code="missing_intent_blueprint",
            error_message=(
                f"no reviewed question-intent blueprint covers {job.skill_id}; "
                "author one under authoring/blueprints/ "
                "(see authoring/question_intents.py) before retrying"
            ),
        )
        return

    intents_for_skill = intents_by_skill(blueprint).get(job.skill_id, [])
    output_dir = _batch_output_dir(manifest, blueprint.batch_id, job.skill_id)
    # An explicit blueprint.base_seed is deliberate operator intent (e.g. reproducing
    # a specific calibration run) and always wins. Absent one, the seed is derived
    # from job_id as before -- unique per replenishment episode, with no reproducible
    # meaning beyond this one job.
    seed = (
        blueprint.base_seed
        if blueprint.base_seed is not None
        else int(hashlib.sha256(job.job_id.encode("utf-8")).hexdigest()[:8], 16)
    )

    # Recomputed fresh on every tick, right before the only network call this stage
    # makes: an approved item added by another job or a human since retrieval last
    # ran must be respected -- never generate a slot that became satisfied in the
    # meantime. See authoring/replenishment/demand.py.
    approved_item_ids = load_approved_item_ids(manifest)
    demand = compute_blueprint_slot_demand(blueprint, job.skill_id, approved_item_ids)
    if demand and not deficient_slots(demand):
        job_repository.mark_no_longer_needed(
            job.job_id,
            error_code="demand_already_satisfied",
            error_message=(
                f"every intent slot in {blueprint.batch_id} for {job.skill_id} became "
                "an approved bank item since this job last ran; nothing left to generate"
            ),
            metadata={
                "demand_fingerprint": _demand_fingerprint(
                    blueprint, job.skill_id, approved_item_ids, manifest
                ),
            },
        )
        return
    skip_question_indices = frozenset(slot.question_index for slot in demand if slot.satisfied)

    # Checked before model_factory() is ever called: a blueprint/intent mismatch is
    # deterministic given the current blueprint file, so it must fail closed without
    # spending a single model call, not surface only after one is attempted.
    try:
        difficulty = _blueprint_generation_difficulty(job.skill_id, intents_for_skill)
    except BatchGenerationError as exc:
        job_repository.mark_permanent_failure(
            job.job_id,
            error_code="generation_config_error",
            error_message=str(exc),
        )
        return

    try:
        model = model_factory()
        config = BatchConfig(
            batch_id=blueprint.batch_id,
            skill_ids=[job.skill_id],
            # The full pool, not job.requested_count: skip_question_indices (not a
            # truncated count) is what excludes already-satisfied slots, so every
            # deficient slot is reachable regardless of where in the pool it sits.
            questions_per_skill=len(intents_for_skill),
            base_seed=seed,
            model_id=model.model_id,
            prompt_version=blueprint.prompt_version,
            difficulty=difficulty,
        )
        result = generate_batch(
            config,
            model,
            output_dir,
            skills_path=manifest.skills_path(),
            references_path=manifest.references_path(),
            provenance_path=manifest.provenance_path(),
            clock=clock,
            git_commit=current_commit_allow_dirty(),
            resume=(output_dir / "manifest.json").is_file(),
            skip_question_indices=skip_question_indices,
            templates=DETERMINISTIC_TEMPLATES,
        )
    except ModelUnavailableError:
        job_repository.mark_waiting(
            job.job_id,
            "waiting_for_model",
            job_type="generate_questions",
            metadata={"batch_id": blueprint.batch_id, "output_dir": str(output_dir)},
        )
        return
    except BatchGenerationError as exc:
        # Also deterministic given the current blueprint/candidate state:
        # generate_batch raises this only from its own pre-generation validation,
        # before any model.generate() call, so retrying without a human changing that
        # state would reproduce the identical failure forever.
        job_repository.mark_permanent_failure(
            job.job_id,
            error_code="generation_config_error",
            error_message=str(exc),
        )
        return

    if result.status == "incomplete":
        # generate_batch catches every exception model.generate() raises
        # internally (including ModelUnavailableError) and records it as a
        # plain invalid attempt, so a model-down condition mid-batch never
        # escapes as an exception -- it has to be recognized from the audit.
        model_unavailable = any(
            attempt.validation_error is not None
            and attempt.validation_error.startswith("ModelUnavailableError:")
            for attempt in result.attempts
        )
        if model_unavailable:
            job_repository.mark_waiting(
                job.job_id,
                "waiting_for_model",
                job_type="generate_questions",
                metadata={"batch_id": blueprint.batch_id, "output_dir": str(output_dir)},
            )
            return
        job_repository.mark_retryable_failure(
            job.job_id,
            error_code="validation_incomplete",
            error_message=f"batch {blueprint.batch_id} did not complete: {result.summary}",
            max_attempts=max_attempts,
            clock=clock,
        )
        return

    review_path = manifest.review_store_path / f"{blueprint.batch_id}__{job.skill_id}.json"
    if not review_path.exists():
        GroundedReviewStore(review_path).save(build_pending_review(output_dir))

    # Automated review runs next, before a human ever sees this batch -- see
    # _handle_automated_review below. output_dir is not carried in metadata: it is
    # always re-derived from (manifest, review.batch_id, job.skill_id) via
    # _batch_output_dir, so it can never drift from the review file's own batch_id.
    job_repository.mark_queued(
        job.job_id,
        job_type="automated_review",
        metadata={"batch_id": blueprint.batch_id, "review_path": str(review_path)},
    )


def _reviewed_head(
    item: CurationItem, source: PendingQuestion
) -> tuple[QuizQuestion, list]:
    """The exact content currently up for automated review: the latest pending
    revision if one exists, else the immutable original candidate."""
    pending_revisions = [
        revision for revision in item.revisions if revision.final_review_status == "pending"
    ]
    head = pending_revisions[-1].question if pending_revisions else source.question
    return head, pending_revisions


def _curation_recommendation(report: AutomatedReviewReport) -> str:
    """Map an AutomatedReviewReport's recommendation onto CurationItem's
    Recommendation vocabulary.

    Never emits "approve_as_written": that value is immutable curation metadata in
    this codebase (CurationItem's decision_metadata_matches_status validator forbids
    ever approving a revision once it is set), and the worker-driven pipeline has
    always routed every item through propose_revision + approve_revision, even for a
    candidate that turns out to need no changes -- automated review's "nothing wrong
    here" opinion is communicated through recommendation_reason and the report itself,
    not by claiming an editorial authority only a human curator (see
    authoring/pilot_curation_v3.py) has ever exercised.
    """
    if report.recommendation == "recommend_human_approval":
        return "propose_revision"
    return report.recommendation


def _summarize_report(report: AutomatedReviewReport) -> str:
    if report.blocking_reasons:
        return f"Automated review ({report.risk_level}): " + "; ".join(report.blocking_reasons)
    if report.warnings:
        return f"Automated review ({report.risk_level}): " + "; ".join(report.warnings)
    return f"Automated review ({report.risk_level}): no issues found."


def _apply_automated_recommendation(
    review_store: GroundedReviewStore, question_id: str, recommendation: str, reason: str
) -> None:
    """Reload fresh and only touch an item still pending -- if a human decided it
    while this job was mid-batch, automated review must never overwrite that."""
    review = review_store.load()
    try:
        current = get_item(review, question_id)
    except KeyError:
        return
    if current.final_review_status != "pending":
        return
    review_store.replace_item(
        current.model_copy(update={"recommendation": recommendation, "recommendation_reason": reason})
    )


def _automated_revision_attempts(job: ReplenishmentJob) -> dict[str, int]:
    return dict(job.metadata.get("automated_revision_attempts", {}))


def _automated_revision_attempt_count(item: CurationItem, attempts: dict[str, int]) -> int:
    # Counts attempts, not successful revisions: a "correction" identical to the
    # original candidate raises ValueError in propose_automated_revision without ever
    # adding a QuestionRevision, so counting item.revisions could never bound a
    # generator stuck producing no-op edits -- attempts (tracked in job.metadata,
    # incremented whether or not a revision was actually recorded) always advance.
    return attempts.get(item.original_question_id, 0)


def _handle_automated_review(
    job: ReplenishmentJob,
    manifest: CourseManifest,
    *,
    job_repository: SQLiteReplenishmentJobRepository,
    reviewer_factory: Callable[[], ContentReviewer],
    review_config: ReviewPolicyConfig,
    clock: Callable[[], datetime],
) -> None:
    review_path_value = job.metadata.get("review_path")
    if not review_path_value:
        job_repository.mark_permanent_failure(
            job.job_id,
            error_code="missing_review_reference",
            error_message="job has no recorded review_path for automated review",
        )
        return

    review_path = Path(review_path_value)
    review_store = GroundedReviewStore(review_path)
    review = review_store.load()
    output_dir = _batch_output_dir(manifest, review.batch_id, job.skill_id)
    source_questions = {question.question_id: question for question in load_source_questions(output_dir)}

    catalogue = load_skills(manifest.skills_path(), manifest.references_path())
    skill = next((candidate for candidate in catalogue.skills if candidate.skill_id == job.skill_id), None)
    if skill is None:
        job_repository.mark_permanent_failure(
            job.job_id,
            error_code="missing_review_context",
            error_message=f"cannot run automated review for {job.skill_id} without its skill/blueprint",
        )
        return
    # review.batch_id is the exact, already-resolved batch this review file was built
    # from at generation time (see _handle_generate_questions) -- loaded directly by
    # id, never re-scanned by blueprint file modification time.
    try:
        blueprint = load_blueprint_for_batch(review.batch_id)
    except ValueError as exc:
        job_repository.mark_permanent_failure(
            job.job_id, error_code="invalid_batch_id", error_message=str(exc)
        )
        return
    intents_for_skill = {
        intent.intent_id: intent for intent in intents_by_skill(blueprint).get(job.skill_id, [])
    }

    known_skill_ids = {candidate.skill_id for candidate in catalogue.skills}
    provenance = load_reference_provenance(manifest.provenance_path(), known_skill_ids)
    approved_references = [record for record in provenance if record.skill_id == job.skill_id]

    report_store = AutomatedReviewReportStore(
        review_report_path(manifest.review_store_path, review.batch_id, job.skill_id)
    )
    reviewer = reviewer_factory()
    # "Duplicate item id" means already promoted, not merely present in this pending
    # batch's own review file -- every item in that file is, trivially, itself.
    current_bank_path = active_bank_path(manifest)
    existing_item_ids = (
        {bank_item.item_id for bank_item in load_approved_bank(current_bank_path) if bank_item.item_id}
        if current_bank_path.is_file()
        else set()
    )
    attempts = _automated_revision_attempts(job)

    calls_made = 0
    budget_exhausted = False
    any_pending = False
    any_require_full_human_review = False
    any_revision_eligible = False
    any_needs_human_review = False
    any_terminally_rejected_semantic = False
    any_terminally_rejected_deterministic = False

    for item in list_pending(review):
        source = source_questions.get(item.original_question_id)
        if source is None:
            continue
        any_pending = True
        head_question, pending_revisions = _reviewed_head(item, source)
        content_hash = question_content_hash(head_question)

        if report_store.latest_for_hash(content_hash) is None:
            if calls_made >= review_config.max_calls_per_job_tick:
                budget_exhausted = True
                continue
            calls_made += 1

        intent = intents_for_skill.get(item.intent_id)
        if intent is None:
            continue
        candidate = (
            source if not pending_revisions else source.model_copy(update={"question": head_question})
        )

        try:
            report = review_candidate(
                candidate,
                skill,
                intent,
                approved_references,
                reviewer=reviewer,
                config=review_config,
                report_store=report_store,
                existing_item_ids=existing_item_ids,
                clock=clock,
            )
        except ReviewerUnavailableError:
            job_repository.mark_waiting(
                job.job_id, "waiting_for_model", job_type="automated_review", metadata=job.metadata
            )
            return

        recommendation = _curation_recommendation(report)
        _apply_automated_recommendation(
            review_store, item.original_question_id, recommendation, _summarize_report(report)
        )

        automated_count = _automated_revision_attempt_count(item, attempts)
        if report.recommendation == "require_full_human_review":
            any_require_full_human_review = True
        elif (
            not review_config.shadow_mode
            and (
                report.recommendation == "propose_revision"
                or (report.recommendation == "reject" and is_localized_issue(report))
            )
            and automated_count < review_config.max_automatic_revisions
        ):
            any_revision_eligible = True
        elif report.recommendation == "reject":
            # A deterministic-only reject (e.g. no_duplicate_item_id) never consulted
            # the semantic reviewer at all -- it must not be mistaken for one that did.
            if report.semantic_review_performed:
                any_terminally_rejected_semantic = True
            else:
                any_terminally_rejected_deterministic = True
        else:
            any_needs_human_review = True

    if budget_exhausted:
        job_repository.mark_queued(job.job_id, job_type="automated_review")
        return
    if not any_pending:
        job_repository.mark_permanent_failure(
            job.job_id,
            error_code="no_pending_items_to_review",
            error_message="automated review found no matching pending items for this batch",
        )
        return
    if any_require_full_human_review:
        job_repository.mark_waiting(
            job.job_id,
            "waiting_for_full_human_review",
            job_type="automated_review",
            metadata=job.metadata,
        )
        return
    if any_revision_eligible:
        job_repository.mark_queued(job.job_id, job_type="automated_revision")
        return
    if any_needs_human_review:
        job_repository.mark_waiting(
            job.job_id, "waiting_for_question_review", job_type="automated_review", metadata=job.metadata
        )
        return
    if any_terminally_rejected_semantic:
        job_repository.mark_waiting(
            job.job_id,
            "rejected_by_automated_review",
            job_type="automated_review",
            metadata=job.metadata,
        )
        return
    if any_terminally_rejected_deterministic:
        job_repository.mark_waiting(
            job.job_id,
            "rejected_deterministically",
            job_type="automated_review",
            metadata=job.metadata,
        )
        return
    job_repository.mark_waiting(
        job.job_id, "waiting_for_question_review", job_type="automated_review", metadata=job.metadata
    )


def _handle_automated_revision(
    job: ReplenishmentJob,
    manifest: CourseManifest,
    *,
    job_repository: SQLiteReplenishmentJobRepository,
    model_factory: Callable[[], BatchModel],
    review_config: ReviewPolicyConfig,
    clock: Callable[[], datetime],
) -> None:
    review_path_value = job.metadata.get("review_path")
    if not review_path_value:
        job_repository.mark_permanent_failure(
            job.job_id,
            error_code="missing_review_reference",
            error_message="job has no recorded review_path for automated revision",
        )
        return

    review_path = Path(review_path_value)
    review_store = GroundedReviewStore(review_path)
    review = review_store.load()
    output_dir = _batch_output_dir(manifest, review.batch_id, job.skill_id)
    source_questions = {question.question_id: question for question in load_source_questions(output_dir)}

    catalogue = load_skills(manifest.skills_path(), manifest.references_path())
    skill = next((candidate for candidate in catalogue.skills if candidate.skill_id == job.skill_id), None)
    if skill is None:
        job_repository.mark_permanent_failure(
            job.job_id,
            error_code="missing_review_context",
            error_message=f"cannot run automated revision for {job.skill_id} without its skill/blueprint",
        )
        return
    # See _handle_automated_review's identical comment: review.batch_id is already
    # resolved and immutable, loaded directly by id, never re-scanned by mtime.
    try:
        blueprint = load_blueprint_for_batch(review.batch_id)
    except ValueError as exc:
        job_repository.mark_permanent_failure(
            job.job_id, error_code="invalid_batch_id", error_message=str(exc)
        )
        return
    intents_for_skill = {
        intent.intent_id: intent for intent in intents_by_skill(blueprint).get(job.skill_id, [])
    }

    known_skill_ids = {candidate.skill_id for candidate in catalogue.skills}
    provenance = load_reference_provenance(manifest.provenance_path(), known_skill_ids)
    approved_references = [record for record in provenance if record.skill_id == job.skill_id]

    report_store = AutomatedReviewReportStore(
        review_report_path(manifest.review_store_path, review.batch_id, job.skill_id)
    )

    try:
        model = model_factory()
    except ModelUnavailableError:
        job_repository.mark_waiting(
            job.job_id, "waiting_for_model", job_type="automated_revision", metadata=job.metadata
        )
        return

    seed_base = int(hashlib.sha256(job.job_id.encode("utf-8")).hexdigest()[:8], 16)
    attempts = _automated_revision_attempts(job)

    for item in list_pending(review):
        source = source_questions.get(item.original_question_id)
        if source is None:
            continue
        head_question, pending_revisions = _reviewed_head(item, source)
        content_hash = question_content_hash(head_question)
        report = report_store.latest_for_hash(content_hash)
        if report is None:
            continue  # not yet reviewed this tick -- automated_review handles it next

        automated_count = _automated_revision_attempt_count(item, attempts)
        needs_revision = report.recommendation == "propose_revision" or (
            report.recommendation == "reject" and is_localized_issue(report)
        )
        if not needs_revision or automated_count >= review_config.max_automatic_revisions:
            continue

        intent = intents_for_skill.get(item.intent_id)
        if intent is None:
            continue

        # Counted whether or not this attempt actually produces a usable revision --
        # a generator stuck reproducing the original candidate must still exhaust its
        # budget and stop, not loop between this stage and automated_review forever.
        attempts[item.original_question_id] = automated_count + 1

        try:
            revised_question = generate_revision_candidate(
                source,
                report,
                skill,
                intent,
                approved_references,
                model=model,
                seed=seed_base + automated_count,
            )
        except ModelUnavailableError:
            job_repository.mark_waiting(
                job.job_id,
                "waiting_for_model",
                job_type="automated_revision",
                metadata={**job.metadata, "automated_revision_attempts": attempts},
            )
            return
        except ValueError:
            continue  # generation produced something unusable; leave it for a human

        current = review_store.load()
        try:
            current_item = get_item(current, item.original_question_id)
        except KeyError:
            continue
        if current_item.final_review_status != "pending":
            continue  # a human already decided this item while the worker was busy

        try:
            updated_item = propose_automated_revision(
                current_item,
                source.question,
                revised_question,
                report,
                RevisionProvenance.from_source(source),
                clock=clock,
            )
        except ValueError:
            continue  # e.g. the "correction" is identical to the original candidate
        review_store.replace_item(updated_item)

    # Every item this stage touched now has a fresh, unreviewed content hash (or was
    # left untouched for a reason automated_review will re-derive) -- restart review
    # from the top rather than tracking per-item next-steps here.
    job_repository.mark_queued(
        job.job_id,
        job_type="automated_review",
        metadata={**job.metadata, "automated_revision_attempts": attempts},
    )


def _handle_promote_approved_items(
    job: ReplenishmentJob,
    manifest: CourseManifest,
    *,
    job_repository: SQLiteReplenishmentJobRepository,
    clock: Callable[[], datetime],
) -> None:
    review_path_value = job.metadata.get("review_path")
    if not review_path_value:
        job_repository.mark_permanent_failure(
            job.job_id,
            error_code="missing_review_reference",
            error_message="job has no recorded review_path to promote from",
        )
        return

    review_path = Path(review_path_value)
    review = GroundedReviewStore(review_path).load()
    output_dir = _batch_output_dir(manifest, review.batch_id, job.skill_id)
    source_questions = (
        {question.question_id: question for question in load_source_questions(output_dir)}
        if output_dir.is_dir()
        else {}
    )

    # Promotion safety: an unedited, Llama-generated item (approve_as_written) may not
    # enter the bank without evidence automated review actually looked at it. A human
    # who explicitly authored or approved a revision (propose_revision/approve_revision)
    # remains the ultimate authority regardless of automated review's opinion -- this
    # gate only closes the one path where autonomous generated content could otherwise
    # reach a learner with no review at all.
    approve_as_written_items = [
        item
        for item in review.items
        if item.final_review_status == "approved" and item.recommendation == "approve_as_written"
    ]
    if approve_as_written_items:
        report_store = AutomatedReviewReportStore(
            review_report_path(manifest.review_store_path, review.batch_id, job.skill_id)
        )
        unreviewed = sorted(
            item.original_question_id
            for item in approve_as_written_items
            if (source := source_questions.get(item.original_question_id)) is None
            or report_store.latest_for_hash(question_content_hash(source.question)) is None
        )
        if unreviewed:
            job_repository.mark_permanent_failure(
                job.job_id,
                error_code="missing_automated_review_report",
                error_message=(
                    "approve_as_written item(s) have no automated review report on file, "
                    "so an unedited generated item cannot be promoted: " + ", ".join(unreviewed)
                ),
            )
            return

    new_items = export_approved_bank_items(review, source_questions)
    if not new_items:
        job_repository.mark_permanent_failure(
            job.job_id,
            error_code="no_approved_items",
            error_message=f"no approved questions in {review_path} to promote",
        )
        return

    current_bank_path = active_bank_path(manifest)
    existing_items = (
        load_approved_bank(current_bank_path) if current_bank_path.is_file() else []
    )
    version = _next_bank_version(manifest)
    new_bank_path = (
        manifest.approved_bank_path.parent / f"{manifest.course_id}-approved-bank-v{version}.jsonl"
    )
    temporary_items_path = new_bank_path.with_suffix(".new-items.tmp.jsonl")
    _write_bank_jsonl(temporary_items_path, new_items)
    try:
        sources = (
            (current_bank_path, temporary_items_path)
            if existing_items
            else (temporary_items_path,)
        )
        merge_approved_banks(sources, output=new_bank_path)
    finally:
        temporary_items_path.unlink(missing_ok=True)

    validate_bank(
        new_bank_path,
        course=manifest.course_id,
        expected_count=len(existing_items) + len(new_items),
        required_skill_ids=[],
        skills_path=manifest.skills_path(),
        references_path=manifest.references_path(),
    )
    _write_active_pointer(manifest, new_bank_path, version, clock=clock)
    job_repository.mark_completed(job.job_id, clock=clock)


def process_job(
    job: ReplenishmentJob,
    manifest: CourseManifest,
    *,
    job_repository: SQLiteReplenishmentJobRepository,
    search_provider: SearchProvider,
    fetcher: PageFetcher,
    model_factory: Callable[[], BatchModel] = LlamaBatchModel,
    reviewer_factory: Callable[[], ContentReviewer] = lambda: ModelBackedContentReviewer(LlamaBatchModel()),
    review_config: ReviewPolicyConfig = ReviewPolicyConfig(),
    clock: Callable[[], datetime] = utc_now,
    max_attempts: int = 5,
) -> None:
    """Run the stage matching the job's current job_type, once."""
    if job.job_type in ("replenish_skill", "retrieve_references"):
        _handle_retrieve_references(
            job,
            manifest,
            job_repository=job_repository,
            search_provider=search_provider,
            fetcher=fetcher,
            clock=clock,
            max_attempts=max_attempts,
        )
    elif job.job_type in ("generate_questions", "validate_questions"):
        _handle_generate_questions(
            job,
            manifest,
            job_repository=job_repository,
            model_factory=model_factory,
            clock=clock,
            max_attempts=max_attempts,
        )
    elif job.job_type == "automated_review":
        _handle_automated_review(
            job,
            manifest,
            job_repository=job_repository,
            reviewer_factory=reviewer_factory,
            review_config=review_config,
            clock=clock,
        )
    elif job.job_type == "automated_revision":
        _handle_automated_revision(
            job,
            manifest,
            job_repository=job_repository,
            model_factory=model_factory,
            review_config=review_config,
            clock=clock,
        )
    elif job.job_type == "promote_approved_items":
        _handle_promote_approved_items(
            job, manifest, job_repository=job_repository, clock=clock
        )
    else:
        job_repository.mark_permanent_failure(
            job.job_id,
            error_code="unknown_job_type",
            error_message=f"no handler for job_type {job.job_type!r}",
        )


def ready_to_resume(job: ReplenishmentJob, manifest: CourseManifest) -> bool:
    """Whether a waiting_for_* job's blocking condition has cleared.

    Discovers readiness by re-reading the same store a human reviewer just
    edited (CandidateStore, GroundedReviewStore) -- approving a reference or
    a question needs no change to the review code itself.
    """
    if job.status == "waiting_for_reference_review":
        candidates = CandidateStore(manifest.candidate_store_path).load()
        return any(
            candidate.skill_id == job.skill_id and candidate.review_status == "approved"
            for candidate in candidates
        )
    if job.status in (
        "waiting_for_question_review",
        "waiting_for_full_human_review",
        "rejected_by_automated_review",
        "rejected_deterministically",
    ):
        # A human can approve/reject a pending CurationItem regardless of which of
        # these four job statuses the job row happens to be parked in -- automated
        # review never sets final_review_status, so every one of them leaves every
        # item exactly as human-actionable as waiting_for_question_review always was.
        review_path_value = job.metadata.get("review_path")
        if not review_path_value:
            return False
        path = Path(review_path_value)
        if not path.is_file():
            return False
        try:
            review: GroundedReview = GroundedReviewStore(path).load()
        except (OSError, ValueError):
            return False
        return any(item.final_review_status == "approved" for item in review.items)
    if job.status == "waiting_for_model":
        return True
    return False


def process_one(
    job_repository: SQLiteReplenishmentJobRepository,
    manifests: dict[str, CourseManifest],
    *,
    search_provider_factory: Callable[[CourseManifest], SearchProvider],
    fetcher_factory: Callable[[CourseManifest], PageFetcher],
    model_factory: Callable[[], BatchModel] = LlamaBatchModel,
    reviewer_factory: Callable[[], ContentReviewer] = lambda: ModelBackedContentReviewer(LlamaBatchModel()),
    review_config: ReviewPolicyConfig = ReviewPolicyConfig(),
    clock: Callable[[], datetime] = utc_now,
    max_attempts: int = 5,
    lease_seconds: int = 300,
) -> ReplenishmentJob | None:
    """Recover/requeue what's ready, then claim and process one job."""
    job_repository.recover_expired_leases(clock=clock)
    job_repository.requeue_retryable(clock=clock)

    for waiting_job in job_repository.list_waiting():
        manifest = manifests.get(waiting_job.course_id)
        if manifest is not None and ready_to_resume(waiting_job, manifest):
            next_job_type = (
                "promote_approved_items"
                if waiting_job.status
                in (
                    "waiting_for_question_review",
                    "waiting_for_full_human_review",
                    "rejected_by_automated_review",
                    "rejected_deterministically",
                )
                else waiting_job.job_type
            )
            job_repository.mark_queued(waiting_job.job_id, job_type=next_job_type)

    job = job_repository.claim_next(lease_seconds=lease_seconds, clock=clock)
    if job is None:
        return None

    manifest = manifests[job.course_id]
    process_job(
        job,
        manifest,
        job_repository=job_repository,
        search_provider=search_provider_factory(manifest),
        fetcher=fetcher_factory(manifest),
        model_factory=model_factory,
        reviewer_factory=reviewer_factory,
        review_config=review_config,
        clock=clock,
        max_attempts=max_attempts,
    )
    return job
