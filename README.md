# Adaptive Quiz Recommender

An adaptive quiz-generation and rendering system built around Llama 3.1.

## Project goals

- Generate structured multiple-choice questions.
- Adapt quiz difficulty using learner performance.
- Track learner knowledge across concepts.
- Fine-tune Llama 3.1 using LoRA or QLoRA.
- Expose model inference through an API.
- Render quizzes through a Streamlit application.

## Project structure

- `app/` — Streamlit user interface
- `api/` — inference API
- `training/` — fine-tuning pipeline
- `knowledge_tracing/` — learner-state and adaptation logic
- `evaluation/` — model and quiz evaluation
- `configs/` — model and training configuration
- `data/` — dataset samples and generated training data
- `notebooks/` — Kaggle experiments
- `tests/` — automated tests

## Environment

The project is initially developed using a Kaggle GPU environment accessed through a VS Code Remote Tunnel.

## Setup

```bash
python -m pip install -r requirements.txt
```

## Grounded pilot batch

The grounded batch command writes pending questions, a manifest, an attempt
audit, and a summary into the output directory. It does not add questions to
the approved learner-facing bank. `MODEL_REPOSITORY` must equal `--model-id`,
and `HF_TOKEN` must be configured before running the live model. The command
also requires a clean, committed worktree so its recorded git commit identifies
the exact generation code and canonical references.

The first live batch, `grounded-pilot-20260805-v1`, is incomplete and
superseded. Its audit log is retained unchanged. Before any v2 model run, print
and review the checked-in intent blueprint:

```bash
python -m scripts.print_grounded_blueprints
```

After review, start the new batch with:

```bash
python -m scripts.generate_grounded_batch \
  --batch-id grounded-pilot-20260805-v2 \
  --skill-id AI-SRC-01 \
  --skill-id AI-SRC-02 \
  --skill-id AI-SRC-08 \
  --questions-per-skill 10 \
  --base-seed 20260805 \
  --output outputs/grounded-pilot-20260805-v2 \
  --model-id "$MODEL_REPOSITORY" \
  --prompt-version v3.3
```

If a run is interrupted or a slot is exhausted, the manifest remains
`incomplete` and accepted questions stay on disk. Use the same command with
`--resume`; accepted slots and intents are not regenerated.

## Grounded-question review

The generated v2 directory is immutable source evidence. Curation lives in a
separate review store and never rewrites generated questions, raw responses,
the manifest, or the audit log. Prepare the preliminary review queue with:

```bash
python -m scripts.review_grounded_batch \
  --batch outputs/grounded-pilot-20260805-v2 \
  --store outputs/grounded-pilot-20260805-v2-curation/review.json \
  prepare
```

The same command supports `list`, `inspect QUESTION_ID`, `propose QUESTION_ID`,
`approve QUESTION_ID REVISION_ID`, `reject QUESTION_ID --reason ...`, and
`export --output APPROVED_BANK.jsonl`. Run the command with `--help` for the
editor/reviewer arguments.
Approval always names a proposed revision, and rejection always requires a
reason. Export emits only explicitly approved revisions as learner-facing
`BankItem` JSON lines; review notes, raw prompts, failed attempts, and rejected
material are not part of that runtime contract.

Automated quality checks identify recognizable structural defects. They do not
prove factual correctness: human review remains the final factual-quality gate.

Learner-facing code must call `api.presentation.present_bank_item` immediately
before rendering an item. It deterministically shuffles a copy of the options
from the item, learner, and attempt identifiers and returns the presentation ID,
seed, stable option IDs, and displayed order needed for reconstruction. Score
with `api.presentation.score_response`; never score by option position.

## Content replenishment pipeline

A background pipeline monitors every active skill's approved-content supply
and drives it through the existing retrieval, generation, and curation
stages — never the learner request path. It reuses the modules above
unchanged: retrieval (`authoring/retrieval/`), generation
(`authoring/grounded_batch.py`), and curation (`authoring/grounded_review.py`).
It is course-neutral: `authoring/replenishment/manifests/ai.json` is the only
active manifest today; a later course adds its own manifest file, its own
reviewed intent blueprints, and its own grounding briefs — no other code
changes.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `BRAVE_SEARCH_API_KEY` | Brave Search credential (existing) | — |
| `MODEL_REPOSITORY`, `HF_TOKEN` | Llama model/tokenizer (existing) | — |
| `MODEL_REVISION` | Recorded provenance for the loaded model | `unknown` |
| `QUIZ_DATABASE_PATH` | Shared SQLite file for jobs and learner state (existing) | `data/adaptive_quiz.sqlite3` |
| `QUIZ_REPLENISHMENT_POLL_SECONDS` | Worker poll interval in continuous mode | `30` |
| `QUIZ_REPLENISHMENT_MAX_ATTEMPTS` | Attempts before a retryable failure becomes permanent | `5` |
| `QUIZ_REPLENISHMENT_LOW_SUPPLY_THRESHOLD`, `QUIZ_REPLENISHMENT_TARGET_SUPPLY` | Override the manifest's course-wide thresholds | manifest values |
| `QUIZ_REPLENISHMENT_INFERENCE_PROVIDER` | `local` (in-process `transformers`/`torch`) or `modal` (a deployed Modal endpoint) | `local` |
| `MODAL_INFERENCE_ENDPOINT` | Modal web endpoint URL (`modal` provider only) | — |
| `MODAL_PROXY_TOKEN_ID`, `MODAL_PROXY_TOKEN_SECRET` | Modal proxy auth token, sent as `Modal-Key`/`Modal-Secret` headers (`modal` provider only; omit for an unprotected endpoint) | — |

### Modal inference provider

`authoring/replenishment/modal_inference.py:ModalBatchModel` is a sibling of
`worker.py`'s in-process `LlamaBatchModel` — same `BatchModel` shape
(`model_id`, `model_revision`, `generate`), reached over HTTP instead of
loaded locally. Selecting it (`QUIZ_REPLENISHMENT_INFERENCE_PROVIDER=modal`)
requires the Modal endpoint to accept:

```
POST <MODAL_INFERENCE_ENDPOINT>
Headers: Modal-Key, Modal-Secret (if configured), Content-Type: application/json
Body:    {"messages": [...], "seed": <int>, "generation_parameters": {...}}
Response (200): {
  "text": "<raw model completion>",
  "finish_reason": "stop" | "length" | "unknown",
  "input_tokens": <int>,
  "output_tokens": <int>,
  "max_new_tokens": <int>,
  "model_revision": "<resolved commit hash>" | null
}
```

Any non-2xx response, connection failure, invalid JSON, or a missing/empty
`text` field is treated as `ModelUnavailableError` — the same outcome as the
local adapter's model failing to load — so a replenishment job moves to
`waiting_for_model` and retries on the worker's next pass rather than
failing. The timeout (180s by default) is deliberately generous to tolerate
a cold-started GPU container without misreporting a slow start as an outage.

`finish_reason`/`input_tokens`/`output_tokens`/`max_new_tokens`/`model_revision`
are additive — `ModalBatchModel.generate()` (generation) ignores them; only
`generate_with_metadata()` (used by the automated review layer's reviewer
path, next section) reads them, defaulting to `"unknown"`/`0` if an
older-deployed endpoint doesn't send them yet. Redeploy `modal_app.py`
(`modal deploy modal_app.py`) to start returning real values. `model_revision`
only fills in `ModalBatchModel.model_revision` when no explicit pin
(constructor arg or `MODEL_REVISION` env var) was given — an intentional pin
always wins over the server-reported value.

### Automated review layer's reviewer token budget

The reviewer's own output budget is independent of generation's
`max_new_tokens` (which defaults to 800 — see `modal_app.py`/`worker.py`'s
`LlamaBatchModel`): `QUIZ_REVIEW_MAX_NEW_TOKENS` (default `1000`) controls it,
read by `authoring/review/config.py:ReviewPolicyConfig.reviewer_max_new_tokens`
and passed through `ModelBackedContentReviewer`. Raising or lowering one never
silently moves the other.

### Local worker startup

Continuous polling, for local demonstration:

```bash
python -m authoring.replenishment.cli worker
```

### One-shot execution

Claims and processes at most one job, then exits — this is the shape a cron
job or a manual "run one step" invocation should use:

```bash
python -m authoring.replenishment.cli worker --once
```

### Trigger a low-supply scan

Computes inventory for every active manifest and enqueues a `replenish_skill`
job for each skill whose unseen approved supply is below its threshold.
Rerunning `scan` is idempotent — an already-active job blocks a duplicate:

```bash
python -m authoring.replenishment.cli scan
python -m authoring.replenishment.cli status
```

### Approve references

Retrieval pauses at `waiting_for_reference_review`. Approve or reject through
the existing retrieval CLI — it reads and writes the same candidate store the
worker uses, so no new tooling exists for this step:

```bash
python -m authoring.retrieval.cli list --status pending
python -m authoring.retrieval.cli approve <candidate_id> --reviewer <you>
```

### Resume generation

Nothing to run by hand: the worker's next pass re-checks every
`waiting_for_reference_review` job against the candidate store, and once an
approved reference exists it advances the job to `generate_questions`
automatically (`worker --once` or the continuous `worker` loop).

### Approve questions

Generation pauses at `waiting_for_question_review`. Review and approve
through the existing curation CLI, using the immutable batch directory
(`outputs/replenishment/<course>/batches/<batch_id>__<skill_id>/`, also
recorded as `output_dir` in the job's metadata) and the review store path the
worker created (`metadata.review_path`, printed by `status`):

```bash
python -m scripts.review_grounded_batch --batch <output_dir> --store <review_path> list
python -m scripts.review_grounded_batch --batch <output_dir> --store <review_path> propose QUESTION_ID --question-json edited.json --editor <you> --note "..."
python -m scripts.review_grounded_batch --batch <output_dir> --store <review_path> approve QUESTION_ID REVISION_ID --reviewer <you>
```

### Promote a new bank

Automatic once a question is approved: the worker's next pass moves the job
to `promote_approved_items`, which writes a new versioned bank file
(`<course>-approved-bank-v<N>.jsonl`, next to the existing approved bank),
validates it, and atomically updates `<course>-active-bank.json` to point at
it. The original approved bank file is never modified. Flipping the running
app's `QUIZ_APPROVED_BANK_PATH` to the new version is a separate, manual
deploy step — see limitations below.

### Behavior when Brave or Llama is offline

- Missing `BRAVE_SEARCH_API_KEY`: the retrieval stage records a
  `retryable_failure` (`missing_brave_credentials`) with bounded backoff.
  Quizzes are unaffected.
- Missing `MODEL_REPOSITORY`/`HF_TOKEN`, or any model load/inference failure:
  the job moves to `waiting_for_model` and is retried on every worker pass
  until the model becomes reachable. It is never treated as a failure.
- In both cases the learner-facing app never calls either service and keeps
  serving from the currently active approved bank.

### Limitations of Streamlit Community Cloud background processing

Streamlit Community Cloud runs one web process with no persistent background
worker slot. `authoring.replenishment.cli worker` must run outside that
process — locally, via `cron`/`systemd`, or as a separate scheduled job —
against the same SQLite database file. The Streamlit app itself only ever
reads job/inventory state for the admin sidebar; it never starts a worker
thread.

### Production migration path

The current design targets a single local SQLite file shared by one
Streamlit process and one worker process. Moving to production means:

- Replacing the SQLite job table with a durable, concurrent-safe database
  (Postgres) behind the same `SQLiteReplenishmentJobRepository` interface.
- Running the worker as a standing hosted process (or a scheduled task queue
  worker) instead of a manual/cron-invoked CLI.
- Replacing the `QUIZ_APPROVED_BANK_PATH` manual deploy step with a small
  release step that reads `<course>-active-bank.json` and redeploys the app
  with the new path.
