# Adaptive Quiz Recommender

An adaptive quiz platform that tracks what a learner knows with Bayesian
Knowledge Tracing (BKT) and recommends the next question from that state,
across four courses: **Introduction to AI**, **Data Structures & Algorithms**,
**Linear Algebra**, and **Database Systems** (65 skills total). Question
content is LLM-generated (Llama 3.1) and grounded against retrieved,
human-approved reference material — never invented — then carried through an
automated + human review pipeline before it reaches a learner.

## What's here

- **Adaptive learner loop** (`bkt/`, `recommendation/`, `app/`, `api/`): a
  per-skill BKT model estimates mastery from each attempt; the recommender
  picks the next question from that state; Streamlit renders it.
- **Grounded content generation** (`authoring/`): retrieval finds candidate
  reference material on an explicit domain allowlist, generation drafts
  multiple-choice questions grounded in an approved reference, and curation
  (automated review + human approval) gates everything before it reaches the
  approved bank a learner actually sees.
- **Content replenishment pipeline** (`authoring/replenishment/`): a
  background worker monitors each course's approved-question supply and
  drives deficient skills through retrieval → generation → review →
  promotion automatically, stopping at explicit human-approval boundaries —
  see [Content replenishment pipeline](#content-replenishment-pipeline) below.
- **Taxonomy** (`taxonomy/`): each course's skills, learning objectives, and
  reference provenance live in `taxonomy/data/<course>/`, reviewed and
  versioned independently of code.

## Quick start

```bash
python -m pip install -r requirements.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml  # fill in your keys
streamlit run app/main.py
```

By default the app runs against a local SQLite database
(`data/adaptive_quiz.sqlite3`, created on first run) and the bundled,
already-approved question banks under `outputs/approved_banks/` — no live
Brave/Modal/Llama credentials are required just to try the learner-facing
quiz loop. Those credentials are only needed to run content
retrieval/generation yourself (see below).

Run the test suite with:

```bash
python -m pytest
```

## Project structure

- `app/` — Streamlit UI and the learner-facing application controller
  (course selection, session state, presentation)
- `api/` — question presentation/scoring contracts and the prompt builder
- `bkt/` — Bayesian Knowledge Tracing model, training, and repository
- `recommendation/` — next-question selection policy
- `authoring/` — grounded question generation, curation, retrieval, and the
  replenishment worker/pipeline
- `taxonomy/` — per-course skills, learning objectives, and reference
  provenance (`taxonomy/data/<course>/`)
- `evaluation/` — model, retrieval, and quiz-quality evaluation
- `training/` — Llama 3.1 LoRA/QLoRA fine-tuning pipeline
- `knowledge_tracing/` — shared learner-state utilities
- `scripts/` — operational CLIs and one-off tooling (content review, batch
  generation, replenishment control, calibration)
- `configs/` — model and training configuration
- `data/` — local SQLite database and generated training data
- `outputs/` — generated/approved content artifacts (mostly gitignored; the
  active approved banks under `outputs/approved_banks/` are tracked)
- `docs/` — standalone reference documents (e.g. the cross-course content
  review packet)
- `tests/` — automated tests

## Environment

Model development was done in a Kaggle GPU environment (see `ENVIRONMENT.md`
for that runtime's specifics); the application itself is plain Python 3.12
and runs anywhere the dependencies above install.

## Manual batch generation and the content replenishment pipeline

The sections below document the underlying mechanics: a one-off manual batch
command (`scripts/generate_grounded_batch.py`), then the automated
replenishment pipeline built on top of the same generation/curation modules,
which is what actually keeps all four courses' approved banks stocked today
(see [Content replenishment pipeline](#content-replenishment-pipeline)).

### Grounded pilot batch

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

### Grounded-question review

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
It is course-neutral: `authoring/replenishment/manifests/*.json` holds one
manifest per active course (`intro-ai`, `dsa`, `linear-algebra`,
`database-systems` today); a later course adds its own manifest file, its
own reviewed intent blueprints, and its own grounding briefs — no other code
changes.

### Environment variables

| Variable | Purpose | Default |
|---|---|---|
| `BRAVE_SEARCH_API_KEY` | Brave Search credential (existing) | — |
| `MODEL_REPOSITORY`, `HF_TOKEN` | Llama model/tokenizer (existing) | — |
| `MODEL_REVISION` | Recorded provenance for the loaded model | `unknown` |
| `QUIZ_DATABASE_URL` | Production PostgreSQL DSN for jobs and learner state (e.g. Supabase). When set, used exclusively -- never falls back to SQLite. | — |
| `QUIZ_DATABASE_PATH` | Local-dev/test SQLite file for jobs and learner state, used only when `QUIZ_DATABASE_URL` is unset | `data/adaptive_quiz.sqlite3` |
| `QUIZ_ADMIN_STATUS_ENABLED` | Shows the read-only replenishment/content-status sidebar in the learner-facing app (`true`/`1`/`yes`/`on`). An operator/debugging aid, not a learner feature -- the public app must never scan every course's taxonomy and bank to render it. Leave unset (disabled) in production. When enabled, the scan is cached with a 60s TTL, shared across all sessions. | `false` |
| `QUIZ_REPLENISHMENT_POLL_SECONDS` | Worker poll interval in continuous mode | `30` |
| `QUIZ_REPLENISHMENT_MAX_ATTEMPTS` | Attempts before a retryable failure becomes permanent | `5` |
| `QUIZ_REPLENISHMENT_LOW_SUPPLY_THRESHOLD`, `QUIZ_REPLENISHMENT_TARGET_SUPPLY` | Override the manifest's course-wide thresholds | manifest values |
| `QUIZ_REPLENISHMENT_INFERENCE_PROVIDER` | `local` (in-process `transformers`/`torch`) or `modal` (a deployed Modal endpoint) | `local` |
| `MODAL_INFERENCE_ENDPOINT` | Modal web endpoint URL (`modal` provider only) | — |
| `MODAL_PROXY_TOKEN_ID`, `MODAL_PROXY_TOKEN_SECRET` | Modal proxy auth token, sent as `Modal-Key`/`Modal-Secret` headers (`modal` provider only; omit for an unprotected endpoint) | — |
| `QUIZ_REPLENISHMENT_MAX_NEW_CANDIDATES` | `scripts/run_replenishment_cycle.py`: brand-new skill episodes (never claimed before) a single run may start | `3` |
| `QUIZ_REPLENISHMENT_MAX_GENERATION_CALLS` | `scripts/run_replenishment_cycle.py`: combined generation+review calls a single run may make | `20` |
| `QUIZ_REPLENISHMENT_MAX_COST_USD`, `QUIZ_REPLENISHMENT_COST_PER_GENERATION_CALL_USD`, `QUIZ_REPLENISHMENT_COST_PER_REVIEW_CALL_USD`, `QUIZ_REPLENISHMENT_COST_PER_SEARCH_CALL_USD` | `scripts/run_replenishment_cycle.py`'s dollar ceiling and its per-call estimate (coarse -- no per-token accounting is plumbed out of generation today) | `5.00`, `0.05`, `0.02`, `0.00` |
| `QUIZ_REPLENISHMENT_MAX_TICKS` | Safety valve bounding total job-processing ticks in one run, independent of the caps above | `500` |
| `QUIZ_REPLENISHMENT_RETENTION_DAYS` | Days a terminal job's snapshot under `outputs/replenishment/<course>/<job_id>/` keeps its full artifacts before being compacted to `archived.json` | `14` |

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
against the same database. The Streamlit app itself only ever
reads job/inventory state for the admin sidebar; it never starts a worker
thread.

### Scheduled GitHub Actions replenishment

`.github/workflows/replenishment.yml` is one such separate scheduled job: it
runs `scripts/run_replenishment_cycle.py` on a daily schedule (currently left
disabled pending a reviewed manual dry run -- see the workflow file's own
rollout comment) plus on-demand via `workflow_dispatch`. It is a thin
orchestration layer over the unmodified pipeline above:

- Scans the four active courses and enqueues deficient skills exactly like
  `cli.py scan` (idempotent -- the job repository's uniqueness constraint
  prevents duplicates regardless of how many times it runs).
- Drives the worker loop for at most `QUIZ_REPLENISHMENT_MAX_NEW_CANDIDATES`
  brand-new skill episodes under an explicit call/cost budget
  (`authoring/replenishment/budget.py`), stopping between ticks so the next
  scheduled run can always resume cleanly.
- Snapshots every job it touches to a deterministic, job-scoped directory --
  `outputs/replenishment/<course_id>/<job_id>/` (candidates, review, review
  reports, batch manifest, content hashes, call counts, cost estimate;
  never a credential, DSN, or raw API response) -- and commits those,
  idempotently, to a dedicated `content-ops/replenishment` branch. The
  PostgreSQL job queue (`QUIZ_DATABASE_URL`) remains the authoritative
  source of job status; this branch exists purely so a GitHub-hosted
  (ephemeral) runner can resume later runs and an admin can actually see and
  act on pending content.
- Opens or updates a standing PR from `content-ops/replenishment` into the
  default branch so an admin has a reviewable diff of everything pending --
  opening/updating that PR is never itself an approval or promotion.
  Content approval still requires the existing explicit admin action
  (`propose_revision`/`approve_revision` in `authoring/grounded_review.py`),
  and promotion still only ever writes to `content-ops/replenishment`,
  never directly to the default branch; merging the PR is the separate,
  human, final step that ships it.
- Compacts old terminal jobs' snapshots to a small `archived.json` summary
  after `QUIZ_REPLENISHMENT_RETENTION_DAYS` so the branch does not grow
  without bound.

### Production migration path

Local development and tests default to a single SQLite file shared by one
Streamlit process and one worker process. In production, setting
`QUIZ_DATABASE_URL` (e.g. a Supabase PostgreSQL DSN) switches every
repository (`SQLiteBKTRepository`, `SQLiteRecommendationRepository`,
`SQLiteReplenishmentJobRepository`) onto PostgreSQL exclusively via the
shared SQLAlchemy engine in `database.py` — no other code changes needed.
Remaining production steps:

- Running the worker as a standing hosted process (or a scheduled task queue
  worker) instead of a manual/cron-invoked CLI.
- Replacing the `QUIZ_APPROVED_BANK_PATH` manual deploy step with a small
  release step that reads `<course>-active-bank.json` and redeploys the app
  with the new path.
