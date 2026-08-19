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
  see [docs/OPERATIONS.md](docs/OPERATIONS.md).
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

## Reproducibility

- **Learner loop**: `python -m pytest` exercises the full BKT → recommendation
  → presentation → scoring path against the bundled approved banks and
  fitted models — no external credentials needed.
- **BKT models**: each course's model (`outputs/bkt_*_model_v*.pkl`, referenced
  by `authoring/replenishment/manifests/<course>.json`) is trained offline and
  checked in fitted, never fit at app startup. Retrain with `bkt/train_dev_model.py`
  (seeded via `--seed` for deterministic synthetic attempts/fitting).
- **Question content**: every approved question is grounded in a specific,
  human-approved reference passage and carries full provenance (reviewer,
  model, prompt version, reference IDs) — see `docs/three-course-content-review.html`
  for a reviewable snapshot of the full approved-bank content.

See [docs/OPERATIONS.md](docs/OPERATIONS.md) for running content
retrieval/generation/review and the background replenishment pipeline
yourself.
