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
