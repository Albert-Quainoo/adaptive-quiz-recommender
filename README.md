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