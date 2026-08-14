# Calibration: 38-item vs. 43-item approved bank

`outputs/approved_banks/pilot-approved-bank-43-v1.jsonl` is the active bank
(`ai-active-bank.json`, version 1), promoted from
`pilot-approved-bank-38-v1.jsonl` by adding five human-reviewed items:

- Three approved-as-written originals: `AI-SRC-01-4a2bc2762583fe53`,
  `AI-SRC-08-43ffef9fd390d752`, `AI-SRC-08-f09ba0aded06df47`.
- Two approved revisions (human override in one case — see below):
  `AI-SRC-01-dbbdfe703c3487af-rev-0892ac08baca`,
  `AI-SRC-01-75b8495a29a203c3-rev-057486360c58`.

The 38-item bank is preserved byte-for-byte
(`sha256:97a3f01b7ebe555aabbaaf48de927b5563b0ec9f64c16c325dced0512971ecf1`);
the 43-item bank is a separate, independently-validated file
(`sha256:8945e4f7f56032bb61435b5e8446c050966f7440795c3cf8440b845b76009d31`).

## Reviewer-calibration evidence preserved

Both new revisions' original automated-review verdicts remain on record,
unchanged, alongside the human approval that overrode/accepted them:

- **INT-10** (`AI-SRC-01-dbbdfe703c3487af`): automated review missed a
  stem/answer scope mismatch on the original candidate, and its own
  review-v5 pass on the revision returned `risk_level: critical`,
  `recommendation: reject` — but the reviewer's own
  `selected_option_text` was byte-identical to the declared correct answer;
  the blocking reason was a `declared_answer_matches` self-contradiction.
  Human review overrode this to `approved`, leaving `recommendation: reject`
  and its reason untouched as historical evidence.
- **INT-09** (`AI-SRC-01-75b8495a29a203c3`): automated review reported
  `satisfies_intent_blueprint: true` for a candidate that never required the
  actor/movie/endpoint mapping the intent's `required_concepts` call for.
  The revision fixed this; automated review scored it `low` risk on its own
  merits and it was approved normally.

## Simulation setup

Six synthetic learner personas (`scripts/run_pilot_simulation.py`), fixed
accuracy targets, seed `20260811`, BKT model `bkt-synthetic-v4`
(`outputs/bkt_dev_model_v4.pkl`), run against each bank from a fresh SQLite
database.

## Results

| Metric | 38-item | 43-item (same 40-attempt cap) |
|---|---|---|
| Quiz completion rate (reached the full 40-question session) | 0/6 (0%) | 6/6 (100%) |
| Bank-exhaustion rate at that cap | 6/6 (100%) | 0/6 (0%) |
| True exhaustion point (uncapped rerun, 60-cap) | 38 attempts, 6/6 | 43 attempts, 6/6 |
| Avg. questions delivered | 38.0 | 40.0 (43.0 uncapped) |
| Skill coverage | 6/6 required skills | 6/6 required skills |
| Introductory / intermediate bank supply | 18 / 20 | 20 / 23 |
| Introductory / intermediate delivered | 108 / 120 | 115 / 125 |
| Fallback-difficulty selections | 83/228 (36.4%) | 82/240 (34.2%) |
| Repeated item selections | 0 | 0 |
| Content-gap errors | 0 | 0 |
| BKT mastery ordering across personas | preserved | preserved |

At the standard 40-question session length, the 38-item bank could not
sustain a full session for any persona (all six hit
`AllEligibleItemsAttemptedError` / `BankExhaustedBelowMasteryError` exactly
at their eligible-item ceiling). The 43-item bank sustained the full session
for every persona, with a slightly *lower* fallback-difficulty rate and no
repeated selections — no regression in recommendation or BKT behavior from
the added content.

## Artifacts

- `outputs/pilot_run_summary_38.json` — full per-learner attempt log, 38-item bank, 40-cap.
- `outputs/pilot_run_summary_43.json` — full per-learner attempt log, 43-item bank, 40-cap.
- `outputs/pilot_run_summary_43_extended.json` — 43-item bank, 60-cap, true exhaustion point.
- `outputs/approved_banks/pilot-approved-bank-43-v1-validation.json` — `validate_approved_bank.py` output (43/43 items, 43 unique IDs, 0 duplicates).
