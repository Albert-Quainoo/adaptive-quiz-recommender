# BKT validation: guess-parameter correction (0.40 → 0.25)

Offline synthetic behavioral evaluation, run 2026-08-19
(`python -m evaluation.bkt_guess_parameter_evaluation`), comparing each
course's currently-committed "prior" (`guesses=0.40`) and "candidate"
(`guesses=0.25`, matching true 4-option-MCQ chance rate, commit `7fa7bed`)
BKT artifacts against identical synthetic attempt sequences. Full report:
`outputs/bkt_guess_parameter_evaluation_report.json` (gitignored,
regenerate with `--output`). This is behavioral validation that the
correction changes the model the way it should — not proof `0.25` is
statistically optimal.

## Part A: mastery trajectories (324 computed, 0 non-deterministic)

Six personas × every skill in each course's prior/candidate model pair
(shared skills run both; candidate-only skills — AI-FND-03/04, added after
v4 — run candidate only), each run twice to confirm determinism. All 324
trajectories reproduced exactly on the second run.

**`lucky_guesser`** (8 wrong, 2 right — the direct probe of the guess-rate
change) is the clearest signal, identical in shape across every
course/skill (only `prior`/`slips`/`learns`/`forgets` are shared between
model pairs — `guesses` is the only value that differs):

| step | prior (guesses=0.40) | candidate (guesses=0.25) |
|---|---|---|
| 2 (before 1st correct) | 0.0794 | 0.0676 |
| 3 (after 1st correct) | **0.1813** | **0.2208** |
| 6 (before 2nd correct) | 0.0658 | 0.0589 |
| 7 (after 2nd correct) | **0.1585** | **0.2001** |
| 10 (final) | 0.0646 | 0.0582 |

A lower guess rate makes both directions of evidence more diagnostic: an
unexpected correct answer is attributed more to real learning (bigger
upward jump — 0.14 vs 0.10 at step 3, 0.09 vs 0.06 at step 7), and a wrong
answer counts against mastery more too (candidate settles lower after each
one). Because this persona is mostly wrong, the amplified downward pull
dominates the aggregate, so final mastery ends up marginally *lower* under
the candidate (0.0582 vs 0.0646) despite the individual correct-answer
jumps being larger — both correct, not contradictory: the persona-level
number and the step-level mechanism answer different questions.

## Part B: recommendation ordering, fallback, and course isolation

- Recommendation ordering ran cleanly (0 unexpected exhaustion/error
  entries) for both `prior` and `candidate` across all 4 courses.
- Cross-course isolation, real positive + negative control: one learner's
  real attempt fed into `intro-ai` only. `intro-ai`'s own repository sees
  it (positive control — proves the check can detect a leak). `dsa`,
  `linear-algebra`, `database-systems` — backed by the same physical
  SQLite file — see nothing for that learner (negative control): 3/3
  clean.

## Part C: colliding skill_id adversarial case

Same literal `skill_id` string (`AA-CLSN-01`) reused across two synthetic
`course_id`s in the same database, one seeded at `guesses=0.40` and the
other at `guesses=0.25`, both fed the identical single correct attempt:

- `course_a` (guesses=0.40): mastery → 0.36
- `course_b` (guesses=0.25): mastery → 0.4667
- `course_a_sees_course_b_attempts`: **False** — no leakage even under an
  adversarial identical-skill-id collision.
- Masteries differ solely because of the guess parameter, with identical
  skill_id and identical outcome — isolation and the parameter's effect
  both confirmed at the boundary case, not just the average case.

## Conclusion

The guess-parameter correction behaves exactly as theorized in every probe
run: sensitivity to evidence increases in both directions, determinism
holds under repetition, and course isolation holds even at an adversarial
skill_id collision. No anomalies found.
