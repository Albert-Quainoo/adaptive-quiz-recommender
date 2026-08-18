# Equivalence-gate NLI threshold calibration

Model: `cross-encoder/nli-deberta-v3-xsmall` @ `a150876415327c80daeff35ca6f68f5ed8cf5c24`

Calibration set: 16 pairs. Held-out set: 16 pairs.

Selected threshold: **0.25** (achieved zero false positives on the calibration set).

## Held-out evaluation (selected threshold, no retuning)

- precision: 1.000
- recall: 0.500
- false_positive_rate: 0.000
- false_negative_rate: 0.500
- confusion: tp=2 fp=0 tn=12 fn=2
- mean latency per pair (2 forward passes, CPU): 96.3ms

## Calibration-set threshold sweep

| threshold | precision | recall | f1 | fp | fn |
|---|---|---|---|---|---|
| 0.05 | 0.333 | 0.250 | 0.286 | 2 | 3 |
| 0.1 | 0.500 | 0.250 | 0.333 | 1 | 3 |
| 0.15 | 0.500 | 0.250 | 0.333 | 1 | 3 |
| 0.2 | 0.500 | 0.250 | 0.333 | 1 | 3 |
| 0.25 | 1.000 | 0.250 | 0.400 | 0 | 3 |
| 0.3 | 1.000 | 0.250 | 0.400 | 0 | 3 |
| 0.35 | 1.000 | 0.250 | 0.400 | 0 | 3 |
| 0.4 | 1.000 | 0.250 | 0.400 | 0 | 3 |
| 0.45 | 1.000 | 0.250 | 0.400 | 0 | 3 |
| 0.5 | 1.000 | 0.250 | 0.400 | 0 | 3 |
| 0.55 | 1.000 | 0.000 | 0.000 | 0 | 4 |
| 0.6 | 1.000 | 0.000 | 0.000 | 0 | 4 |
| 0.65 | 1.000 | 0.000 | 0.000 | 0 | 4 |
| 0.7 | 1.000 | 0.000 | 0.000 | 0 | 4 |
| 0.75 | 1.000 | 0.000 | 0.000 | 0 | 4 |
| 0.8 | 1.000 | 0.000 | 0.000 | 0 | 4 |
| 0.85 | 1.000 | 0.000 | 0.000 | 0 | 4 |
| 0.9 | 1.000 | 0.000 | 0.000 | 0 | 4 |
| 0.95 | 1.000 | 0.000 | 0.000 | 0 | 4 |

## Held-out per-pair results

| pair_id | category | label | min_entailment | predicted | correct |
|---|---|---|---|---|---|
| ai-paraphrase-2 | paraphrase | True | 0.982 | True | yes |
| ai-unit-1 | unit_equivalence | False | 0.004 | False | yes |
| ai-related-distractor-1 | related_distractor | False | 0.000 | False | yes |
| ai-scope-1 | scope_quantifier_change | False | 0.002 | False | yes |
| dsa-paraphrase-2 | paraphrase | True | 0.032 | False | NO |
| dsa-math-1 | math_equivalence | False | 0.002 | False | yes |
| dsa-negation-1 | negation | False | 0.000 | False | yes |
| dsa-causal-reversal-1 | causal_reversal | False | 0.001 | False | yes |
| la-paraphrase-2 | paraphrase | True | 0.961 | True | yes |
| la-math-1 | math_equivalence | False | 0.001 | False | yes |
| la-negation-1 | negation | False | 0.000 | False | yes |
| la-causal-reversal-1 | causal_reversal | False | 0.007 | False | yes |
| db-paraphrase-2 | paraphrase | True | 0.081 | False | NO |
| db-math-1 | math_equivalence | False | 0.002 | False | yes |
| db-negation-1 | negation | False | 0.000 | False | yes |
| db-causal-reversal-1 | causal_reversal | False | 0.003 | False | yes |
