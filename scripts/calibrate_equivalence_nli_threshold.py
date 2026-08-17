"""Calibrates authoring/review/equivalence_nli.py's entailment threshold against
evaluation/equivalence_nli_labeled_pairs.py's labeled option-pair dataset.

Threshold selection uses ONLY the "calibration" split: sweep candidate thresholds,
compute precision/recall against each pair's nli_positive label (bidirectional
entailment >= threshold => predicted equivalent), and select the threshold maximizing
recall among every threshold that achieves zero false positives (precision 1.0) on the
calibration split, breaking ties by preferring the lower threshold. If no threshold
achieves zero false positives, falls back to the threshold maximizing F1 and says so
explicitly in the report -- this script never silently reports a threshold as "clean"
that isn't.

The "held_out" split is then scored exactly once, at the selected threshold, with no
retuning -- this is the number reported as this gate's real generalization performance,
not the (necessarily optimistic) calibration-split number.

Every call goes through the real pinned ONNX model (authoring/review/equivalence_nli.py
NliScorer) -- this script does make network/CPU calls, unlike the pytest suite (which
never does, see tests/conftest.py's no_default_nli_model fixture). Run explicitly:

    python -m scripts.calibrate_equivalence_nli_threshold
    python -m scripts.calibrate_equivalence_nli_threshold --output evaluation/equivalence_calibration_report.md
"""

import argparse
import time
from pathlib import Path

from authoring.review.equivalence_nli import NliScorer
from evaluation.equivalence_nli_labeled_pairs import LabeledPair, by_split

_CANDIDATE_THRESHOLDS = [round(0.05 * step, 2) for step in range(1, 20)]  # 0.05 .. 0.95


def _bidirectional_entailment(scorer: NliScorer, pair: LabeledPair) -> tuple[float, float, float]:
    premise_a = f"Given the question: {pair.stem} Claim: {pair.option_a}"
    premise_b = f"Given the question: {pair.stem} Claim: {pair.option_b}"
    started = time.perf_counter()
    forward = scorer.score(premise_a, premise_b)
    backward = scorer.score(premise_b, premise_a)
    elapsed = time.perf_counter() - started
    return min(forward.entailment, backward.entailment), forward.entailment, backward.entailment, elapsed


def _score_at_threshold(scored_pairs: list[tuple[LabeledPair, float]], threshold: float) -> dict:
    true_positive = false_positive = true_negative = false_negative = 0
    for pair, min_entailment in scored_pairs:
        predicted_positive = min_entailment >= threshold
        if pair.nli_positive and predicted_positive:
            true_positive += 1
        elif pair.nli_positive and not predicted_positive:
            false_negative += 1
        elif not pair.nli_positive and predicted_positive:
            false_positive += 1
        else:
            true_negative += 1
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 1.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 1.0
    fpr = false_positive / (false_positive + true_negative) if (false_positive + true_negative) else 0.0
    fnr = false_negative / (false_negative + true_positive) if (false_negative + true_positive) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "threshold": threshold, "precision": precision, "recall": recall,
        "false_positive_rate": fpr, "false_negative_rate": fnr, "f1": f1,
        "true_positive": true_positive, "false_positive": false_positive,
        "true_negative": true_negative, "false_negative": false_negative,
    }


def select_threshold(scored_pairs: list[tuple[LabeledPair, float]]) -> tuple[float, list[dict], bool]:
    sweep = [_score_at_threshold(scored_pairs, threshold) for threshold in _CANDIDATE_THRESHOLDS]
    zero_fp = [row for row in sweep if row["false_positive"] == 0]
    if zero_fp:
        best = max(zero_fp, key=lambda row: (row["recall"], -row["threshold"]))
        return best["threshold"], sweep, True
    best = max(sweep, key=lambda row: row["f1"])
    return best["threshold"], sweep, False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args(argv)

    scorer = NliScorer()

    calibration_pairs = by_split("calibration")
    held_out_pairs = by_split("held_out")

    calibration_scored = []
    calibration_latencies = []
    for pair in calibration_pairs:
        min_entailment, forward, backward, elapsed = _bidirectional_entailment(scorer, pair)
        calibration_scored.append((pair, min_entailment))
        calibration_latencies.append(elapsed)

    threshold, sweep, achieved_zero_fp = select_threshold(calibration_scored)

    held_out_scored = []
    held_out_latencies = []
    for pair in held_out_pairs:
        min_entailment, forward, backward, elapsed = _bidirectional_entailment(scorer, pair)
        held_out_scored.append((pair, min_entailment))
        held_out_latencies.append(elapsed)

    held_out_result = _score_at_threshold(held_out_scored, threshold)

    lines = []
    lines.append("# Equivalence-gate NLI threshold calibration\n")
    lines.append(f"Model: `cross-encoder/nli-deberta-v3-xsmall` @ `a150876415327c80daeff35ca6f68f5ed8cf5c24`\n")
    lines.append(f"Calibration set: {len(calibration_pairs)} pairs. Held-out set: {len(held_out_pairs)} pairs.\n")
    lines.append(
        f"Selected threshold: **{threshold}** "
        f"({'achieved zero false positives on the calibration set' if achieved_zero_fp else 'no threshold achieved zero false positives on the calibration set -- selected by maximum F1 instead'}).\n"
    )
    lines.append("## Held-out evaluation (selected threshold, no retuning)\n")
    lines.append(f"- precision: {held_out_result['precision']:.3f}")
    lines.append(f"- recall: {held_out_result['recall']:.3f}")
    lines.append(f"- false_positive_rate: {held_out_result['false_positive_rate']:.3f}")
    lines.append(f"- false_negative_rate: {held_out_result['false_negative_rate']:.3f}")
    lines.append(
        f"- confusion: tp={held_out_result['true_positive']} fp={held_out_result['false_positive']} "
        f"tn={held_out_result['true_negative']} fn={held_out_result['false_negative']}"
    )
    mean_latency = sum(held_out_latencies) / len(held_out_latencies) if held_out_latencies else 0.0
    lines.append(f"- mean latency per pair (2 forward passes, CPU): {mean_latency*1000:.1f}ms\n")

    lines.append("## Calibration-set threshold sweep\n")
    lines.append("| threshold | precision | recall | f1 | fp | fn |")
    lines.append("|---|---|---|---|---|---|")
    for row in sweep:
        lines.append(
            f"| {row['threshold']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} "
            f"| {row['false_positive']} | {row['false_negative']} |"
        )

    lines.append("\n## Held-out per-pair results\n")
    lines.append("| pair_id | category | label | min_entailment | predicted | correct |")
    lines.append("|---|---|---|---|---|---|")
    for pair, min_entailment in held_out_scored:
        predicted = min_entailment >= threshold
        correct = predicted == pair.nli_positive
        lines.append(
            f"| {pair.pair_id} | {pair.category} | {pair.nli_positive} | {min_entailment:.3f} "
            f"| {predicted} | {'yes' if correct else 'NO'} |"
        )

    report = "\n".join(lines) + "\n"
    print(report)

    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(report, encoding="utf-8")
        print(f"\nWritten to {arguments.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
