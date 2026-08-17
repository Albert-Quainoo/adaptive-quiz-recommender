"""Course-neutral labeled option-pair dataset for calibrating and evaluating the hybrid
option-equivalence gate's NLI detector (authoring/review/equivalence_nli.py).

Every pair is hand-authored, synthetic calibration content (not real learner-facing
bank items) spanning four domains (AI, DSA, Linear Algebra, Database Systems) and seven
categories:

- paraphrase: two options restate the same claim in different words -- the case the
  NLI detector exists to catch. nli_positive=True.
- math_equivalence / unit_equivalence: numeric/unit-notation equivalence -- caught by
  the symbolic-math and unit-conversion detectors, not NLI. Included here so the NLI
  threshold is calibrated *against* these as negatives (NLI is not expected or required
  to flag them; the overall gate still catches them through the other two detectors),
  not because NLI needs to succeed on them.
- negation: a "yes/no" structural mirror that is NOT semantically equivalent (a
  contradiction, not entailment) -- must not falsely trigger NLI equivalence.
- related_distractor: topically/lexically close but genuinely different claims.
- causal_reversal: swaps cause and effect -- superficially similar wording, opposite
  claim.
- scope_quantifier_change: "all" vs "some"/"most", or an added/removed qualifier that
  changes the claim's truth conditions.

nli_positive is the label the NLI-threshold calibration script
(scripts/calibrate_equivalence_nli_threshold.py) trains/evaluates against: True only
for "paraphrase" pairs, False for every other category (including the math/unit pairs,
per above).

split is a fixed, hand-assigned calibration/held-out partition -- not randomized at
runtime -- so the same held-out set is evaluated every time and the threshold is never
implicitly re-derived from it.
"""

from typing import Literal

from pydantic import BaseModel, Field

Domain = Literal["ai", "dsa", "linear-algebra", "database-systems"]
Category = Literal[
    "paraphrase",
    "math_equivalence",
    "unit_equivalence",
    "negation",
    "related_distractor",
    "causal_reversal",
    "scope_quantifier_change",
]
Split = Literal["calibration", "held_out"]


class LabeledPair(BaseModel):
    pair_id: str = Field(min_length=1)
    domain: Domain
    category: Category
    stem: str = Field(min_length=1)
    option_a: str = Field(min_length=1)
    option_b: str = Field(min_length=1)
    nli_positive: bool
    split: Split


LABELED_PAIRS: list[LabeledPair] = [
    # --- AI ---------------------------------------------------------------------
    LabeledPair(
        pair_id="ai-paraphrase-1", domain="ai", category="paraphrase", split="calibration",
        stem="What does the Chinese Room argument challenge?",
        option_a="Whether correct symbol manipulation alone proves genuine understanding.",
        option_b="Whether producing the right output through rule-following demonstrates real comprehension.",
        nli_positive=True,
    ),
    LabeledPair(
        pair_id="ai-paraphrase-2", domain="ai", category="paraphrase", split="held_out",
        stem="What is the primary purpose of a firewall?",
        option_a="To block unauthorized traffic from entering or leaving a network.",
        option_b="To prevent unapproved connections from reaching or exiting a protected network.",
        nli_positive=True,
    ),
    LabeledPair(
        pair_id="ai-math-1", domain="ai", category="math_equivalence", split="calibration",
        stem="A model's accuracy on a held-out set is reported as a fraction.",
        option_a="0.9",
        option_b="9/10",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="ai-unit-1", domain="ai", category="unit_equivalence", split="held_out",
        stem="A training run's checkpoint file size is reported.",
        option_a="1.5 gigabytes",
        option_b="1500 megabytes",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="ai-negation-1", domain="ai", category="negation", split="calibration",
        stem="Does a greedy best-first search guarantee the shortest path?",
        option_a="No, it does not guarantee the shortest path because it ignores accumulated cost.",
        option_b="Yes, it always finds the shortest path because it expands the most promising node first.",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="ai-related-distractor-1", domain="ai", category="related_distractor", split="held_out",
        stem="Which AI subfield is exemplified by a spam filter reading email text?",
        option_a="Natural language processing",
        option_b="Computer vision",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="ai-causal-reversal-1", domain="ai", category="causal_reversal", split="calibration",
        stem="Why does overfitting reduce test accuracy?",
        option_a="Because the model memorizes noise in the training data, which does not generalize.",
        option_b="Because low test accuracy causes the model to memorize noise in the training data.",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="ai-scope-1", domain="ai", category="scope_quantifier_change", split="held_out",
        stem="Do all supervised learning algorithms require labeled data?",
        option_a="All supervised learning algorithms require labeled data.",
        option_b="Most supervised learning algorithms require labeled data.",
        nli_positive=False,
    ),

    # --- DSA ---------------------------------------------------------------------
    LabeledPair(
        pair_id="dsa-paraphrase-1", domain="dsa", category="paraphrase", split="calibration",
        stem="What property must a stack's operations preserve?",
        option_a="The last element pushed must be the first element popped.",
        option_b="The most recently added item is always removed before any earlier item.",
        nli_positive=True,
    ),
    LabeledPair(
        pair_id="dsa-paraphrase-2", domain="dsa", category="paraphrase", split="held_out",
        stem="What does a binary search require of its input array?",
        option_a="The array must be sorted before binary search can be applied.",
        option_b="Binary search only works correctly on an array that is already in sorted order.",
        nli_positive=True,
    ),
    LabeledPair(
        pair_id="dsa-math-1", domain="dsa", category="math_equivalence", split="held_out",
        stem="A binary search over 1024 sorted elements takes at most how many comparisons?",
        option_a="10",
        option_b="log2(1024)",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="dsa-unit-1", domain="dsa", category="unit_equivalence", split="calibration",
        stem="An algorithm's measured runtime on a benchmark input is reported.",
        option_a="2000 milliseconds",
        option_b="2 seconds",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="dsa-negation-1", domain="dsa", category="negation", split="held_out",
        stem="Is a hash table's average-case lookup time linear in the number of elements?",
        option_a="No, average-case lookup is O(1), not linear in the number of elements.",
        option_b="Yes, average-case lookup grows linearly with the number of elements stored.",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="dsa-related-distractor-1", domain="dsa", category="related_distractor", split="calibration",
        stem="Which traversal visits a binary tree's root between its left and right subtrees?",
        option_a="In-order traversal",
        option_b="Pre-order traversal",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="dsa-causal-reversal-1", domain="dsa", category="causal_reversal", split="held_out",
        stem="Why does an unbalanced binary search tree degrade to O(n) lookup?",
        option_a="Because skewed insertions make the tree behave like a linked list.",
        option_b="Because O(n) lookup time causes the tree's insertions to become skewed.",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="dsa-scope-1", domain="dsa", category="scope_quantifier_change", split="calibration",
        stem="Do all comparison-based sorting algorithms run in O(n log n) time?",
        option_a="All comparison-based sorting algorithms run in O(n log n) time or better.",
        option_b="Some comparison-based sorting algorithms run in O(n log n) time or better.",
        nli_positive=False,
    ),

    # --- Linear Algebra ------------------------------------------------------------
    LabeledPair(
        pair_id="la-paraphrase-1", domain="linear-algebra", category="paraphrase", split="calibration",
        stem="What does it mean for a square matrix to be invertible?",
        option_a="There exists another matrix that, multiplied with it, yields the identity matrix.",
        option_b="A matrix exists such that multiplying the two together produces the identity matrix.",
        nli_positive=True,
    ),
    LabeledPair(
        pair_id="la-paraphrase-2", domain="linear-algebra", category="paraphrase", split="held_out",
        stem="What does a zero determinant indicate about a square matrix?",
        option_a="The matrix is singular and has no inverse.",
        option_b="The matrix is not invertible because it is singular.",
        nli_positive=True,
    ),
    LabeledPair(
        pair_id="la-math-1", domain="linear-algebra", category="math_equivalence", split="held_out",
        stem="The determinant of a 2x2 matrix [[2,0],[0,3]] is computed.",
        option_a="6",
        option_b="2*3",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="la-unit-1", domain="linear-algebra", category="unit_equivalence", split="calibration",
        stem="A vector's magnitude, measured in a physics application, is reported.",
        option_a="0.5 meters",
        option_b="50 centimeters",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="la-negation-1", domain="linear-algebra", category="negation", split="held_out",
        stem="Are the columns of every square matrix linearly independent?",
        option_a="No, a square matrix's columns are linearly independent only if its determinant is nonzero.",
        option_b="Yes, every square matrix has linearly independent columns regardless of its determinant.",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="la-related-distractor-1", domain="linear-algebra", category="related_distractor", split="calibration",
        stem="What does an eigenvector of a matrix represent?",
        option_a="A direction the matrix only scales, without rotating.",
        option_b="A direction the matrix rotates by exactly ninety degrees.",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="la-causal-reversal-1", domain="linear-algebra", category="causal_reversal", split="held_out",
        stem="Why is a matrix with two identical rows singular?",
        option_a="Because identical rows make the rows linearly dependent, forcing determinant zero.",
        option_b="Because a zero determinant causes two of the matrix's rows to become identical.",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="la-scope-1", domain="linear-algebra", category="scope_quantifier_change", split="calibration",
        stem="Do all linear transformations preserve the origin?",
        option_a="Every linear transformation maps the origin to itself.",
        option_b="Most linear transformations map the origin to itself.",
        nli_positive=False,
    ),

    # --- Database Systems ------------------------------------------------------------
    LabeledPair(
        pair_id="db-paraphrase-1", domain="database-systems", category="paraphrase", split="calibration",
        stem="What does a database index primarily improve?",
        option_a="It speeds up lookups by avoiding a full table scan.",
        option_b="It makes queries faster by letting the engine skip scanning every row.",
        nli_positive=True,
    ),
    LabeledPair(
        pair_id="db-paraphrase-2", domain="database-systems", category="paraphrase", split="held_out",
        stem="What guarantee does a database transaction's atomicity provide?",
        option_a="All of the transaction's operations complete, or none of them do.",
        option_b="Either every operation in the transaction succeeds, or the whole transaction is rolled back.",
        nli_positive=True,
    ),
    LabeledPair(
        pair_id="db-math-1", domain="database-systems", category="math_equivalence", split="held_out",
        stem="A table's storage footprint is measured after compaction.",
        option_a="0.25",
        option_b="1/4",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="db-unit-1", domain="database-systems", category="unit_equivalence", split="calibration",
        stem="A query's observed latency is reported from a benchmark run.",
        option_a="0.25 seconds",
        option_b="250 milliseconds",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="db-negation-1", domain="database-systems", category="negation", split="held_out",
        stem="Does a foreign key constraint allow a referencing row to point at a nonexistent primary key?",
        option_a="No, a foreign key constraint prevents referencing a primary key that does not exist.",
        option_b="Yes, a foreign key constraint permits referencing any primary key value, existing or not.",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="db-related-distractor-1", domain="database-systems", category="related_distractor", split="calibration",
        stem="Which normal form eliminates partial dependency on a composite primary key?",
        option_a="Second normal form",
        option_b="Third normal form",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="db-causal-reversal-1", domain="database-systems", category="causal_reversal", split="held_out",
        stem="Why does an unindexed WHERE clause on a large table slow a query?",
        option_a="Because the engine must scan every row to find matches, without an index to narrow the search.",
        option_b="Because slow queries cause the engine to remove the index from the table.",
        nli_positive=False,
    ),
    LabeledPair(
        pair_id="db-scope-1", domain="database-systems", category="scope_quantifier_change", split="calibration",
        stem="Do all NoSQL databases sacrifice strong consistency for availability?",
        option_a="All NoSQL databases sacrifice strong consistency for availability.",
        option_b="Some NoSQL databases sacrifice strong consistency for availability.",
        nli_positive=False,
    ),
]


def by_split(split: Split) -> list[LabeledPair]:
    return [pair for pair in LABELED_PAIRS if pair.split == split]
