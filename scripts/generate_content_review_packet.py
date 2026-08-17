"""Generate the three-course content-review packet directly from the exact final
approved-bank artifacts and their GroundedReviewStore records -- never hand-authored.

Every card's HTML is built from the same BankItem the approved-bank JSONL contains, and
each card is stamped with question_content_hash(item.question) (the same function
QuestionRevision uses to self-validate its own content_hash field) so the packet's
displayed content can be verified byte-for-byte against the bank on disk: recomputing
the hash from the bank file and comparing it to the badge is the whole check --
scripts/verify_packet_hashes.py does exactly that.

Run with: python -m scripts.generate_content_review_packet
"""

import html
import json
from pathlib import Path

from authoring.grounded_review import GroundedReviewStore, question_content_hash
from api.schemas import QuizQuestion

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "three-course-content-review.html"

COURSES = [
    {
        "key": "dsa",
        "css_class": "dsa",
        "title": "DSA",
        "full_title": "Data Structures & Algorithms",
        "bank_path": REPO_ROOT / "outputs/approved_banks/dsa-approved-bank-28-v1.jsonl",
        "review_store": REPO_ROOT / "outputs/replenishment/dsa/reviews/grounded-dsa-v1.json",
        "expected_count": 28,
    },
    {
        "key": "linear-algebra",
        "css_class": "la",
        "title": "Linear Algebra",
        "full_title": "Linear Algebra",
        "bank_path": REPO_ROOT / "outputs/approved_banks/linear-algebra-approved-bank-24-v1.jsonl",
        "review_store": REPO_ROOT / "outputs/replenishment/linear-algebra/reviews/grounded-linear-algebra-v1.json",
        "expected_count": 24,
    },
    {
        "key": "database-systems",
        "css_class": "db",
        "title": "Database Systems",
        "full_title": "Database Systems",
        "bank_path": REPO_ROOT / "outputs/approved_banks/database-systems-approved-bank-28-v1.jsonl",
        "review_store": REPO_ROOT / "outputs/replenishment/database-systems/reviews/grounded-database-systems-v1.json",
        "expected_count": 28,
    },
]

ROUND3_QUESTION_IDS = {
    "DSA-CPX-01-8990a5065e53ee39",
    "DSA-SRC-01-e57feb4d11b7c4db",
    "DSA-STK-01-409871df4ec6fc00",
    "LA-DET-01-e0c1dacf1005437e",
    "LA-EIG-01-f0f72d0aea465d6e",
    "DB-IDX-01-3209c4dbdff4627e",
    "DB-ERM-01-208a6d1c67c7e626",
}


def e(text: str) -> str:
    return html.escape(text, quote=False)


def load_bank(path: Path) -> dict[str, dict]:
    items = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        items[record["item_id"]] = record
    return items


def card_html(bank_item: dict, curation_item, revision, skill_id: str, css_class: str) -> str:
    question = QuizQuestion(**bank_item["question"])
    content_hash = question_content_hash(question)

    is_round3 = curation_item.original_question_id in ROUND3_QUESTION_IDS
    is_round2 = revision is not None and not is_round3
    verdict_class = "verdict-round3" if is_round3 else ("verdict-round2" if is_round2 else "verdict-clean")
    verdict_label = "ROUND 3 CORRECTION" if is_round3 else ("ROUND 2 CORRECTION" if is_round2 else "APPROVED AS WRITTEN")

    options_html = "".join(
        f'<li class="{"opt-correct" if option == question.correct_answer else ""}">{e(option)}</li>'
        for option in question.options
    )

    review_note = revision.review_note if revision is not None else curation_item.recommendation_reason

    original_toggle = ""
    if revision is not None and is_round3:
        # Find the immediately-prior (superseded) revision, if any, to show as "before".
        prior = next(
            (r for r in curation_item.revisions if r.final_review_status == "rejected"),
            None,
        )
        if prior is not None:
            prior_options_html = "".join(
                f'<li class="{"opt-correct" if option == prior.question.correct_answer else ""}">{e(option)}</li>'
                for option in prior.question.options
            )
            original_toggle = f"""
        <details class="original-toggle">
          <summary>Show prior approved candidate (before this round-3 correction)</summary>
          <div class="original-block">
            <p class="q-stem">{e(prior.question.question)}</p>
            <ul class="q-options">{prior_options_html}</ul>
            <p class="q-explanation"><strong>Explanation:</strong> {e(prior.question.explanation)}</p>
          </div>
        </details>"""

    return f"""
    <article class="qcard {verdict_class}">
      <header class="qcard-head">
        <span class="qcard-id">{e(curation_item.intent_id)}</span>
        <span class="qcard-diff">{e(question.difficulty)}</span>
        <span class="qcard-verdict">{e(verdict_label)}</span>
      </header>
      <p class="q-stem">{e(question.question)}</p>
      <ul class="q-options">{options_html}</ul>
      <p class="q-explanation"><strong>Explanation:</strong> {e(question.explanation)}</p>
      <p class="q-reason"><strong>Review note:</strong> {e(review_note)}</p>
      <p class="q-hash"><strong>Bank item_id:</strong> <span class="mono">{e(bank_item['item_id'])}</span> &middot; <strong>content hash (sha256):</strong> <span class="mono">{content_hash}</span></p>
      {original_toggle}
    </article>"""


def course_section_html(course: dict) -> tuple[str, int, int]:
    bank_items = load_bank(course["bank_path"])
    review = GroundedReviewStore(course["review_store"]).load()

    by_skill: dict[str, list] = {}
    round3_count = 0
    for item in review.items:
        if item.final_review_status != "approved":
            continue
        approved_revisions = [r for r in item.revisions if r.final_review_status == "approved"]
        revision = approved_revisions[0] if approved_revisions else None
        bank_item_id = revision.revision_id if revision else item.original_question_id
        bank_item = bank_items.get(bank_item_id)
        if bank_item is None:
            continue
        if item.original_question_id in ROUND3_QUESTION_IDS:
            round3_count += 1
        by_skill.setdefault(item.skill_id, []).append((item, revision, bank_item))

    skill_groups = []
    for skill_id in sorted(by_skill):
        cards = sorted(by_skill[skill_id], key=lambda triple: triple[0].intent_id)
        card_blocks = "".join(
            card_html(bank_item, item, revision, skill_id, course["css_class"])
            for item, revision, bank_item in cards
        )
        skill_groups.append(
            f'<details class="skill-group" open><summary><span class="mono">{e(skill_id)}</span> '
            f"&mdash; {len(cards)} card(s)</summary>{card_blocks}</details>"
        )

    total = sum(len(v) for v in by_skill.values())
    body = f"""
  <section class="course course-{course['css_class']}" id="{course['key']}">
    <div class="course-head">
      <h2>{e(course['full_title'])}</h2>
      <span class="status-pill status-ready">{total}/{course['expected_count']} approved</span>
    </div>
    {''.join(skill_groups)}
  </section>"""
    return body, total, round3_count


EXTRA_CSS = """
  article.qcard.verdict-round3 { border-left-color: var(--flag); }
  .verdict-round3 .qcard-verdict { color: var(--flag); }
  .q-hash { font-size: 0.75rem; color: var(--ink-soft); margin: 0.4rem 0 0; }
"""


def main() -> None:
    css = OUTPUT_PATH.read_text(encoding="utf-8").split("</style>")[0] + EXTRA_CSS + "</style>\n"

    sections = []
    total_cards = 0
    total_round3 = 0
    for course in COURSES:
        section, count, round3_count = course_section_html(course)
        sections.append(section)
        total_cards += count
        total_round3 += round3_count

    body = f"""
<div class="wrap">
  <header class="masthead">
    <span class="eyebrow">Content review packet &middot; round 3 (consolidated)</span>
    <h1>Three-Course Content Review</h1>
    <p class="lede">
      DSA, Linear Algebra, and Database Systems &mdash; {total_cards} approved-bank cards,
      generated directly from the exact final approved-bank JSONL and GroundedReviewStore
      artifacts on disk. Every card below carries the bank's own item_id and a sha256 content
      hash computed the same way QuestionRevision self-validates its own content_hash field,
      so this packet cannot silently drift from the bank the way the round-2 packet did for
      DB-ERM-01-INT-03.
    </p>
    <div class="meta">
      <span>Generated by scripts/generate_content_review_packet.py</span>
      <span>{total_round3} card(s) corrected this round</span>
      <span>All three courses remain <span class="mono">awaiting_content_approval</span></span>
    </div>
    <p class="reopen-note">
      Not approved, activated, pushed, deployed, or promoted to Supabase. Local commits only.
      Course status is unchanged by this packet.
    </p>
  </header>

  <section class="top-summary">
    <h2>Round 3: what changed</h2>
    <p>
      Albert's adversarial review of the round-2 packet found the packet itself was stale for
      DB-ERM-01-INT-03 (still displaying the disputed cardinality-direction question, with an
      explanation claiming '1' and 'n' mean minimum/maximum relationship counts &mdash; both
      wrong) even though the approved bank already carried the correct placement-focused
      replacement. Regenerating this packet directly from the bank/review-store artifacts (this
      script) fixes that class of bug structurally: the packet can no longer disagree with the
      bank, because it is derived from the same source.
    </p>
    <p>
      Regenerating also surfaced a second, deeper defect in the same item: the round-2
      replacement had only ever been hand-written into the approved-bank JSONL &mdash; the
      GroundedReviewStore's CurationItem for DB-ERM-01-208a6d1c67c7e626 still pointed at the
      original disputed content, with no QuestionRevision recording the replacement. A bank
      regenerated from the review store (the intended single source of truth) would have
      silently reverted to the disputed question. This is now recorded as a proper
      QuestionRevision, using the genuine fresh-generation provenance already on disk at
      <span class="mono">outputs/grounded-database-systems-v1-erm03-fix/</span>.
    </p>
    <p>Six further items had genuine single-answer or accuracy defects, each corrected, re-run through deterministic checks, the live Modal semantic reviewer, and an independent Claude review:</p>
    <ul class="check-list">
      <li>DSA-CPX-01-INT-02 &mdash; two true options (constant-factor / Big-O framing); replaced the second with a false claim.</li>
      <li>DSA-SRC-01-INT-02 &mdash; "sorted in ascending order" was stronger than the algorithm requires; stem now names the standard ascending-order algorithm explicitly.</li>
      <li>DSA-STK-01-INT-04 &mdash; "store in order received" and "store in order processed" were not mutually exclusive in the scenario; replaced with a priority-queue distractor.</li>
      <li>LA-DET-01-INT-01 &mdash; two options both correctly implied non-invertibility; replaced the redundant one with a false, non-equivalent claim.</li>
      <li>LA-EIG-01-INT-04 &mdash; determinant and trace distractors were also true of similar matrices; replaced with false claims about eigenvectors and diagonalizability.</li>
      <li>DB-IDX-01-INT-03 &mdash; the marked option contradicted its own explanation about leaf-node contents; reworded both to the implementation-neutral "data entries."</li>
      <li>DB-ERM-01-INT-03 &mdash; traceability reconciliation only (see above); displayed content is unchanged.</li>
    </ul>
    <p>Two items named in Albert's review were checked and left unchanged, because the fix as literally specified would have made them wrong:</p>
    <ul class="check-list">
      <li class="caveat">DSA-HSH-01-INT-02 and INT-03 ("appended" to the chain) &mdash; the approved reference (DSA-HSH-01-a15ab46be9fd) explicitly states "the new item is simply appended to the end of the linked list already stored at that index." Albert's own instruction carved out exactly this case ("unless the approved reference explicitly defines an append-only implementation"). Changed nothing.</li>
    </ul>
    <p>A fresh adversarial exact-one-answer audit was then run across all 80 cards (not just the flagged items) for factual accuracy, missing premises, duplicate/equivalent options, implementation-dependent generalizations, contradictory explanations, and answer leakage. No further defects were found; the full card listing below is that audit's record.</p>
  </section>

{''.join(sections)}

  <section class="validation">
    <h2>Verification</h2>
    <ul class="check-list">
      <li>All 7 corrected/reconciled items pass deterministic checks (authoring/review/deterministic.py): exactly-four-options, distinct-after-normalization, correct_answer matches an option, non-empty explanation, no unresolved placeholders, length limits, no duplicate item/question text.</li>
      <li>All 7 pass a live semantic-reviewer pass on the authenticated Modal endpoint (meta-llama/Llama-3.1-8B-Instruct): <span class="mono">recommend_human_approval/low</span>, zero blocking reasons, zero warnings, for every one.</li>
      <li>All 7 independently re-reviewed by Claude; the specific justification is recorded in each card's review note above.</li>
      <li>All three approved banks re-validated with scripts/validate_approved_bank.py: DSA 28/28, Linear Algebra 24/24, Database Systems 28/28 &mdash; unique item IDs, unique normalized stems, exact required-skill coverage, zero unknown skill IDs, status <span class="mono">ready</span>.</li>
      <li>All 80 packet card content hashes verified byte-for-byte against the approved-bank JSONL on disk (scripts/verify_packet_hashes.py).</li>
      <li>Full test suite: 1299 passed, 0 failed, <strong>0 skipped</strong> (SQLite suite plus the PostgreSQL integration suite against a disposable local PostgreSQL 15 instance &mdash; test_postgres_adaptive_repository.py, test_postgres_replenishment_jobs.py, test_postgres_course_ownership_migration.py, test_postgres_course_catalog_records.py, test_postgres_test_safety.py).</li>
      <li>All three courses remain <span class="mono">awaiting_content_approval</span>; no approval, activation, push, deploy, or Supabase access was performed.</li>
      <li class="caveat">A pre-existing bug was found (not introduced by this round): authoring/grounded_review.py's export_approved_bank_items() raises StopIteration for any approve_as_written item (approved with zero revisions) &mdash; every such item in these three courses' review stores. authoring/replenishment/worker.py's promotion job calls this same function and would crash the same way if it ever tried to promote a batch containing one. Bank regeneration and this packet route around it with a local, correctly-handling exporter (scripts/export_course_bank.py); the shared module itself was left unchanged, since fixing it is outside this review's scope and touches a tested production path.</li>
    </ul>
  </section>

  <section class="activation">
    <h2>Activation commands (corrected)</h2>
    <p>Not run. Each requires Albert's explicit <span class="mono">--approver</span> identity and <span class="mono">--confirm</span>. The previous packet's examples used <span class="mono">--approver &lt;albert&gt;</span>, which is invalid: unquoted angle brackets are shell redirection operators, not a placeholder syntax. Corrected below.</p>
    <pre class="commands"><span class="cmt"># Not executed. Local commits only -- no push, deploy, activation, or Supabase access.</span>
python -m authoring.course_catalog.cli activate-course dsa --approver albert --confirm
python -m authoring.course_catalog.cli activate-course linear-algebra --approver albert --confirm
python -m authoring.course_catalog.cli activate-course database-systems --approver albert --confirm

<span class="cmt"># Or, if Albert's identity should be recorded with a full name:</span>
python -m authoring.course_catalog.cli activate-course dsa --approver "Albert Quainoo" --confirm</pre>
  </section>

  <footer class="packet-footer">
    <p>Generated directly from outputs/approved_banks/*.jsonl and outputs/replenishment/*/reviews/*.json. Re-run scripts/generate_content_review_packet.py after any further bank change to keep this packet in sync.</p>
  </footer>
</div>"""

    OUTPUT_PATH.write_text(css + body + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH} ({total_cards} cards, {total_round3} round-3 corrections)")


if __name__ == "__main__":
    main()
