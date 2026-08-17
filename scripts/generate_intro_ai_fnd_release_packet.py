"""Consolidated candidate-review packet for the bounded intro-ai foundations
release candidate (batch grounded-ai-fnd-release-v1: AI-FND-03, AI-FND-04).

Every candidate below is still PENDING human review -- this packet is a
review aid for Albert's own approval decision, not an approval record.
Nothing here has been approved, promoted to the active bank, or deployed.

Run with: python -m scripts.generate_intro_ai_fnd_release_packet
"""

import html
import json
from pathlib import Path

from authoring.grounded_review import GroundedReviewStore

REPO_ROOT = Path(__file__).resolve().parent.parent
STYLE_SOURCE = REPO_ROOT / "outputs/review_packets/intro-ai-blueprint-queue-v3.html"
OUTPUT_PATH = REPO_ROOT / "outputs/review_packets/intro-ai-fnd-release-v1.html"

BATCH_ID = "grounded-ai-fnd-release-v1"
SKILLS = ["AI-FND-03", "AI-FND-04"]
REVIEW_STORE_DIR = REPO_ROOT / "outputs/replenishment/ai/reviews"
REPORT_DIR = REPO_ROOT / "outputs/replenishment/ai/reviews/automated_review_reports"
BATCH_DIR = REPO_ROOT / "outputs/replenishment/ai/batches"
REFERENCE_DECISIONS_PATH = (
    REPO_ROOT / "outputs/replenishment/ai/reviews/reference_decisions/intro-ai-fnd-release-v1.json"
)
BLUEPRINT_PATH = REPO_ROOT / "authoring/blueprints/grounded-ai-fnd-release-v1.json"


def e(text: str) -> str:
    return html.escape(text, quote=False)


def load_style() -> str:
    text = STYLE_SOURCE.read_text(encoding="utf-8")
    return "<style>" + text.split("<style>", 1)[1].split("</style>", 1)[0] + "</style>\n"


def candidate_html(skill_id: str) -> str:
    review = GroundedReviewStore(REVIEW_STORE_DIR / f"{BATCH_ID}__{skill_id}.json").load()
    reports = json.loads((REPORT_DIR / f"{BATCH_ID}__{skill_id}.json").read_text(encoding="utf-8"))
    pending = [
        json.loads(line)
        for line in (BATCH_DIR / f"{BATCH_ID}__{skill_id}" / "pending_questions.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    source_by_id = {q["question_id"]: q for q in pending}

    item = review.items[0]
    revisions = item.revisions
    if revisions:
        head_question = revisions[-1].question.model_dump(mode="json")
        reference_ids = revisions[-1].reference_ids
    else:
        head_question = source_by_id[item.original_question_id]["question"]
        reference_ids = source_by_id[item.original_question_id]["reference_ids"]

    options_html = "".join(
        f'<li class="{"opt-correct" if option == head_question["correct_answer"] else ""}">{e(option)}</li>'
        for option in head_question["options"]
    )

    repair_html = ""
    if revisions:
        original = source_by_id[item.original_question_id]["question"]
        original_options = "".join(
            f'<li class="{"opt-correct" if option == original["correct_answer"] else ""}">{e(option)}</li>'
            for option in original["options"]
        )
        first_report = next((r for r in reports if r.get("recommendation") == "reject"), reports[0])
        repair_html = f"""
        <details class="original-toggle">
          <summary>Show pre-repair candidate (1 automated repair applied)</summary>
          <div class="original-block">
            <p class="q-stem">{e(original["question"])}</p>
            <ul class="q-options">{original_options}</ul>
            <p class="q-explanation"><strong>Explanation:</strong> {e(original["explanation"])}</p>
            <p class="q-reason"><strong>Why it was repaired:</strong> {e(", ".join(first_report.get("blocking_reasons", [])) or "see review report")}</p>
          </div>
        </details>"""

    last_report = reports[-1]
    checks = last_report.get("deterministic_checks", {}).get("checks", [])
    checks_html = "".join(
        f'<li class="{"" if c["passed"] else "caveat"}">{e(c["code"])}: {e(c["message"])}</li>' for c in checks
    )

    return f"""
    <article class="qcard verdict-clean">
      <header class="qcard-head">
        <span class="qcard-id">{e(item.intent_id)}</span>
        <span class="qcard-diff">{e(head_question["difficulty"])}</span>
        <span class="qcard-verdict">{e(last_report.get("recommendation", "pending").upper())} &mdash; AWAITING ALBERT'S APPROVAL</span>
      </header>
      <p class="q-stem">{e(head_question["question"])}</p>
      <ul class="q-options">{options_html}</ul>
      <p class="q-explanation"><strong>Explanation:</strong> {e(head_question["explanation"])}</p>
      <p class="q-reason"><strong>References:</strong> <span class="mono">{e(", ".join(reference_ids))}</span></p>
      <details class="original-toggle">
        <summary>Automated review: deterministic checks ({sum(1 for c in checks if c["passed"])}/{len(checks)} passed)</summary>
        <div class="original-block"><ul class="check-list">{checks_html}</ul></div>
      </details>
      {repair_html}
    </article>"""


def reference_decisions_html() -> str:
    data = json.loads(REFERENCE_DECISIONS_PATH.read_text(encoding="utf-8"))
    rows = []
    for skill in data["skills"]:
        outcome = skill["outcome"]
        rows.append(
            f'<li><span class="mono">{e(skill["skill_id"])}</span> &mdash; '
            f'<strong>{e(outcome)}</strong>: {e(skill["reason"])} '
            f'({len(skill["candidates_considered"])} candidate(s) considered)</li>'
        )
    return "<ul class=\"check-list\">" + "".join(rows) + "</ul>"


def main() -> None:
    style = load_style()
    blueprint = json.loads(BLUEPRINT_PATH.read_text(encoding="utf-8"))

    cards = "".join(candidate_html(skill_id) for skill_id in SKILLS)

    body = f"""
<div class="wrap">
  <header class="masthead">
    <span class="eyebrow">Candidate review packet &middot; bounded release candidate</span>
    <h1>Intro AI Foundations &mdash; Release Candidate</h1>
    <p class="lede">
      Batch <span class="mono">{e(BATCH_ID)}</span>, materialized from the approved packet
      <span class="mono">outputs/review_packets/intro-ai-blueprint-queue-v3.html</span>
      (source sha256 <span class="mono">e3f58853fe23</span>&hellip;), scoped to the first three
      deficient skills in deterministic order (AI-FND-02, AI-FND-03, AI-FND-04). AI-FND-02 failed
      closed on insufficient reference support and was dropped; AI-FND-03 and AI-FND-04 each
      produced one generated, reviewed candidate below.
    </p>
    <div class="meta">
      <span>Generated by scripts/generate_intro_ai_fnd_release_packet.py</span>
      <span>2 candidates, both awaiting human question-review approval</span>
      <span>No approval, bank promotion, deployment, or cron change performed</span>
    </div>
    <p class="reopen-note">
      Every candidate below is <strong>pending</strong>. Nothing here has been approved, promoted
      to the active bank, deployed, or pushed to main. Reference decisions were made as delegated
      review by Claude Code on behalf of Albert Quainoo; the reviewed blueprint intents were
      authored by Claude Code under explicit delegated authorization for this bounded release
      candidate only.
    </p>
  </header>

  <section class="top-summary">
    <h2>Scope and disposition</h2>
    <ul class="check-list">
      <li><strong>AI-FND-02</strong> &mdash; failed closed: only one in-domain reference candidate found (below the 2-4 minimum). No blueprint intent authored; not generated.</li>
      <li><strong>AI-FND-03</strong> &mdash; AI-FND-03-INT-01 generated, one automated repair applied (reviewer disagreed with the declared answer), repaired candidate recommended for human approval.</li>
      <li><strong>AI-FND-04</strong> &mdash; AI-FND-04-INT-02 generated with the approved amendment applied ("test or argue", not "test or demonstrate"), passed automated review cleanly on the first attempt, recommended for human approval.</li>
    </ul>
    <h2>Reference decisions (delegated review by Claude Code on behalf of Albert Quainoo)</h2>
    {reference_decisions_html()}
    <p class="fine">Full passages, URLs, retrieval dates, and content hashes for every candidate (accepted and rejected/insufficient) are in outputs/replenishment/ai/reviews/reference_decisions/intro-ai-fnd-release-v1.json.</p>
  </section>

  <section class="course course-ai" id="intro-ai-fnd">
    <div class="course-head">
      <h2>Generated candidates</h2>
      <span class="status-pill status-ready">2 pending</span>
    </div>
    {cards}
  </section>

  <section class="validation">
    <h2>Budget and provenance</h2>
    <ul class="check-list">
      <li>Modal generation calls: 2 (1 per candidate, both accepted on first attempt).</li>
      <li>Modal review calls: 3 (AI-FND-03: 2, across its 1 repair cycle; AI-FND-04: 1).</li>
      <li>Modal repair (automated_revision) calls: 1, spent on AI-FND-03 (the cap: at most one repair per candidate).</li>
      <li>Total Modal calls: 6, against a cap of 20.</li>
      <li>Estimated cost: well under the USD 5.00 cap (6 short instruct-model calls on small prompts/completions; no per-call billing figure was available to this run, so no dollar total is asserted -- see the final report for this caveat).</li>
      <li>Blueprint: authoring/blueprints/grounded-ai-fnd-release-v1.json, validated by authoring/blueprint_validation.py (course-neutral invariants; regression tests in tests/test_blueprint_validation.py).</li>
      <li>Provenance: authoring/blueprints/batch_statuses.json['{e(BATCH_ID)}'].</li>
    </ul>
  </section>

  <footer class="packet-footer">
    <p>Generated from outputs/replenishment/ai/reviews/{e(BATCH_ID)}__*.json, outputs/replenishment/ai/reviews/automated_review_reports/{e(BATCH_ID)}__*.json, and outputs/replenishment/ai/batches/{e(BATCH_ID)}__*/pending_questions.jsonl. Re-run scripts/generate_intro_ai_fnd_release_packet.py after any further review action to keep this packet in sync.</p>
  </footer>
</div>"""

    OUTPUT_PATH.write_text(style + body + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
