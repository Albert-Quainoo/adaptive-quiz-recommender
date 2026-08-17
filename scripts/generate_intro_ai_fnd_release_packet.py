"""Consolidated candidate-review packet for the bounded intro-ai foundations
release candidate (batch grounded-ai-fnd-release-v1: AI-FND-03, AI-FND-04).

AI-FND-03 has been approved by Albert Quainoo (genuine human review, recorded via
authoring.grounded_review.approve_revision). AI-FND-04's original candidate was
rejected by Albert for semantically overlapping options; a human revision was
proposed, preserving the original as calibration evidence, and given a fresh full
review pass -- still pending. Nothing here has been promoted to the active bank
or deployed.

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
REVIEW_STORE_DIR = REPO_ROOT / "outputs/replenishment/ai/reviews"
REPORT_DIR = REPO_ROOT / "outputs/replenishment/ai/reviews/automated_review_reports"
BATCH_DIR = REPO_ROOT / "outputs/replenishment/ai/batches"
REFERENCE_DECISIONS_PATH = (
    REPO_ROOT / "outputs/replenishment/ai/reviews/reference_decisions/intro-ai-fnd-release-v1.json"
)


def e(text: str) -> str:
    return html.escape(text, quote=False)


def load_style() -> str:
    text = STYLE_SOURCE.read_text(encoding="utf-8")
    return "<style>" + text.split("<style>", 1)[1].split("</style>", 1)[0] + "</style>\n"


def _checks_html(report: dict) -> str:
    checks = report.get("deterministic_checks", {}).get("checks", [])
    checks_html = "".join(
        f'<li class="{"" if c["passed"] else "caveat"}">{e(c["code"])}: {e(c["message"])}</li>' for c in checks
    )
    passed = sum(1 for c in checks if c["passed"])
    return f"({passed}/{len(checks)} deterministic checks passed)", checks_html


def _question_block(question: dict) -> str:
    options_html = "".join(
        f'<li class="{"opt-correct" if option == question["correct_answer"] else ""}">{e(option)}</li>'
        for option in question["options"]
    )
    return f"""
            <p class="q-stem">{e(question["question"])}</p>
            <ul class="q-options">{options_html}</ul>
            <p class="q-explanation"><strong>Explanation:</strong> {e(question["explanation"])}</p>"""


def candidate_html_03() -> str:
    """AI-FND-03: approved as-is (the existing automated repair), by Albert Quainoo."""
    review = GroundedReviewStore(REVIEW_STORE_DIR / f"{BATCH_ID}__AI-FND-03.json").load()
    reports = json.loads((REPORT_DIR / f"{BATCH_ID}__AI-FND-03.json").read_text(encoding="utf-8"))
    pending = [
        json.loads(line)
        for line in (BATCH_DIR / f"{BATCH_ID}__AI-FND-03" / "pending_questions.jsonl")
        .read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    source = {q["question_id"]: q for q in pending}[review.items[0].original_question_id]

    item = review.items[0]
    approved_revision = next(r for r in item.revisions if r.final_review_status == "approved")
    head_question = approved_revision.question.model_dump(mode="json")
    content_hash = approved_revision.content_hash
    report = next(r for r in reports if r["reviewed_content_hash"] == content_hash)
    summary, checks_html = _checks_html(report)

    original_options = "".join(
        f'<li class="{"opt-correct" if option == source["question"]["correct_answer"] else ""}">{e(option)}</li>'
        for option in source["question"]["options"]
    )

    return f"""
    <article class="qcard verdict-clean">
      <header class="qcard-head">
        <span class="qcard-id">{e(item.intent_id)}</span>
        <span class="qcard-diff">{e(head_question["difficulty"])}</span>
        <span class="qcard-verdict">APPROVED &mdash; reviewer: {e(item.reviewed_by)}</span>
      </header>
      {_question_block(head_question)}
      <p class="q-reason"><strong>References:</strong> <span class="mono">{e(", ".join(approved_revision.reference_ids))}</span></p>
      <p class="q-reason"><strong>Approved revision:</strong> <span class="mono">{e(approved_revision.revision_id)}</span> (automated repair, human-approved as written -- no further edits)</p>
      <details class="original-toggle">
        <summary>Automated review: {e(summary)}</summary>
        <div class="original-block"><ul class="check-list">{checks_html}</ul></div>
      </details>
      <details class="original-toggle">
        <summary>Show pre-repair candidate (1 automated repair applied before Albert's review)</summary>
        <div class="original-block">
          <p class="q-stem">{e(source["question"]["question"])}</p>
          <ul class="q-options">{original_options}</ul>
          <p class="q-explanation"><strong>Explanation:</strong> {e(source["question"]["explanation"])}</p>
        </div>
      </details>
    </article>"""


def candidate_html_04() -> str:
    """AI-FND-04: original preserved as calibration evidence (superseded, not deleted);
    Albert's human revision shown pending with its own fresh review result."""
    review = GroundedReviewStore(REVIEW_STORE_DIR / f"{BATCH_ID}__AI-FND-04.json").load()
    reports = json.loads((REPORT_DIR / f"{BATCH_ID}__AI-FND-04.json").read_text(encoding="utf-8"))
    pending = [
        json.loads(line)
        for line in (BATCH_DIR / f"{BATCH_ID}__AI-FND-04" / "pending_questions.jsonl")
        .read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    item = review.items[0]
    source = {q["question_id"]: q for q in pending}[item.original_question_id]
    original_question = source["question"]

    revision = item.revisions[-1]
    revision_question = revision.question.model_dump(mode="json")
    revision_report = next(r for r in reports if r["reviewed_content_hash"] == revision.content_hash)
    original_report = next(r for r in reports if r["reviewed_content_hash"] != revision.content_hash)
    orig_summary, orig_checks_html = _checks_html(original_report)
    orig_aa = original_report.get("answer_assessment") or {}
    rev_summary, rev_checks_html = _checks_html(revision_report)
    rev_aa = revision_report.get("answer_assessment") or {}

    return f"""
    <article class="qcard verdict-clean">
      <header class="qcard-head">
        <span class="qcard-id">{e(item.intent_id)}</span>
        <span class="qcard-diff">{e(original_question["difficulty"])}</span>
        <span class="qcard-verdict">ORIGINAL &mdash; PRESERVED AS CALIBRATION EVIDENCE (superseded by pending revision)</span>
      </header>
      {_question_block(original_question)}
      <p class="q-reason"><strong>Original automated-review verdict (unchanged):</strong> {e(original_report["recommendation"])} / {e(original_report["risk_level"])} risk</p>
      <p class="q-reason"><strong>Why superseded:</strong> Albert Quainoo (human review): the four options are semantically overlapping -- multiple options describe materially the same proposition in different words, so more than one could be argued correct. The automated reviewer's own answer_assessment shows how it missed this: matches_declared_answer={e(str(orig_aa.get("matches_declared_answer")))}, multiple_defensible_answers={e(str(orig_aa.get("multiple_defensible_answers")))}, duplicate_or_rephrased_distractors={e(str(orig_aa.get("duplicate_or_rephrased_distractors")))}.</p>
      <details class="original-toggle">
        <summary>Original automated review: {e(orig_summary)}</summary>
        <div class="original-block"><ul class="check-list">{orig_checks_html}</ul></div>
      </details>
    </article>
    <article class="qcard verdict-round3">
      <header class="qcard-head">
        <span class="qcard-id">{e(item.intent_id)}</span>
        <span class="qcard-diff">{e(revision_question["difficulty"])}</span>
        <span class="qcard-verdict">HUMAN REVISION &mdash; editor: {e(revision.editor)} &mdash; PENDING (not yet approved)</span>
      </header>
      {_question_block(revision_question)}
      <p class="q-reason"><strong>Revision:</strong> <span class="mono">{e(revision.revision_id)}</span> supersedes original <span class="mono">{e(item.original_question_id)}</span></p>
      <p class="q-reason"><strong>Review note (Albert Quainoo):</strong> {e(revision.review_note)}</p>
      <p class="q-reason"><strong>References (unchanged from original):</strong> <span class="mono">{e(", ".join(revision.reference_ids))}</span></p>
      <p class="q-reason"><strong>Fresh full review of the revision:</strong> {e(revision_report["recommendation"])} / {e(revision_report["risk_level"])} risk. matches_declared_answer={e(str(rev_aa.get("matches_declared_answer")))}, multiple_defensible_answers={e(str(rev_aa.get("multiple_defensible_answers")))}, duplicate_or_rephrased_distractors={e(str(rev_aa.get("duplicate_or_rephrased_distractors")))}, answer_confidence={e(str(rev_aa.get("answer_confidence")))}.</p>
      <p class="fine">The reviewer does not perform genuine per-option independent assessment (AnswerAssessment.option_assessments is a schema field the real reviewer never populates -- authoring/review/response_parser.py hardcodes it to {{}}). "Assess each option independently" maps to this single holistic multiple_defensible_answers judgment, not a real per-option breakdown -- reported here honestly rather than fabricated.</p>
      <details class="original-toggle">
        <summary>Revision automated review: {e(rev_summary)}</summary>
        <div class="original-block"><ul class="check-list">{rev_checks_html}</ul></div>
      </details>
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

    cards = candidate_html_03() + candidate_html_04()

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
      closed on insufficient reference support and was dropped. Albert Quainoo has now personally
      reviewed both candidates: AI-FND-03 is approved as-is; AI-FND-04's original candidate was
      rejected for semantically overlapping options and revised.
    </p>
    <div class="meta">
      <span>Generated by scripts/generate_intro_ai_fnd_release_packet.py</span>
      <span>AI-FND-03: approved by Albert Quainoo</span>
      <span>AI-FND-04: human-revised, pending re-approval</span>
      <span>No bank promotion, deployment, or cron change performed</span>
    </div>
    <p class="reopen-note">
      AI-FND-03 has been approved by Albert Quainoo (recorded via
      <span class="mono">authoring.grounded_review.approve_revision</span>) but not promoted to the
      active bank, deployed, or pushed to main. AI-FND-04's revision is still <strong>pending</strong>
      -- it has a fresh full review result below but has not been approved. Reference decisions were
      made as delegated review by Claude Code on behalf of Albert Quainoo; the reviewed blueprint
      intents were authored by Claude Code under explicit delegated authorization for this bounded
      release candidate only.
    </p>
  </header>

  <section class="top-summary">
    <h2>Scope and disposition</h2>
    <ul class="check-list">
      <li><strong>AI-FND-02</strong> &mdash; failed closed: only one in-domain reference candidate found (below the 2-4 minimum). No blueprint intent authored; not generated.</li>
      <li><strong>AI-FND-03</strong> &mdash; AI-FND-03-INT-01, one automated repair applied, then <strong>approved as-is by Albert Quainoo</strong> (genuine human review, not delegated).</li>
      <li><strong>AI-FND-04</strong> &mdash; AI-FND-04-INT-02 passed automated review cleanly, but Albert Quainoo rejected it as written: its four options were semantically overlapping (multiple options stated the same proposition in different words). A human revision was created, preserving the original as calibration evidence, and given a fresh full review pass (see card below). Still pending.</li>
      <li><strong>Reviewer regression finding</strong> &mdash; the live automated reviewer did not itself catch the AI-FND-04 semantic-overlap issue (multiple_defensible_answers=False, duplicate_or_rephrased_distractors=[] on the original). Captured as a strict-xfail regression fixture: <span class="mono">tests/test_review_ai_fnd_04_semantic_overlap_regression.py</span>. Not a scoring-logic bug (authoring/review/risk.py already blocks correctly on both fields when populated) -- the reviewer model itself missed it on this pass. No narrow code fix was available; not attempted.</li>
    </ul>
    <h2>Reference decisions (delegated review by Claude Code on behalf of Albert Quainoo)</h2>
    {reference_decisions_html()}
    <p class="fine">Full passages, URLs, retrieval dates, and content hashes for every candidate (accepted and rejected/insufficient) are in outputs/replenishment/ai/reviews/reference_decisions/intro-ai-fnd-release-v1.json.</p>
  </section>

  <section class="course course-ai" id="intro-ai-fnd">
    <div class="course-head">
      <h2>Candidates</h2>
      <span class="status-pill status-ready">1 approved, 1 pending (revised)</span>
    </div>
    {cards}
  </section>

  <section class="validation">
    <h2>Budget and provenance (unchanged ledger, appended)</h2>
    <ul class="check-list">
      <li>Original generation run: 2 Modal generation calls, 3 review calls, 1 repair -- 6 calls total (unchanged from the prior packet).</li>
      <li>This review round added exactly 1 Modal review call: the fresh full review of the AI-FND-04 human revision. No generation or repair calls were spent (Albert's edit did not go through automated generation/repair).</li>
      <li>Running total: 7 Modal calls, against a cap of 20.</li>
      <li>Estimated cost: still well under the USD 5.00 cap (7 short instruct-model calls total; no per-call billing figure is available to this run, so no dollar total is asserted).</li>
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
