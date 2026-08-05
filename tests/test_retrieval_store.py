"""The store and the CLI: what survives between one run and the next.

Review happens across separate invocations, so the tests here are written the
way the workflow is used - one run writes, a later run reads what the first
one left, and a decision made in between is still there afterwards.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from authoring.retrieval.brave import API_KEY_VARIABLE
from authoring.retrieval.cli import main
from authoring.retrieval.models import SearchResult, approve, new_candidate, reject
from authoring.retrieval.search import FetchedPage
from authoring.retrieval.store import CandidateStore, StoreError

REVIEWER = "albert"
FIXED_TIME = datetime(2026, 8, 5, 12, 0, tzinfo=timezone.utc)
REVIEWED_TIME = datetime(2026, 8, 6, 9, 30, tzinfo=timezone.utc)

REFERENCES_CSV = Path("taxonomy/data/ai/references.csv")


def candidate(passage: str, skill_id: str = "AI-SRC-08"):
    return new_candidate(
        skill_id=skill_id,
        title="Heuristics",
        source_url="https://aima.cs.berkeley.edu/heuristics.html",
        source_domain="aima.cs.berkeley.edu",
        passage=passage,
        retrieved_at=FIXED_TIME,
    )


@pytest.fixture
def store(tmp_path) -> CandidateStore:
    return CandidateStore(tmp_path / "candidates.json")


# Persistence


def test_an_empty_store_reads_as_empty(store):
    assert store.load() == []


def test_a_candidate_survives_a_separate_run(store):
    store.add([candidate("A heuristic estimates remaining cost.")])

    reopened = CandidateStore(store.path).load()

    assert len(reopened) == 1
    assert reopened[0].retrieved_at == FIXED_TIME
    assert reopened[0].review_status == "pending"


def test_a_decision_survives_a_separate_run(store):
    store.add([candidate("A heuristic estimates remaining cost.")])
    held = store.load()[0]

    store.replace(approve(held, REVIEWER, note="Good.", reviewed_at=REVIEWED_TIME))

    reopened = CandidateStore(store.path).get(held.candidate_id)

    assert reopened.review_status == "approved"
    assert reopened.reviewer_id == REVIEWER
    assert reopened.review_note == "Good."
    assert reopened.reviewed_at == REVIEWED_TIME


def test_retrieving_again_does_not_undo_a_decision(store):
    retrieved = candidate("A heuristic estimates remaining cost.")
    store.add([retrieved])
    store.replace(approve(store.load()[0], REVIEWER, reviewed_at=REVIEWED_TIME))

    added = store.add([retrieved])

    assert added == []
    assert store.load()[0].review_status == "approved"


def test_the_file_is_deterministic(store, tmp_path):
    first = candidate("A heuristic estimates remaining cost.")
    second = candidate("A state space is the set of reachable states.", "AI-SRC-02")

    store.save([first, second])
    forwards = store.path.read_text()

    store.save([second, first])

    assert store.path.read_text() == forwards
    assert json.loads(forwards)[0]["candidate_id"] < json.loads(forwards)[1]["candidate_id"]


def test_an_unknown_candidate_id_is_an_error(store):
    with pytest.raises(StoreError, match="is not in"):
        store.get("AI-SRC-08-000000000000")


def test_candidates_can_be_read_by_status(store):
    store.add(
        [
            candidate("A heuristic estimates remaining cost."),
            candidate("A heuristic is a guess."),
        ]
    )
    store.replace(reject(store.load()[0], REVIEWER, reviewed_at=REVIEWED_TIME))

    assert len(store.with_status("pending")) == 1
    assert len(store.with_status("rejected")) == 1
    assert store.with_status("approved") == []


# The CLI


def test_a_dry_run_reaches_no_network_and_writes_nothing(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(API_KEY_VARIABLE, raising=False)
    store_path = tmp_path / "candidates.json"

    assert main(["--store", str(store_path), "retrieve", "--dry-run"]) == 0

    output = capsys.readouterr().out

    assert "Dry run" in output
    assert "AI-SRC-01" in output and "AI-SRC-02" in output and "AI-SRC-08" in output
    assert not store_path.exists()


def test_retrieval_without_a_key_stops_with_a_named_variable(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(API_KEY_VARIABLE, raising=False)

    code = main(["--store", str(tmp_path / "candidates.json"), "retrieve"])

    assert code == 2
    assert API_KEY_VARIABLE in capsys.readouterr().err


def test_listing_shows_status_and_filters_by_it(store, capsys):
    store.add(
        [
            candidate("A heuristic estimates remaining cost."),
            candidate("A heuristic is a guess."),
        ]
    )
    store.replace(approve(store.load()[0], REVIEWER, reviewed_at=REVIEWED_TIME))

    main(["--store", str(store.path), "list"])
    listed = capsys.readouterr().out

    assert "2 candidate(s)" in listed
    assert "[approved by albert]" in listed
    assert "[pending]" in listed

    main(["--store", str(store.path), "list", "--status", "pending"])

    assert "1 candidate(s)" in capsys.readouterr().out


def test_approving_by_id_records_the_reviewer(store, capsys):
    store.add([candidate("A heuristic estimates remaining cost.")])
    candidate_id = store.load()[0].candidate_id

    code = main(
        [
            "--store",
            str(store.path),
            "approve",
            candidate_id,
            "--reviewer",
            REVIEWER,
            "--note",
            "Matches the objective.",
        ]
    )

    decided = store.get(candidate_id)

    assert code == 0
    assert decided.review_status == "approved"
    assert decided.reviewer_id == REVIEWER
    assert decided.review_note == "Matches the objective."
    assert "[approved by albert]" in capsys.readouterr().out


def test_rejecting_by_id_records_the_reviewer(store):
    store.add([candidate("A heuristic is a guess.")])
    candidate_id = store.load()[0].candidate_id

    main(["--store", str(store.path), "reject", candidate_id, "--reviewer", REVIEWER])

    assert store.get(candidate_id).review_status == "rejected"


def test_deciding_an_unknown_id_fails_clearly(store, capsys):
    code = main(
        ["--store", str(store.path), "approve", "nope", "--reviewer", REVIEWER]
    )

    assert code == 2
    assert "not in" in capsys.readouterr().err


def test_a_decision_needs_a_reviewer_on_the_command_line(store):
    store.add([candidate("A heuristic estimates remaining cost.")])
    candidate_id = store.load()[0].candidate_id

    with pytest.raises(SystemExit):
        main(["--store", str(store.path), "approve", candidate_id])


def test_export_prints_approved_rows_only(store, capsys):
    keeper = candidate("A heuristic estimates remaining cost.")
    store.add([keeper, candidate("A heuristic is a guess.")])
    store.replace(approve(keeper, REVIEWER, reviewed_at=REVIEWED_TIME))

    assert main(["--store", str(store.path), "export"]) == 0

    exported = capsys.readouterr().out
    rows = exported.splitlines()

    assert rows[0] == "skill_id,reference_material"
    assert len(rows) == 2
    assert "A heuristic is a guess." not in exported


def test_export_of_nothing_approved_says_so(store, capsys):
    store.add([candidate("A heuristic estimates remaining cost.")])

    assert main(["--store", str(store.path), "export"]) == 1
    assert "No approved candidates" in capsys.readouterr().err


def test_no_command_ever_writes_references_csv(store, capsys):
    before = REFERENCES_CSV.read_bytes()
    store.add([candidate("A heuristic estimates remaining cost.")])
    candidate_id = store.load()[0].candidate_id

    main(["--store", str(store.path), "retrieve", "--dry-run"])
    main(["--store", str(store.path), "approve", candidate_id, "--reviewer", REVIEWER])
    main(["--store", str(store.path), "export"])
    main(["--store", str(store.path), "list"])

    assert REFERENCES_CSV.read_bytes() == before


# What the CLI says when a run produces nothing

PAGE_TEXT = (
    "A heuristic function estimates the cost of the cheapest path from a given "
    "state to a goal state. It lets an informed search order the frontier by "
    "how promising a state looks rather than by how far it already is, which "
    "is what separates informed search from uninformed search."
)


class StubProvider:
    """Stands in for Brave inside the CLI, with no key and no network."""

    def __init__(self, url: str):
        self.url = url

    @classmethod
    def bound_to(cls, url: str):
        class Bound:
            @staticmethod
            def from_environment():
                return cls(url)

        return Bound

    def search(self, schedule, diagnostics, budget):
        for step in schedule:
            if not budget.may_request():
                return

            budget.spend_request()
            diagnostics.record_query(step.domain)

            yield step, SearchResult(title="Heuristics", url=self.url, snippet="")


class StubFetcher:
    def __init__(self, *arguments, **keywords):
        pass

    def fetch(self, url: str) -> FetchedPage:
        return FetchedPage(url=url, text=PAGE_TEXT)


def run_retrieve(monkeypatch, tmp_path, url: str, limit: str = "5") -> int:
    monkeypatch.setattr("authoring.retrieval.cli.BraveSearchProvider",
                        StubProvider.bound_to(url))
    monkeypatch.setattr("authoring.retrieval.cli.HttpPageFetcher", StubFetcher)

    return main([
        "--store", str(tmp_path / "candidates.json"), "retrieve", "--limit", limit
    ])


def test_a_run_that_creates_nothing_says_where_the_results_went(
    monkeypatch, tmp_path, capsys
):
    code = run_retrieve(monkeypatch, tmp_path, "https://example.com/heuristics")
    printed = capsys.readouterr().out

    assert code == 1
    assert "Retrieval diagnostics" in printed
    assert "rejected by allowlist" in printed
    assert "Every result was off the allowlist" in printed


def test_a_working_run_reports_its_counts_and_succeeds(monkeypatch, tmp_path, capsys):
    code = run_retrieve(
        monkeypatch, tmp_path, "https://aima.cs.berkeley.edu/heuristics.html"
    )
    printed = " ".join(capsys.readouterr().out.split())

    assert code == 0
    assert "candidates created 3" in printed
    assert "Every result" not in printed


def test_the_run_reports_each_skill_as_well_as_the_whole(monkeypatch, tmp_path, capsys):
    run_retrieve(monkeypatch, tmp_path, "https://aima.cs.berkeley.edu/heuristics.html")
    printed = capsys.readouterr().out

    for skill_id in ("AI-SRC-01", "AI-SRC-02", "AI-SRC-08"):
        assert f"Retrieval diagnostics: {skill_id}" in printed

    assert "Retrieval diagnostics: whole run" in printed


def test_the_summary_never_prints_the_key_or_a_whole_passage(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv(API_KEY_VARIABLE, "sk-live-secret")

    run_retrieve(monkeypatch, tmp_path, "https://aima.cs.berkeley.edu/heuristics.html")
    printed = capsys.readouterr().out

    assert "sk-live-secret" not in printed
    assert PAGE_TEXT not in printed


# Running again against a store that already holds candidates


def test_a_second_run_costs_nothing_once_the_targets_are_held(
    monkeypatch, tmp_path, capsys
):
    """The quota is spent on what is missing, not on what is already there."""
    url = "https://aima.cs.berkeley.edu/heuristics.html"

    assert run_retrieve(monkeypatch, tmp_path, url, limit="1") == 0
    capsys.readouterr()

    assert run_retrieve(monkeypatch, tmp_path, url, limit="1") == 0
    printed = " ".join(capsys.readouterr().out.split())

    assert "Retrieved 0 new candidate(s)" in printed
    assert "search requests made 0" in printed
    assert "already held 3" in printed  # one per pilot skill
    assert "already holds every skill's target" in printed


def test_a_second_run_leaves_the_store_as_it_was(monkeypatch, tmp_path):
    url = "https://aima.cs.berkeley.edu/heuristics.html"
    store = CandidateStore(tmp_path / "candidates.json")

    run_retrieve(monkeypatch, tmp_path, url, limit="1")
    before = store.path.read_bytes()

    run_retrieve(monkeypatch, tmp_path, url, limit="1")

    assert store.path.read_bytes() == before


def test_a_rejected_candidate_is_not_retrieved_again(monkeypatch, tmp_path, capsys):
    """A page a reviewer turned down is the last page worth fetching again."""
    url = "https://aima.cs.berkeley.edu/heuristics.html"
    store = CandidateStore(tmp_path / "candidates.json")

    run_retrieve(monkeypatch, tmp_path, url, limit="1")

    for held in store.load():
        store.replace(reject(held, REVIEWER))

    capsys.readouterr()
    run_retrieve(monkeypatch, tmp_path, url, limit="1")
    printed = " ".join(capsys.readouterr().out.split())

    assert [held.review_status for held in store.load()] == ["rejected"] * 3
    assert "already held 0" in printed  # a rejection is not a reference
    assert "duplicate urls" in printed  # it was searched for, and skipped


# The relevance record: what the reviewer is shown, and what survives


def scored_candidate(score: int = 14, terms=("concept:heuristic", "context:frontier")):
    return new_candidate(
        skill_id="AI-SRC-08",
        title="Heuristics",
        source_url="https://aima.cs.berkeley.edu/heuristics.html",
        source_domain="aima.cs.berkeley.edu",
        passage="A heuristic estimates the remaining cost to a goal state.",
        retrieved_at=FIXED_TIME,
        relevance_score=score,
        matched_terms=terms,
    )


def test_the_score_and_its_terms_survive_a_separate_run(store):
    """A reviewer reads these later, in another session, from the file."""
    store.add([scored_candidate()])

    reopened = CandidateStore(store.path).load()[0]

    assert reopened.relevance_score == 14
    assert reopened.matched_terms == ["concept:heuristic", "context:frontier"]


def test_a_decision_leaves_the_relevance_record_intact(store):
    """Why it was offered stays readable next to what was decided about it."""
    store.add([scored_candidate()])
    held = store.load()[0]

    decided = store.replace(approve(held, REVIEWER, reviewed_at=REVIEWED_TIME))

    assert decided.relevance_score == 14
    assert decided.matched_terms == held.matched_terms
    assert decided.reviewer_id == REVIEWER


def test_a_store_written_before_relevance_filtering_still_loads(store):
    """Older candidates predate the fields and must not become unreadable."""
    store.path.parent.mkdir(parents=True, exist_ok=True)
    record = json.loads(scored_candidate().model_dump_json())
    del record["relevance_score"]
    del record["matched_terms"]
    store.path.write_text(json.dumps([record]), encoding="utf-8")

    loaded = store.load()

    assert len(loaded) == 1
    assert loaded[0].relevance_score == 0
    assert loaded[0].matched_terms == []


def test_listing_shows_why_a_candidate_was_kept(store, capsys):
    store.add([scored_candidate()])

    main(["--store", str(store.path), "list"])

    printed = capsys.readouterr().out

    assert "relevance 14" in printed
    assert "concept:heuristic" in printed


def test_an_unscored_candidate_says_so_rather_than_showing_a_zero(store, capsys):
    """A store from before the filter should not read as a failed score."""
    store.add([candidate("A heuristic estimates remaining cost.")])

    main(["--store", str(store.path), "list"])

    assert "not scored" in capsys.readouterr().out
