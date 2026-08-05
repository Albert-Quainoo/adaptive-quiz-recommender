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

    def search(self, query, limit, allowed_domains, diagnostics):
        diagnostics.record_query(allowed_domains[0])

        return [SearchResult(title="Heuristics", url=self.url, snippet="")]


class StubFetcher:
    def __init__(self, *arguments, **keywords):
        pass

    def fetch(self, url: str) -> FetchedPage:
        return FetchedPage(url=url, text=PAGE_TEXT)


def run_retrieve(monkeypatch, tmp_path, url: str) -> int:
    monkeypatch.setattr("authoring.retrieval.cli.BraveSearchProvider",
                        StubProvider.bound_to(url))
    monkeypatch.setattr("authoring.retrieval.cli.HttpPageFetcher", StubFetcher)

    return main(["--store", str(tmp_path / "candidates.json"), "retrieve"])


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


def test_the_summary_never_prints_the_key_or_a_whole_passage(
    monkeypatch, tmp_path, capsys
):
    monkeypatch.setenv(API_KEY_VARIABLE, "sk-live-secret")

    run_retrieve(monkeypatch, tmp_path, "https://aima.cs.berkeley.edu/heuristics.html")
    printed = capsys.readouterr().out

    assert "sk-live-secret" not in printed
    assert PAGE_TEXT not in printed
