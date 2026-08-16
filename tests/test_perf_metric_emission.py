"""Proves PERF_METRIC lines actually reach captured process stdout -- not
just that a logger method was called. A mocked-logger assertion would pass
even in the exact regime that broke on Streamlit Cloud (an unconfigured
logging.getLogger(...).info() call whose record is silently dropped
because no handler is attached anywhere in the logger's chain), so this
runs app/perf.py in a genuinely fresh subprocess and inspects its real
stdout, the same channel Streamlit Cloud's "Manage app" log viewer reads.
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_SUBPROCESS_SCRIPT = """
import sys
sys.path.insert(0, {repo_root!r})
from app.perf import phase

with phase("course_selection", correlation_id="fresh-subprocess-check", course_id="intro-ai"):
    pass
"""


def _run_in_fresh_subprocess() -> subprocess.CompletedProcess:
    script = _SUBPROCESS_SCRIPT.format(repo_root=str(REPO_ROOT))
    return subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_perf_metric_line_appears_in_fresh_subprocess_stdout():
    completed = _run_in_fresh_subprocess()

    assert completed.returncode == 0, completed.stderr

    metric_lines = [
        line for line in completed.stdout.splitlines() if line.startswith("PERF_METRIC ")
    ]
    assert len(metric_lines) == 1, (
        f"expected exactly one PERF_METRIC line in stdout, got {len(metric_lines)}: "
        f"{completed.stdout!r}"
    )

    payload = json.loads(metric_lines[0][len("PERF_METRIC ") :])
    assert payload["correlation_id"] == "fresh-subprocess-check"
    assert payload["phase"] == "course_selection"
    assert payload["course_id"] == "intro-ai"
    assert isinstance(payload["elapsed_ms"], (int, float))
    assert payload["db_queries"] == 0
    assert payload["db_time_ms"] == 0
    assert payload["pool_checkouts"] == 0
    assert payload["new_connections"] == 0
    assert payload["tx_begins"] == 0
    assert payload["tx_commits"] == 0
    assert payload["tx_rollbacks"] == 0


def test_perf_metric_line_never_contains_learner_or_secret_looking_keys():
    completed = _run_in_fresh_subprocess()
    metric_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("PERF_METRIC ")
    )
    payload = json.loads(metric_line[len("PERF_METRIC ") :])

    forbidden_keys = {
        "learner_id",
        "answer",
        "selected_option_id",
        "database_url",
        "dsn",
        "password",
        "secret",
        "token",
        "credential",
    }
    assert forbidden_keys.isdisjoint(payload.keys())

    allowed_keys = {
        "correlation_id",
        "phase",
        "course_id",
        "elapsed_ms",
        "db_queries",
        "db_time_ms",
        "cache_hit",
        "pool_checkouts",
        "checkout_time_ms",
        "new_connections",
        "tx_begins",
        "tx_commits",
        "tx_rollbacks",
        "tx_time_ms",
    }
    assert set(payload.keys()) <= allowed_keys


def test_nested_phases_do_not_each_emit_their_own_perf_metric_line():
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from app.perf import phase

with phase("total_submit_rerun", correlation_id="nesting-check", course_id="intro-ai"):
    with phase("answer_persistence", course_id="intro-ai"):
        pass
    with phase("bkt_mastery_calculation", course_id="intro-ai"):
        pass
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    metric_lines = [
        line for line in completed.stdout.splitlines() if line.startswith("PERF_METRIC ")
    ]
    assert len(metric_lines) == 1, (
        "only the top-level phase should emit a PERF_METRIC line, nested phases "
        f"are already rolled into it -- got {len(metric_lines)}: {metric_lines}"
    )
    payload = json.loads(metric_lines[0][len("PERF_METRIC ") :])
    assert payload["phase"] == "total_submit_rerun"


def test_pool_and_transaction_instrumentation_reports_real_activity(tmp_path):
    db_path = tmp_path / "perf_instrumentation_check.sqlite3"
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from sqlalchemy import text
from app.perf import phase
from database import create_engine_for

engine = create_engine_for({str(db_path)!r})
with phase("course_selection", correlation_id="pool-tx-check", course_id="intro-ai"):
    with engine.begin() as connection:
        connection.execute(text("CREATE TABLE t (x INTEGER)"))
        connection.execute(text("INSERT INTO t (x) VALUES (1)"))
    with engine.connect() as connection:
        connection.execute(text("SELECT * FROM t")).fetchall()
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    metric_line = next(
        line for line in completed.stdout.splitlines() if line.startswith("PERF_METRIC ")
    )
    payload = json.loads(metric_line[len("PERF_METRIC ") :])

    # CREATE TABLE, INSERT, SELECT, plus the 2 PRAGMA statements
    # database.py's SQLite engine setup issues on the first physical
    # connection (see database.py's "connect" event: foreign_keys, journal_mode).
    assert payload["db_queries"] == 5
    assert payload["pool_checkouts"] == 2  # one engine.begin(), one engine.connect()
    assert payload["new_connections"] >= 1  # at least the first physical connection
    # 2: engine.begin() opens one explicitly, and engine.connect() also
    # auto-begins one under SQLAlchemy's normal transactional bookkeeping
    # once a statement executes on it (a documented SQLAlchemy behavior,
    # not specific to this app).
    assert payload["tx_begins"] == 2
    assert payload["tx_commits"] == 1  # the explicit engine.begin() block
    # The plain engine.connect() read never calls .commit(), so SQLAlchemy
    # closes its auto-begun (read-only) transaction with an implicit
    # rollback -- also standard SQLAlchemy behavior, not this app's.
    assert payload["tx_rollbacks"] == 1
    assert payload["checkout_time_ms"] >= 0
    assert payload["tx_time_ms"] >= 0


def test_repeated_module_import_does_not_duplicate_perf_metric_lines():
    """Simulates the "handler deduplication across Streamlit reruns"
    requirement: app/perf.py's module-level logger setup running more than
    once in the same process (Python caches the module itself, so a plain
    re-import wouldn't re-run this code, but _configure_logger() is called
    directly here to prove it is idempotent even if invoked again)."""
    script = f"""
import sys
sys.path.insert(0, {str(REPO_ROOT)!r})
from app.perf import phase, _configure_logger

_configure_logger()
_configure_logger()
_configure_logger()

with phase("settings_loading", correlation_id="dedup-check"):
    pass
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr

    metric_lines = [
        line for line in completed.stdout.splitlines() if line.startswith("PERF_METRIC ")
    ]
    assert len(metric_lines) == 1, metric_lines

    human_readable_lines = [
        line for line in completed.stdout.splitlines() if line.startswith("phase=")
    ]
    assert len(human_readable_lines) == 1, (
        f"repeated logger configuration produced {len(human_readable_lines)} human-readable "
        f"lines for one phase -- handlers were stacked instead of deduplicated: "
        f"{human_readable_lines}"
    )
