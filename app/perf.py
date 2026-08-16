"""Structured server-side timing instrumentation for the learner path.

`phase()` wraps a logical step (settings load, controller resolution, a
DB write, ...) and logs one structured line with the step's wall-clock
elapsed time plus the number and total time of any SQL statements executed
anywhere underneath it -- attribution works across module boundaries (e.g.
app/flow.py calling into recommendation/sqlite_repository.py) because the
SQLAlchemy `before/after_cursor_execute` events below are global (installed
once, on the `Engine` class) and simply add to whichever `phase()` is
innermost-active on the current thread, via a contextvar. Phases can nest:
a query executed inside a nested phase is counted on that phase and also
rolled up into every ancestor phase, so an outer "total_submit_rerun" phase
reports the sum of everything that happened inside it.

correlation_id is also contextvar-propagated: the outermost phase() of one
Streamlit rerun generates it once; every nested phase() further down the
call stack -- including inside app/controller.py, bkt/service.py, and
recommendation/service.py -- picks it up automatically with no signature
changes needed on any of those methods, so this instrumentation stays
additive-only over the existing learner-path code.

Never logs learner IDs, submitted answers, database URLs, or credentials --
only a phase name, the per-run correlation_id, course_id, elapsed_ms, and
aggregate DB query count/time.

PERF_METRIC lines: one compact JSON line per completed *top-level* phase
(a phase with no other phase active when it started -- e.g.
"course_selection" or "total_submit_rerun", not the reads/writes nested
inside them, which are already rolled into that top-level phase's own
db_queries/db_time_ms), prefixed "PERF_METRIC " and written via
print(..., flush=True) directly to sys.stdout. This deliberately bypasses
the logging module for this specific line: a bare, unconfigured
logging.getLogger(...).info() call was observed to never reach Streamlit
Cloud's captured logs at all, because Python's logging module silently
drops INFO records when no handler is attached anywhere in the logger's
chain (they fall through to the built-in "handler of last resort", which
only handles WARNING and above) -- there was no Streamlit-specific
suppression involved. The human-readable "phase=..." line below now has a
properly configured handler and is unaffected by this change.
"""

import json
import logging
import sys
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar

from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.pool import Pool

_LOGGER_NAME = "app.perf"
_PERF_METRIC_PREFIX = "PERF_METRIC "


def _configure_logger() -> logging.Logger:
    """Idempotent by construction: Python caches Logger instances by name,
    so app/perf.py being imported more than once in the same process (it
    isn't, under normal execution, but this stays correct regardless of
    Streamlit's rerun/reload mechanics) must not stack a second
    StreamHandler onto the same logger -- that would double (then triple,
    ...) every subsequent human-readable phase line."""
    configured = logging.getLogger(_LOGGER_NAME)
    configured.setLevel(logging.INFO)
    configured.propagate = False  # don't also hand records to the root logger
    already_installed = any(
        getattr(existing_handler, "_app_perf_handler", False)
        for existing_handler in configured.handlers
    )
    if not already_installed:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._app_perf_handler = True
        configured.addHandler(handler)
    return configured


LOGGER = _configure_logger()

_current_frame: ContextVar["_Frame | None"] = ContextVar("_current_frame", default=None)
_current_correlation_id: ContextVar[str | None] = ContextVar(
    "_current_correlation_id", default=None
)
# SQLAlchemy 2.0 has no before/after event pair around the actual commit()/
# rollback() DBAPI call (unlike cursor execute) -- "commit"/"rollback" fire
# once, after the fact. As a correctness-safe proxy for how long the commit/
# rollback itself took, this records perf_counter() right after every query
# completes; the commit/rollback handlers below then measure the gap since
# that last query -- accurate for the common case where nothing else runs
# between the last statement and the transaction's end.
_last_query_end: ContextVar[float | None] = ContextVar("_last_query_end", default=None)


class _Frame:
    __slots__ = (
        "parent",
        "query_count",
        "db_time_ms",
        "pool_checkouts",
        "checkout_time_ms",
        "new_connections",
        "tx_begins",
        "tx_commits",
        "tx_rollbacks",
        "tx_time_ms",
    )

    def __init__(self, parent: "_Frame | None") -> None:
        self.parent = parent
        self.query_count = 0
        self.db_time_ms = 0.0
        self.pool_checkouts = 0
        self.checkout_time_ms = 0.0
        self.new_connections = 0
        self.tx_begins = 0
        self.tx_commits = 0
        self.tx_rollbacks = 0
        self.tx_time_ms = 0.0

    def _absorb_child(self, child: "_Frame") -> None:
        self.query_count += child.query_count
        self.db_time_ms += child.db_time_ms
        self.pool_checkouts += child.pool_checkouts
        self.checkout_time_ms += child.checkout_time_ms
        self.new_connections += child.new_connections
        self.tx_begins += child.tx_begins
        self.tx_commits += child.tx_commits
        self.tx_rollbacks += child.tx_rollbacks
        self.tx_time_ms += child.tx_time_ms


def new_correlation_id() -> str:
    return uuid.uuid4().hex[:12]


@event.listens_for(Engine, "before_cursor_execute")
def _before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    context._perf_start = time.perf_counter()


@event.listens_for(Engine, "after_cursor_execute")
def _after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
    now = time.perf_counter()
    _last_query_end.set(now)
    frame = _current_frame.get()
    if frame is None:
        return
    start = getattr(context, "_perf_start", None)
    if start is None:
        return
    frame.query_count += 1
    frame.db_time_ms += (now - start) * 1000


@event.listens_for(Pool, "connect")
def _on_pool_connect(dbapi_connection, connection_record):
    """Fires only when the pool creates a brand-new physical DBAPI
    connection (a pool miss/cold start) -- never on a checkout that reuses
    an already-open pooled connection."""
    frame = _current_frame.get()
    if frame is not None:
        frame.new_connections += 1


@event.listens_for(Pool, "checkout")
def _on_pool_checkout(dbapi_connection, connection_record, connection_proxy):
    connection_record.info["_perf_checkout_start"] = time.perf_counter()
    frame = _current_frame.get()
    if frame is not None:
        frame.pool_checkouts += 1


@event.listens_for(Pool, "checkin")
def _on_pool_checkin(dbapi_connection, connection_record):
    start = connection_record.info.pop("_perf_checkout_start", None)
    if start is None:
        return
    frame = _current_frame.get()
    if frame is not None:
        frame.checkout_time_ms += (time.perf_counter() - start) * 1000


@event.listens_for(Engine, "begin")
def _on_begin(conn):
    frame = _current_frame.get()
    if frame is not None:
        frame.tx_begins += 1


def _time_since_last_query() -> float:
    """See _last_query_end's comment: SQLAlchemy 2.0 has no before/after
    pair around the actual commit()/rollback() DBAPI call, so this measures
    the gap since the last query completed as a safe, non-invasive proxy --
    never replaces or alters the actual commit/rollback operation itself."""
    last = _last_query_end.get()
    if last is None:
        return 0.0
    return max(0.0, (time.perf_counter() - last) * 1000)


@event.listens_for(Engine, "commit")
def _on_commit(conn):
    frame = _current_frame.get()
    if frame is not None:
        frame.tx_commits += 1
        frame.tx_time_ms += _time_since_last_query()


@event.listens_for(Engine, "rollback")
def _on_rollback(conn):
    frame = _current_frame.get()
    if frame is not None:
        frame.tx_rollbacks += 1
        frame.tx_time_ms += _time_since_last_query()


@contextmanager
def phase(
    name: str,
    *,
    correlation_id: str | None = None,
    course_id: str | None = None,
    **extra: object,
):
    """Time one logical step. Extra keyword args (e.g. cache_hit=True) are
    appended to the log line as key=value; never pass learner-identifying
    or secret values here.

    correlation_id: pass explicitly only at a request's outermost phase
    (e.g. app/main.py's per-rerun phases) if you want a caller-chosen id;
    otherwise omit it everywhere and nested phases automatically share the
    id generated by the outermost phase of the current call stack."""
    correlation_token = None
    if correlation_id is None:
        correlation_id = _current_correlation_id.get()
        if correlation_id is None:
            correlation_id = new_correlation_id()
            correlation_token = _current_correlation_id.set(correlation_id)
    else:
        correlation_token = _current_correlation_id.set(correlation_id)

    parent = _current_frame.get()
    is_top_level = parent is None
    frame = _Frame(parent)
    token = _current_frame.set(frame)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000
        _current_frame.reset(token)
        if frame.parent is not None:
            frame.parent._absorb_child(frame)
        extras = "".join(f" {key}={value}" for key, value in extra.items())
        LOGGER.info(
            "phase=%s correlation_id=%s course_id=%s elapsed_ms=%.2f "
            "db_queries=%d db_time_ms=%.2f pool_checkouts=%d checkout_time_ms=%.2f "
            "new_connections=%d tx_begins=%d tx_commits=%d tx_rollbacks=%d "
            "tx_time_ms=%.2f%s",
            name,
            correlation_id,
            course_id or "-",
            elapsed_ms,
            frame.query_count,
            frame.db_time_ms,
            frame.pool_checkouts,
            frame.checkout_time_ms,
            frame.new_connections,
            frame.tx_begins,
            frame.tx_commits,
            frame.tx_rollbacks,
            frame.tx_time_ms,
            extras,
        )
        if is_top_level:
            metric = {
                "correlation_id": correlation_id,
                "phase": name,
                "course_id": course_id or "-",
                "elapsed_ms": round(elapsed_ms, 2),
                "db_queries": frame.query_count,
                "db_time_ms": round(frame.db_time_ms, 2),
                "pool_checkouts": frame.pool_checkouts,
                "checkout_time_ms": round(frame.checkout_time_ms, 2),
                "new_connections": frame.new_connections,
                "tx_begins": frame.tx_begins,
                "tx_commits": frame.tx_commits,
                "tx_rollbacks": frame.tx_rollbacks,
                "tx_time_ms": round(frame.tx_time_ms, 2),
            }
            if "cache_hit" in extra:
                metric["cache_hit"] = extra["cache_hit"]
            print(_PERF_METRIC_PREFIX + json.dumps(metric, sort_keys=True), flush=True)
        if correlation_token is not None:
            _current_correlation_id.reset(correlation_token)
