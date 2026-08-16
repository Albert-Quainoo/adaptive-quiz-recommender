"""Shared SQLAlchemy engine construction for SQLite (local/test) and
PostgreSQL (production).

`QUIZ_DATABASE_URL`, when set, selects PostgreSQL exclusively. Every other
value passed to `create_engine_for` -- a filesystem path, a `Path`, or the
literal ":memory:" -- selects SQLite, matching this project's
pre-SQLAlchemy behavior.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from sqlalchemy import Connection, Engine, create_engine, event, text
from sqlalchemy.pool import StaticPool

_POSTGRES_PREFIXES = ("postgres://", "postgresql://", "postgresql+psycopg://")

# Ambient (engine, connection) for the current thread's unit_of_work(), if
# any -- see unit_of_work() and connection_scope() below. Keyed by engine
# identity, not just "is one active": two different repositories (e.g. the
# learner-path repository and the unrelated replenishment job repository)
# can each construct their own Engine pointing at the same database file/
# DSN, and an ambient connection from one must never be handed to the
# other's connection_scope() call -- that would mix a transaction from one
# Engine's pool with statements issued through a different Engine object,
# which SQLAlchemy does not support and would misbehave in confusing ways.
_active_unit_of_work: ContextVar[tuple[Engine, Connection] | None] = ContextVar(
    "_active_unit_of_work", default=None
)


@contextmanager
def unit_of_work(engine: Engine):
    """Opens exactly one connection and one transaction for the duration of
    the block. Every connection_scope(engine, ...) call anywhere underneath
    it on this thread -- directly, or many function calls deep, across
    module boundaries -- reuses this same connection/transaction instead of
    opening its own, *provided it's the same engine*: one pool checkout for
    the whole learner action instead of one per read/write, and any
    exception anywhere inside rolls back everything the action did, not
    just the one statement that failed.

    Nesting with the *same* engine is a no-op: calling unit_of_work(engine)
    again while one for that engine is already active on this thread just
    yields the existing connection rather than opening a second one, so
    code that is safe to call either standalone or from within a caller's
    unit of work doesn't need to know which situation it's in. Nesting
    with a *different* engine opens its own separate connection/transaction
    as normal -- unit_of_work never shares a connection across engines.

    Callers outside a unit_of_work() block are completely unaffected:
    connection_scope() falls back to opening its own connection exactly as
    every repository method already did before this existed."""
    existing = _active_unit_of_work.get()
    if existing is not None and existing[0] is engine:
        yield existing[1]
        return
    with engine.begin() as connection:
        token = _active_unit_of_work.set((engine, connection))
        try:
            yield connection
        finally:
            _active_unit_of_work.reset(token)


@contextmanager
def connection_scope(engine: Engine, *, write: bool):
    """The one seam every repository read/write method should acquire its
    connection through, instead of calling engine.connect()/engine.begin()
    directly. Participates in an ambient unit_of_work()'s connection if one
    is active on this thread *for this same engine*; otherwise behaves
    exactly as a bare engine.connect() (write=False) or engine.begin()
    (write=True) always did -- unchanged for every caller that never uses
    unit_of_work, standalone scripts and tests included, and unchanged for
    any engine other than the one an active unit_of_work belongs to.
    `write` only selects which kind of connection to open in that fallback
    case; it carries no authority and cannot bypass anything -- it is not a
    flag a caller uses to skip the ambient-connection check, since that
    check always happens first."""
    active = _active_unit_of_work.get()
    if active is not None and active[0] is engine:
        yield active[1]
        return
    if write:
        with engine.begin() as connection:
            yield connection
    else:
        with engine.connect() as connection:
            yield connection


def is_postgres_dsn(value: str) -> bool:
    return value.startswith(_POSTGRES_PREFIXES)


def create_engine_for(
    database: str | Path, *, immediate_transactions: bool = False
) -> Engine:
    if isinstance(database, str) and is_postgres_dsn(database):
        return create_engine(_with_psycopg_driver(database), pool_pre_ping=True)
    return _create_sqlite_engine(database, immediate_transactions=immediate_transactions)


def _with_psycopg_driver(dsn: str) -> str:
    """Providers (Supabase included) hand out bare postgres://.../postgresql://
    connection strings. SQLAlchemy's default driver for that bare scheme is
    psycopg2, which this project does not install -- only psycopg (v3, pinned
    in requirements.txt) is. Force the +psycopg dialect so a DSN copied
    straight from a provider works without the caller needing to know that.
    """
    if dsn.startswith("postgresql+"):
        return dsn
    if dsn.startswith("postgresql://"):
        return "postgresql+psycopg://" + dsn[len("postgresql://") :]
    if dsn.startswith("postgres://"):
        return "postgresql+psycopg://" + dsn[len("postgres://") :]
    return dsn


def _create_sqlite_engine(
    database: str | Path, *, immediate_transactions: bool
) -> Engine:
    path = str(database)
    url = "sqlite://" if path == ":memory:" else f"sqlite:///{path}"
    engine = create_engine(
        url,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )
    begin_statement = "BEGIN IMMEDIATE" if immediate_transactions else "BEGIN"

    @event.listens_for(engine, "connect")
    def _configure_connection(dbapi_connection, connection_record) -> None:
        # pysqlite's own isolation-level tracking otherwise fights
        # SQLAlchemy's transaction tracking; disabling it and letting
        # SQLAlchemy emit BEGIN itself is the documented recipe for
        # consistent transactional behavior on the sqlite3 driver.
        dbapi_connection.isolation_level = None
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("PRAGMA journal_mode = WAL")
        cursor.close()

    @event.listens_for(engine, "begin")
    def _begin(connection: Connection) -> None:
        connection.exec_driver_sql(begin_statement)

    return engine


def execute_schema_script(connection: Connection, schema_sql: str) -> None:
    for statement in schema_sql.split(";"):
        statement = statement.strip()
        if statement:
            connection.execute(text(statement))
