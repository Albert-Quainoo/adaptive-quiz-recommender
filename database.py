"""Shared SQLAlchemy engine construction for SQLite (local/test) and
PostgreSQL (production).

`QUIZ_DATABASE_URL`, when set, selects PostgreSQL exclusively. Every other
value passed to `create_engine_for` -- a filesystem path, a `Path`, or the
literal ":memory:" -- selects SQLite, matching this project's
pre-SQLAlchemy behavior.
"""

from pathlib import Path

from sqlalchemy import Connection, Engine, create_engine, event, text
from sqlalchemy.pool import StaticPool

_POSTGRES_PREFIXES = ("postgres://", "postgresql://", "postgresql+psycopg://")


def is_postgres_dsn(value: str) -> bool:
    return value.startswith(_POSTGRES_PREFIXES)


def create_engine_for(
    database: str | Path, *, immediate_transactions: bool = False
) -> Engine:
    if isinstance(database, str) and is_postgres_dsn(database):
        return create_engine(database, pool_pre_ping=True)
    return _create_sqlite_engine(database, immediate_transactions=immediate_transactions)


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
