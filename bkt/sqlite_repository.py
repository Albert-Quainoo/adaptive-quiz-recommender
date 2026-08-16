import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from uuid import uuid4

from sqlalchemy import Connection, Engine, text
from sqlalchemy.engine import RowMapping
from sqlalchemy.exc import IntegrityError

from bkt.repository import AttemptConflictError
from bkt.schemas import AttemptEvent, BKTModelMetadata, MasterySnapshot
from database import connection_scope, create_engine_for, execute_schema_script
from database import unit_of_work as _unit_of_work

# The course_id every pre-migration row is backfilled to: the catalog id the
# original single-course "ai" deployment was migrated to (see
# authoring/replenishment/manifests/intro-ai.json). Every table this module
# owns predates multi-course support, so every row that exists before a
# database's first migration run belongs to this one course by definition.
DEFAULT_MIGRATION_COURSE_ID = "intro-ai"

# Bumped whenever SCHEMA's shape changes in a way that requires a migration
# pass (not just an additive CREATE TABLE IF NOT EXISTS). Recorded in
# schema_migrations by both a fresh CREATE (initialize_schema's EMPTY path)
# and an explicit legacy migration (run_course_ownership_migration), so
# readiness is always determined by reading this value back, never by
# inferring from table shape or catching a generic database error.
CURRENT_SCHEMA_VERSION = "course_id_v1"

SCHEMA_MIGRATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version TEXT PRIMARY KEY,
    applied_at TEXT NOT NULL
);
"""

_APPLICATION_TABLES = (
    "attempt_events",
    "learner_sessions",
    "mastery_snapshots",
    "question_presentations",
    "recommendation_events",
    "content_gap_events",
    "bkt_model_metadata",
)


class SchemaStatus(Enum):
    """What check_schema_status found, read-only. EMPTY: no application
    tables exist yet -- safe to create the latest schema directly, nothing
    to preserve. CURRENT: schema_migrations already records
    CURRENT_SCHEMA_VERSION -- nothing to do. MIGRATION_REQUIRED: either a
    legacy pre-course_id database (application tables exist but
    schema_migrations does not) or a schema_migrations row for an older
    version -- must not be touched except via the explicit, protected
    run_course_ownership_migration()."""

    EMPTY = "empty"
    CURRENT = "current"
    MIGRATION_REQUIRED = "migration_required"


class SchemaMigrationRequiredError(RuntimeError):
    """Raised by SQLiteBKTRepository.initialize_schema() -- never by
    run_course_ownership_migration() -- when check_schema_status finds
    MIGRATION_REQUIRED. Never mutates anything itself; the caller (see
    app/bootstrap.py's build_controller) is expected to translate this into
    a maintenance-mode-style message for the learner, not a raw crash."""

    user_message = (
        "The quiz service is in maintenance mode: a required database "
        "migration has not been applied yet. Please try again later."
    )


# Final, course-scoped shape. A brand-new database gets this shape directly
# via CREATE TABLE IF NOT EXISTS; an existing pre-course_id database gets
# here via migrate_course_ownership's table rebuild (see below) -- either
# path lands on the same schema, so every table is always in exactly this
# shape once initialize_schema() returns.
#
# Two composite foreign keys make cross-course references impossible at the
# database level, not just by convention:
#   mastery_snapshots(course_id, source_attempt_id)
#       REFERENCES attempt_events(course_id, attempt_id)
#   question_presentations(learner_id, course_id)
#       REFERENCES learner_sessions(learner_id, course_id)
# A snapshot or presentation naming a real attempt_id/learner_id that exists
# under a *different* course_id still fails the FK, because the referenced
# tuple (course_id, id) doesn't exist -- only (this course_id, id) does.
#
# Deliberately no CREATE INDEX statements here: CREATE TABLE IF NOT EXISTS
# is a no-op against an existing pre-migration table (old shape, no
# course_id yet), so an index on course_id run at this point would fail
# against that not-yet-migrated table. Indexes are created once, always,
# after migrate_course_ownership -- by then every table (freshly created or
# just migrated) is guaranteed to be in this final shape.
SCHEMA = """
CREATE TABLE IF NOT EXISTS attempt_events (
    attempt_id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    presentation_id TEXT NOT NULL,
    learner_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    selected_option_id TEXT NOT NULL,
    correct INTEGER NOT NULL CHECK (correct IN (0, 1)),
    attempt_order INTEGER NOT NULL,
    occurred_at TEXT NOT NULL,
    UNIQUE (learner_id, course_id, skill_id, attempt_order),
    UNIQUE (course_id, attempt_id)
);

CREATE TABLE IF NOT EXISTS learner_sessions (
    learner_id TEXT NOT NULL,
    course_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (learner_id, course_id)
);

CREATE TABLE IF NOT EXISTS mastery_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    learner_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    mastery_probability REAL NOT NULL,
    attempt_count INTEGER NOT NULL,
    source_attempt_id TEXT NOT NULL,
    model_version TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE (source_attempt_id, model_version),
    FOREIGN KEY (course_id, source_attempt_id)
        REFERENCES attempt_events (course_id, attempt_id)
);

CREATE TABLE IF NOT EXISTS question_presentations (
    presentation_id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    learner_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    presentation_seed TEXT NOT NULL,
    recommendation_reason TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (learner_id, course_id)
        REFERENCES learner_sessions (learner_id, course_id)
);

CREATE TABLE IF NOT EXISTS recommendation_events (
    recommendation_id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    learner_id TEXT NOT NULL,
    skill_id TEXT NOT NULL,
    item_id TEXT NOT NULL,
    difficulty TEXT NOT NULL,
    mastery_probability REAL NOT NULL,
    reason TEXT NOT NULL,
    model_version TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS content_gap_events (
    content_gap_id TEXT PRIMARY KEY,
    course_id TEXT NOT NULL,
    learner_id TEXT NOT NULL,
    completed_skill_id TEXT NOT NULL,
    completed_skill_name TEXT NOT NULL,
    newly_unlocked_skill_id TEXT NOT NULL,
    newly_unlocked_skill_name TEXT NOT NULL,
    missing_approved_content INTEGER NOT NULL CHECK (missing_approved_content IN (0, 1)),
    current_mastery_probability REAL NOT NULL,
    prerequisite_mastery_threshold REAL NOT NULL,
    reason TEXT NOT NULL,
    model_version TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bkt_model_metadata (
    model_version TEXT NOT NULL,
    course_id TEXT NOT NULL,
    fitted_at TEXT NOT NULL,
    training_attempt_count INTEGER NOT NULL,
    skill_ids TEXT NOT NULL,
    PRIMARY KEY (course_id, model_version)
);
"""

INDEXES = """
CREATE INDEX IF NOT EXISTS idx_attempt_events_course_learner
ON attempt_events(course_id, learner_id);
CREATE INDEX IF NOT EXISTS idx_mastery_snapshots_identity
ON mastery_snapshots(course_id, learner_id, skill_id);
CREATE INDEX IF NOT EXISTS idx_recommendation_events_course_learner
ON recommendation_events(course_id, learner_id);
CREATE INDEX IF NOT EXISTS idx_content_gap_events_course_learner
ON content_gap_events(course_id, learner_id);
CREATE INDEX IF NOT EXISTS idx_question_presentations_course_learner
ON question_presentations(course_id, learner_id);
"""


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(timezone.utc).isoformat()


def _aware_datetime(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("stored timestamps must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _table_has_column(connection: Connection, table: str, column: str) -> bool:
    """Dialect-explicit column-existence check, used to decide -- once per
    table, at migration time -- whether that table still has the
    pre-course_id shape. Deliberately not the high-level SQLAlchemy
    Inspector: its reflection cache is not guaranteed fresh for DDL just
    issued earlier in the same transaction, and this migration issues DDL
    for one table, then immediately needs an accurate answer for the next."""
    dialect = connection.dialect.name
    if dialect == "sqlite":
        rows = connection.execute(text(f"PRAGMA table_info({table})")).mappings().all()
        return any(row["name"] == column for row in rows)
    if dialect in ("postgresql", "postgres"):
        row = connection.execute(
            text(
                """
                SELECT 1 FROM information_schema.columns
                WHERE table_name = :table AND column_name = :column
                """
            ),
            {"table": table, "column": column},
        ).first()
        return row is not None
    raise RuntimeError(f"unsupported dialect for migration: {dialect}")


def _table_exists(connection: Connection, table: str) -> bool:
    dialect = connection.dialect.name
    if dialect == "sqlite":
        row = connection.execute(
            text("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = :table"),
            {"table": table},
        ).first()
        return row is not None
    if dialect in ("postgresql", "postgres"):
        # Scoped to public: Supabase (and Postgres generally) ships its own
        # same-named tables in other schemas -- e.g. auth.schema_migrations,
        # realtime.schema_migrations -- which an unscoped table_name match
        # would collide with, since this application always creates its
        # tables unqualified (i.e. in the default "public" schema).
        row = connection.execute(
            text(
                "SELECT 1 FROM information_schema.tables "
                "WHERE table_name = :table AND table_schema = 'public'"
            ),
            {"table": table},
        ).first()
        return row is not None
    raise RuntimeError(f"unsupported dialect for schema check: {dialect}")


def check_schema_status(connection: Connection) -> SchemaStatus:
    """Read-only -- issues no DDL and no writes. The sole source of truth
    for whether the application schema is ready to use, so runtime code
    never has to infer readiness by catching a generic database error.

    EMPTY is decided purely by whether any application table exists, never
    by schema_migrations alone: schema_migrations is bookkeeping about the
    application tables, not a substitute for them existing. Without this
    ordering, a database where every application table was dropped but
    schema_migrations happened to survive would misreport CURRENT and
    initialize_schema() would then never recreate the tables it needs."""
    if not any(_table_exists(connection, table) for table in _APPLICATION_TABLES):
        return SchemaStatus.EMPTY
    if _table_exists(connection, "schema_migrations"):
        row = connection.execute(
            text(
                "SELECT version FROM schema_migrations "
                "ORDER BY applied_at DESC LIMIT 1"
            )
        ).first()
        version = row[0] if row is not None else None
        if version == CURRENT_SCHEMA_VERSION:
            return SchemaStatus.CURRENT
    return SchemaStatus.MIGRATION_REQUIRED


def _record_schema_version(
    connection: Connection, *, version: str = CURRENT_SCHEMA_VERSION
) -> None:
    execute_schema_script(connection, SCHEMA_MIGRATIONS_TABLE)
    connection.execute(
        text(
            "INSERT INTO schema_migrations (version, applied_at) "
            "VALUES (:version, :applied_at) ON CONFLICT (version) DO NOTHING"
        ),
        {"version": version, "applied_at": _utc_iso(datetime.now(timezone.utc))},
    )


def _set_sqlite_foreign_keys(engine: Engine, *, enabled: bool) -> None:
    """PRAGMA foreign_keys is a documented no-op while a transaction is
    open -- and a bare `engine.connect()` still auto-begins one under
    SQLAlchemy's normal transactional bookkeeping (verified: toggling
    through a plain Connection silently does nothing). raw_connection()
    bypasses SQLAlchemy's Connection/transaction layer entirely, operating
    the DBAPI connection directly, which is the only way this pragma
    reliably takes effect. Safe with this project's StaticPool: there is
    exactly one underlying DBAPI connection, so this is the same connection
    initialize_schema()'s migration transaction runs on next."""
    raw = engine.raw_connection()
    try:
        cursor = raw.cursor()
        cursor.execute(f"PRAGMA foreign_keys={'ON' if enabled else 'OFF'}")
        cursor.close()
    finally:
        raw.close()


def _sqlite_foreign_key_violations(engine: Engine) -> list[tuple]:
    raw = engine.raw_connection()
    try:
        cursor = raw.cursor()
        cursor.execute("PRAGMA foreign_key_check")
        rows = cursor.fetchall()
        cursor.close()
        return rows
    finally:
        raw.close()


# Each entry: (table, CREATE DDL for the rebuilt table, INSERT ... SELECT
# column mapping). Parent tables are rebuilt before the children that
# reference them, matching the FKs restored to the schema above:
# attempt_events and learner_sessions (parents, no FK of their own) come
# first, then mastery_snapshots (FKs attempt_events) and
# question_presentations (FKs learner_sessions). While a table is mid-swap
# (renamed away, not yet replaced) it briefly cannot be a valid FK target,
# so foreign key enforcement is turned off for SQLite around the whole
# migration transaction (see initialize_schema) rather than relying on
# ordering alone -- ordering here is about producing a clean, dependency-
# respecting migration, not what makes it safe; the PRAGMA toggle is what
# makes it safe. recommendation_events, content_gap_events, and
# bkt_model_metadata have no FK relationships and can run anywhere.
_MIGRATION_STEPS: list[tuple[str, str, str]] = [
    (
        "attempt_events",
        """
        CREATE TABLE attempt_events__migrated (
            attempt_id TEXT PRIMARY KEY,
            course_id TEXT NOT NULL,
            presentation_id TEXT NOT NULL,
            learner_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            skill_id TEXT NOT NULL,
            selected_option_id TEXT NOT NULL,
            correct INTEGER NOT NULL CHECK (correct IN (0, 1)),
            attempt_order INTEGER NOT NULL,
            occurred_at TEXT NOT NULL,
            UNIQUE (learner_id, course_id, skill_id, attempt_order),
            UNIQUE (course_id, attempt_id)
        )
        """,
        """
        INSERT INTO attempt_events__migrated (
            attempt_id, course_id, presentation_id, learner_id, item_id, skill_id,
            selected_option_id, correct, attempt_order, occurred_at
        )
        SELECT attempt_id, :default_course_id, presentation_id, learner_id, item_id,
               skill_id, selected_option_id, correct, attempt_order, occurred_at
        FROM attempt_events
        """,
    ),
    (
        "learner_sessions",
        """
        CREATE TABLE learner_sessions__migrated (
            learner_id TEXT NOT NULL,
            course_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY (learner_id, course_id)
        )
        """,
        """
        INSERT INTO learner_sessions__migrated (learner_id, course_id, created_at)
        SELECT learner_id, :default_course_id, created_at
        FROM learner_sessions
        """,
    ),
    (
        "mastery_snapshots",
        """
        CREATE TABLE mastery_snapshots__migrated (
            snapshot_id TEXT PRIMARY KEY,
            course_id TEXT NOT NULL,
            learner_id TEXT NOT NULL,
            skill_id TEXT NOT NULL,
            mastery_probability REAL NOT NULL,
            attempt_count INTEGER NOT NULL,
            source_attempt_id TEXT NOT NULL,
            model_version TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE (source_attempt_id, model_version),
            FOREIGN KEY (course_id, source_attempt_id)
                REFERENCES attempt_events (course_id, attempt_id)
        )
        """,
        """
        INSERT INTO mastery_snapshots__migrated (
            snapshot_id, course_id, learner_id, skill_id, mastery_probability,
            attempt_count, source_attempt_id, model_version, updated_at
        )
        SELECT snapshot_id, :default_course_id, learner_id, skill_id,
               mastery_probability, attempt_count, source_attempt_id, model_version,
               updated_at
        FROM mastery_snapshots
        """,
    ),
    (
        "question_presentations",
        """
        CREATE TABLE question_presentations__migrated (
            presentation_id TEXT PRIMARY KEY,
            course_id TEXT NOT NULL,
            learner_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            presentation_seed TEXT NOT NULL,
            recommendation_reason TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (learner_id, course_id)
                REFERENCES learner_sessions (learner_id, course_id)
        )
        """,
        """
        INSERT INTO question_presentations__migrated (
            presentation_id, course_id, learner_id, item_id, presentation_seed,
            recommendation_reason, created_at
        )
        SELECT presentation_id, :default_course_id, learner_id, item_id,
               presentation_seed, recommendation_reason, created_at
        FROM question_presentations
        """,
    ),
    (
        "recommendation_events",
        """
        CREATE TABLE recommendation_events__migrated (
            recommendation_id TEXT PRIMARY KEY,
            course_id TEXT NOT NULL,
            learner_id TEXT NOT NULL,
            skill_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            difficulty TEXT NOT NULL,
            mastery_probability REAL NOT NULL,
            reason TEXT NOT NULL,
            model_version TEXT NOT NULL,
            policy_version TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        """
        INSERT INTO recommendation_events__migrated (
            recommendation_id, course_id, learner_id, skill_id, item_id, difficulty,
            mastery_probability, reason, model_version, policy_version, created_at
        )
        SELECT recommendation_id, :default_course_id, learner_id, skill_id, item_id,
               difficulty, mastery_probability, reason, model_version, policy_version,
               created_at
        FROM recommendation_events
        """,
    ),
    (
        "content_gap_events",
        """
        CREATE TABLE content_gap_events__migrated (
            content_gap_id TEXT PRIMARY KEY,
            course_id TEXT NOT NULL,
            learner_id TEXT NOT NULL,
            completed_skill_id TEXT NOT NULL,
            completed_skill_name TEXT NOT NULL,
            newly_unlocked_skill_id TEXT NOT NULL,
            newly_unlocked_skill_name TEXT NOT NULL,
            missing_approved_content INTEGER NOT NULL CHECK (missing_approved_content IN (0, 1)),
            current_mastery_probability REAL NOT NULL,
            prerequisite_mastery_threshold REAL NOT NULL,
            reason TEXT NOT NULL,
            model_version TEXT NOT NULL,
            policy_version TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """,
        """
        INSERT INTO content_gap_events__migrated (
            content_gap_id, course_id, learner_id, completed_skill_id,
            completed_skill_name, newly_unlocked_skill_id, newly_unlocked_skill_name,
            missing_approved_content, current_mastery_probability,
            prerequisite_mastery_threshold, reason, model_version, policy_version,
            created_at
        )
        SELECT content_gap_id, :default_course_id, learner_id, completed_skill_id,
               completed_skill_name, newly_unlocked_skill_id, newly_unlocked_skill_name,
               missing_approved_content, current_mastery_probability,
               prerequisite_mastery_threshold, reason, model_version, policy_version,
               created_at
        FROM content_gap_events
        """,
    ),
    (
        "bkt_model_metadata",
        """
        CREATE TABLE bkt_model_metadata__migrated (
            model_version TEXT NOT NULL,
            course_id TEXT NOT NULL,
            fitted_at TEXT NOT NULL,
            training_attempt_count INTEGER NOT NULL,
            skill_ids TEXT NOT NULL,
            PRIMARY KEY (course_id, model_version)
        )
        """,
        """
        INSERT INTO bkt_model_metadata__migrated (
            model_version, course_id, fitted_at, training_attempt_count, skill_ids
        )
        SELECT model_version, :default_course_id, fitted_at, training_attempt_count,
               skill_ids
        FROM bkt_model_metadata
        """,
    ),
]


def _drop_legacy_postgres_foreign_keys(connection: Connection, tables: list[str]) -> None:
    """PostgreSQL has no session-wide 'disable FK enforcement' equivalent to
    SQLite's PRAGMA foreign_keys -- a legacy child table's live FK into a
    legacy parent table blocks DROP TABLE on that parent (confirmed:
    dropping attempt_events while the old mastery_snapshots still carries
    `mastery_snapshots_source_attempt_id_fkey` raises
    DependentObjectsStillExist). So on Postgres, before any table in this
    migration is dropped and rebuilt, drop every FK constraint where an
    about-to-be-migrated table is the referencing (child) side -- this
    clears the way regardless of which table in the pair happens to be
    processed first. The new, course-aware FK is recreated from scratch
    when that child table is rebuilt further down."""
    for table in tables:
        constraint_names = connection.execute(
            text(
                """
                SELECT constraint_name FROM information_schema.table_constraints
                WHERE table_name = :table AND constraint_type = 'FOREIGN KEY'
                """
            ),
            {"table": table},
        ).scalars().all()
        for constraint_name in constraint_names:
            connection.execute(
                text(f'ALTER TABLE {table} DROP CONSTRAINT "{constraint_name}"')
            )


def migrate_course_ownership(
    connection: Connection, *, default_course_id: str = DEFAULT_MIGRATION_COURSE_ID
) -> dict[str, int]:
    """Idempotent: a table already in the course-scoped shape (course_id
    column present -- true immediately after a fresh CREATE TABLE IF NOT
    EXISTS, or after a previous run of this function) is left untouched.
    Returns {table: rows_backfilled} for tables actually migrated this call
    -- an empty dict means every table was already migrated.

    Does not create indexes itself -- INDEXES is run unconditionally by
    initialize_schema() after this returns, since a freshly-created (never
    migrated) database needs them too, not just a migrated one. Does not
    manage the SQLite foreign_keys pragma itself either -- that must be
    turned off before this function's transaction begins and back on (with
    a foreign_key_check) after it ends; see initialize_schema()."""

    tables_needing_migration = [
        table
        for table, _, _ in _MIGRATION_STEPS
        if not _table_has_column(connection, table, "course_id")
    ]

    if connection.dialect.name in ("postgresql", "postgres") and tables_needing_migration:
        _drop_legacy_postgres_foreign_keys(connection, tables_needing_migration)

    backfilled: dict[str, int] = {}
    for table, create_sql, insert_sql in _MIGRATION_STEPS:
        if table not in tables_needing_migration:
            continue
        connection.execute(text(create_sql))
        result = connection.execute(
            text(insert_sql), {"default_course_id": default_course_id}
        )
        connection.execute(text(f"DROP TABLE {table}"))
        connection.execute(
            text(f"ALTER TABLE {table}__migrated RENAME TO {table}")
        )
        backfilled[table] = result.rowcount if result.rowcount is not None else 0

    return backfilled


def run_course_ownership_migration(
    database: str | Path, *, course_id: str = DEFAULT_MIGRATION_COURSE_ID
) -> dict[str, int]:
    """The sole entry point that performs the legacy course_id migration --
    the only function in this module allowed to rebuild tables and backfill
    rows. Never called from application runtime code (build_controller(),
    course selection, login, or any learner request all go through
    SQLiteBKTRepository.initialize_schema() instead, which raises rather
    than migrates). Intended to be invoked only from the protected CLI at
    scripts/migrate_course_ownership.py, run manually and explicitly.

    Idempotent and safe against every SchemaStatus: CURRENT is a no-op
    (returns {}); EMPTY creates the latest schema directly (nothing to
    migrate); MIGRATION_REQUIRED performs the full rebuild-and-backfill.
    Constructs and disposes its own engine -- independent of any
    already-open repository connection.

    Failure safety: the entire rebuild-and-backfill runs inside one
    transaction (`with engine.begin()`). Any failure partway rolls back
    automatically -- both dialects leave the database in exactly its
    pre-migration state, byte-for-byte, on a failed attempt."""
    engine = create_engine_for(database)
    try:
        with engine.connect() as connection:
            status = check_schema_status(connection)

        if status is SchemaStatus.CURRENT:
            return {}

        is_sqlite = engine.dialect.name == "sqlite"

        # Must happen before the migration transaction begins: PRAGMA
        # foreign_keys is a no-op once a transaction is open, and dropping a
        # table that's mid-swap (renamed away, not yet replaced) would
        # otherwise fail against a still-live FK from an unmigrated sibling.
        if is_sqlite:
            _set_sqlite_foreign_keys(engine, enabled=False)

        try:
            with engine.begin() as connection:
                execute_schema_script(connection, SCHEMA)
                backfilled = migrate_course_ownership(
                    connection, default_course_id=course_id
                )
                execute_schema_script(connection, INDEXES)
                _record_schema_version(connection)
        finally:
            if is_sqlite:
                _set_sqlite_foreign_keys(engine, enabled=True)

        if is_sqlite:
            violations = _sqlite_foreign_key_violations(engine)
            if violations:
                raise RuntimeError(
                    "course_id migration left foreign-key violations, refusing to "
                    f"report success: {violations}"
                )
        # PostgreSQL needs no equivalent post-check: every composite FK
        # above is created as part of the same CREATE TABLE the
        # INSERT ... SELECT then populates, inside migrate_course_ownership's
        # single transaction -- Postgres validates each inserted row against
        # its FK immediately, so an orphan already aborts (and rolls back)
        # the migration transaction rather than needing a separate check.

        return backfilled
    finally:
        engine.dispose()


class SQLiteBKTRepository:
    """SQL persistence (SQLite or PostgreSQL) for immutable attempts and
    versioned mastery history, bound to exactly one course.

    course_id is required at construction and every query filters by it --
    this instance is the actual isolation boundary between courses sharing
    one physical database, not any assumption about skill_id/item_id naming.
    A caller cannot forget to scope a query by course, because there is no
    method on this class that does not already scope by self.course_id.
    Composite foreign keys (see SCHEMA) additionally make it impossible,
    at the database level, for a mastery snapshot or question presentation
    to reference a parent row belonging to a different course.
    """

    def __init__(self, database: str | Path, *, course_id: str) -> None:
        self.database = str(database)
        self.course_id = course_id.strip()
        if not self.course_id:
            raise ValueError("course_id is required")
        self._engine = create_engine_for(database)

    def initialize_schema(self) -> SchemaStatus:
        """Runtime-safe: called from build_controller() on every course
        selection/login, so it must never mutate an existing production
        schema and must never perform the legacy course_id migration --
        that is the one job reserved for the explicit, protected
        run_course_ownership_migration() below, invoked only from
        scripts/migrate_course_ownership.py.

        EMPTY: the database is genuinely empty (no application tables at
        all) -- creates the latest schema directly. This is first-time
        setup, not a migration: there is no pre-existing data to preserve,
        so no table rebuild or foreign_keys pragma dance is needed.

        CURRENT: schema_migrations already records CURRENT_SCHEMA_VERSION --
        no-op.

        MIGRATION_REQUIRED: a legacy pre-course_id database (or one on an
        older recorded version). Raises SchemaMigrationRequiredError without
        touching anything -- the caller must fail safely with a
        maintenance/configuration message, never mutate."""
        with connection_scope(self._engine, write=False) as connection:
            status = check_schema_status(connection)

        if status is SchemaStatus.MIGRATION_REQUIRED:
            raise SchemaMigrationRequiredError(
                "database schema predates course_id support (or is on an "
                "older version); run the explicit course ownership "
                "migration (scripts/migrate_course_ownership.py) before "
                "starting the application"
            )

        if status is SchemaStatus.CURRENT:
            return status

        with connection_scope(self._engine, write=True) as connection:
            execute_schema_script(connection, SCHEMA)
            execute_schema_script(connection, INDEXES)
            _record_schema_version(connection)

        return status

    def close(self) -> None:
        self._engine.dispose()

    def unit_of_work(self):
        """One connection, one transaction, for the duration of the with
        block -- every read/write this repository (or any repository
        sharing this thread, e.g. via app/controller.py calling into
        bkt/service.py) performs inside it participates instead of opening
        its own. See database.unit_of_work() for the full contract."""
        return _unit_of_work(self._engine)

    def __enter__(self) -> "SQLiteBKTRepository":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def _require_own_course(self, course_id: str, *, what: str) -> None:
        if course_id != self.course_id:
            raise ValueError(
                f"{what} has course_id {course_id!r}, but this repository is "
                f"bound to {self.course_id!r}"
            )

    def save_attempt(self, attempt: AttemptEvent) -> None:
        self._require_own_course(attempt.course_id, what="attempt")
        with connection_scope(self._engine, write=False) as connection:
            existing = self._select_attempt(connection, attempt.attempt_id)
        if existing is not None:
            if existing == attempt:
                return
            raise AttemptConflictError(
                f"attempt_id {attempt.attempt_id!r} is already used by a different attempt"
            )

        try:
            with connection_scope(self._engine, write=True) as connection:
                self._insert_attempt(connection, attempt)
        except IntegrityError as exc:
            raise AttemptConflictError(str(exc)) from exc

    def get_attempt(self, attempt_id: str) -> AttemptEvent | None:
        with connection_scope(self._engine, write=False) as connection:
            return self._select_attempt(connection, attempt_id)

    def list_attempts(
        self, *, learner_id: str | None = None, skill_id: str | None = None
    ) -> list[AttemptEvent]:
        clauses: list[str] = ["course_id = :course_id"]
        parameters: dict[str, str] = {"course_id": self.course_id}
        if learner_id is not None:
            clauses.append("learner_id = :learner_id")
            parameters["learner_id"] = learner_id
        if skill_id is not None:
            clauses.append("skill_id = :skill_id")
            parameters["skill_id"] = skill_id
        where = f" WHERE {' AND '.join(clauses)}"
        with connection_scope(self._engine, write=False) as connection:
            rows = (
                connection.execute(
                    text(
                        "SELECT * FROM attempt_events"
                        f"{where} ORDER BY attempt_order, occurred_at, attempt_id"
                    ),
                    parameters,
                )
                .mappings()
                .all()
            )
        return [self._attempt_from_row(row) for row in rows]

    def save_mastery(self, snapshot: MasterySnapshot) -> None:
        self._require_own_course(snapshot.course_id, what="mastery snapshot")
        with connection_scope(self._engine, write=True) as connection:
            self._insert_mastery(connection, snapshot)

    def save_attempt_and_mastery(
        self, attempt: AttemptEvent, snapshot: MasterySnapshot
    ) -> MasterySnapshot:
        if snapshot.source_attempt_id != attempt.attempt_id:
            raise ValueError("snapshot source_attempt_id must match the attempt")
        self._require_own_course(attempt.course_id, what="attempt")
        self._require_own_course(snapshot.course_id, what="mastery snapshot")

        try:
            with connection_scope(self._engine, write=True) as connection:
                self._insert_attempt(connection, attempt)
                return self._insert_mastery(connection, snapshot)
        except IntegrityError as exc:
            raise AttemptConflictError(str(exc)) from exc

    def get_mastery(self, learner_id: str, skill_id: str) -> MasterySnapshot | None:
        """Identity is learner_id + course_id (this repository's) + skill_id."""
        with connection_scope(self._engine, write=False) as connection:
            row = (
                connection.execute(
                    text(
                        """
                        SELECT * FROM mastery_snapshots
                        WHERE learner_id = :learner_id AND course_id = :course_id
                          AND skill_id = :skill_id
                        ORDER BY updated_at DESC, source_attempt_id DESC,
                                 model_version DESC, snapshot_id DESC
                        LIMIT 1
                        """
                    ),
                    {
                        "learner_id": learner_id,
                        "course_id": self.course_id,
                        "skill_id": skill_id,
                    },
                )
                .mappings()
                .first()
            )
        return self._mastery_from_row(row) if row is not None else None

    def get_mastery_for_attempt(self, attempt_id: str) -> MasterySnapshot | None:
        with connection_scope(self._engine, write=False) as connection:
            return self._select_mastery_for_attempt(connection, attempt_id)

    def list_mastery(self, *, learner_id: str | None = None) -> list[MasterySnapshot]:
        clauses = ["course_id = :course_id"]
        parameters: dict[str, str] = {"course_id": self.course_id}
        if learner_id is not None:
            clauses.append("learner_id = :learner_id")
            parameters["learner_id"] = learner_id
        where = f" WHERE {' AND '.join(clauses)}"
        with connection_scope(self._engine, write=False) as connection:
            rows = (
                connection.execute(
                    text(
                        "SELECT * FROM mastery_snapshots"
                        f"{where} ORDER BY updated_at, source_attempt_id, model_version, snapshot_id"
                    ),
                    parameters,
                )
                .mappings()
                .all()
            )
        return [self._mastery_from_row(row) for row in rows]

    def save_model_metadata(self, metadata: BKTModelMetadata) -> None:
        self._require_own_course(metadata.course_id, what="model metadata")
        with connection_scope(self._engine, write=True) as connection:
            existing = self._select_model_metadata(connection, metadata.model_version)
            if existing is not None:
                if existing == metadata:
                    return
                raise ValueError(
                    f"model_version {metadata.model_version!r} already exists "
                    f"for course {self.course_id!r}"
                )
            connection.execute(
                text(
                    """
                    INSERT INTO bkt_model_metadata (
                        model_version, course_id, fitted_at, training_attempt_count,
                        skill_ids
                    ) VALUES (
                        :model_version, :course_id, :fitted_at, :training_attempt_count,
                        :skill_ids
                    )
                    """
                ),
                {
                    "model_version": metadata.model_version,
                    "course_id": metadata.course_id,
                    "fitted_at": _utc_iso(metadata.fitted_at),
                    "training_attempt_count": metadata.training_attempt_count,
                    "skill_ids": json.dumps(metadata.skill_ids),
                },
            )

    def get_model_metadata(self, model_version: str) -> BKTModelMetadata | None:
        with connection_scope(self._engine, write=False) as connection:
            return self._select_model_metadata(connection, model_version)

    def _select_attempt(
        self, connection: Connection, attempt_id: str
    ) -> AttemptEvent | None:
        row = (
            connection.execute(
                text(
                    """
                    SELECT * FROM attempt_events
                    WHERE attempt_id = :attempt_id AND course_id = :course_id
                    """
                ),
                {"attempt_id": attempt_id, "course_id": self.course_id},
            )
            .mappings()
            .first()
        )
        return self._attempt_from_row(row) if row is not None else None

    def _select_mastery_for_attempt(
        self, connection: Connection, attempt_id: str
    ) -> MasterySnapshot | None:
        row = (
            connection.execute(
                text(
                    """
                    SELECT * FROM mastery_snapshots
                    WHERE source_attempt_id = :attempt_id AND course_id = :course_id
                    ORDER BY updated_at DESC, model_version DESC, snapshot_id DESC
                    LIMIT 1
                    """
                ),
                {"attempt_id": attempt_id, "course_id": self.course_id},
            )
            .mappings()
            .first()
        )
        return self._mastery_from_row(row) if row is not None else None

    def _select_model_metadata(
        self, connection: Connection, model_version: str
    ) -> BKTModelMetadata | None:
        row = (
            connection.execute(
                text(
                    """
                    SELECT * FROM bkt_model_metadata
                    WHERE model_version = :model_version AND course_id = :course_id
                    """
                ),
                {"model_version": model_version, "course_id": self.course_id},
            )
            .mappings()
            .first()
        )
        if row is None:
            return None
        return BKTModelMetadata(
            model_version=row["model_version"],
            course_id=row["course_id"],
            fitted_at=_aware_datetime(row["fitted_at"]),
            training_attempt_count=row["training_attempt_count"],
            skill_ids=json.loads(row["skill_ids"]),
        )

    def _insert_attempt(self, connection: Connection, attempt: AttemptEvent) -> AttemptEvent:
        """Atomic insert-or-return-existing in one round trip, replacing
        the old plain INSERT (which relied on a caller's own pre-check, or
        a raised IntegrityError, to catch a colliding attempt_id).
        ON CONFLICT (attempt_id) DO UPDATE with a no-op self-assignment is
        the portable idiom -- supported by both SQLite and PostgreSQL --
        that makes RETURNING yield a row unconditionally (fresh insert, or
        the pre-existing row completely untouched) instead of the zero
        rows ON CONFLICT DO NOTHING RETURNING would give on conflict. Also
        strictly safer than the old SELECT-then-INSERT pattern under
        concurrency: two simultaneous identical submissions could both
        pass a separate pre-check's SELECT before either INSERTed; this is
        atomic, so the database -- not application code racing against
        itself -- is what decides which write wins."""
        row = (
            connection.execute(
                text(
                    """
                    INSERT INTO attempt_events (
                        attempt_id, course_id, presentation_id, learner_id, item_id,
                        skill_id, selected_option_id, correct, attempt_order, occurred_at
                    ) VALUES (
                        :attempt_id, :course_id, :presentation_id, :learner_id, :item_id,
                        :skill_id, :selected_option_id, :correct, :attempt_order, :occurred_at
                    )
                    ON CONFLICT (attempt_id) DO UPDATE SET attempt_id = excluded.attempt_id
                    RETURNING *
                    """
                ),
                {
                    "attempt_id": attempt.attempt_id,
                    "course_id": attempt.course_id,
                    "presentation_id": attempt.presentation_id,
                    "learner_id": attempt.learner_id,
                    "item_id": attempt.item_id,
                    "skill_id": attempt.skill_id,
                    "selected_option_id": attempt.selected_option_id,
                    "correct": int(attempt.correct),
                    "attempt_order": attempt.attempt_order,
                    "occurred_at": _utc_iso(attempt.occurred_at),
                },
            )
            .mappings()
            .first()
        )
        resulting = self._attempt_from_row(row)
        if resulting != attempt:
            raise AttemptConflictError(
                f"attempt_id {attempt.attempt_id!r} is already used by a different attempt"
            )
        return resulting

    def _insert_mastery(
        self, connection: Connection, snapshot: MasterySnapshot
    ) -> MasterySnapshot:
        """Same atomic pattern as _insert_attempt (see its docstring),
        targeting mastery_snapshots' own (source_attempt_id, model_version)
        uniqueness. On conflict, the untouched pre-existing row is
        returned; if its content differs from what was submitted, that is
        a genuine conflict and raises exactly as the old SELECT-then-check
        code did."""
        row = (
            connection.execute(
                text(
                    """
                    INSERT INTO mastery_snapshots (
                        snapshot_id, course_id, learner_id, skill_id, mastery_probability,
                        attempt_count, source_attempt_id, model_version, updated_at
                    ) VALUES (
                        :snapshot_id, :course_id, :learner_id, :skill_id, :mastery_probability,
                        :attempt_count, :source_attempt_id, :model_version, :updated_at
                    )
                    ON CONFLICT (source_attempt_id, model_version)
                        DO UPDATE SET source_attempt_id = excluded.source_attempt_id
                    RETURNING *
                    """
                ),
                {
                    "snapshot_id": str(uuid4()),
                    "course_id": snapshot.course_id,
                    "learner_id": snapshot.learner_id,
                    "skill_id": snapshot.skill_id,
                    "mastery_probability": snapshot.mastery_probability,
                    "attempt_count": snapshot.attempt_count,
                    "source_attempt_id": snapshot.source_attempt_id,
                    "model_version": snapshot.model_version,
                    "updated_at": _utc_iso(snapshot.updated_at),
                },
            )
            .mappings()
            .first()
        )
        resulting = self._mastery_from_row(row)
        if (
            resulting.learner_id == snapshot.learner_id
            and resulting.skill_id == snapshot.skill_id
            and resulting.mastery_probability == snapshot.mastery_probability
            and resulting.attempt_count == snapshot.attempt_count
        ):
            return resulting
        raise ValueError(
            "mastery snapshot already exists for source attempt and model version"
        )

    @staticmethod
    def _attempt_from_row(row: RowMapping) -> AttemptEvent:
        return AttemptEvent(
            attempt_id=row["attempt_id"],
            course_id=row["course_id"],
            presentation_id=row["presentation_id"],
            learner_id=row["learner_id"],
            item_id=row["item_id"],
            skill_id=row["skill_id"],
            selected_option_id=row["selected_option_id"],
            correct=bool(row["correct"]),
            attempt_order=row["attempt_order"],
            occurred_at=_aware_datetime(row["occurred_at"]),
        )

    @staticmethod
    def _mastery_from_row(row: RowMapping) -> MasterySnapshot:
        return MasterySnapshot(
            learner_id=row["learner_id"],
            course_id=row["course_id"],
            skill_id=row["skill_id"],
            mastery_probability=row["mastery_probability"],
            attempt_count=row["attempt_count"],
            source_attempt_id=row["source_attempt_id"],
            model_version=row["model_version"],
            updated_at=_aware_datetime(row["updated_at"]),
        )
