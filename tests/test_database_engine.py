"""Unit coverage for database.py's dialect dispatch.

Engine creation is lazy (no real connection attempt happens until first
use), so these run everywhere without a live database.
"""

from database import create_engine_for, is_postgres_dsn


def test_is_postgres_dsn_recognizes_provider_schemes():
    assert is_postgres_dsn("postgres://user:pw@host/db")
    assert is_postgres_dsn("postgresql://user:pw@host/db")
    assert is_postgres_dsn("postgresql+psycopg://user:pw@host/db")
    assert not is_postgres_dsn("sqlite:///data/adaptive_quiz.sqlite3")
    assert not is_postgres_dsn("data/adaptive_quiz.sqlite3")


def test_bare_postgres_schemes_are_routed_to_the_installed_psycopg_driver():
    """Providers (Supabase included) hand out bare postgres://.../postgresql://
    DSNs. SQLAlchemy's default driver for that scheme is psycopg2, which this
    project does not install -- only psycopg (v3) is -- so a bare scheme must
    be normalized or engine creation succeeds but the first real connection
    fails with ModuleNotFoundError: No module named 'psycopg2'."""
    for dsn in (
        "postgres://user:pw@host:5432/db",
        "postgresql://user:pw@host:5432/db",
    ):
        engine = create_engine_for(dsn)
        assert str(engine.url).startswith("postgresql+psycopg://")


def test_a_dsn_that_already_names_the_driver_is_left_unchanged():
    engine = create_engine_for("postgresql+psycopg://user:pw@host:5432/db")
    assert str(engine.url).startswith("postgresql+psycopg://")
