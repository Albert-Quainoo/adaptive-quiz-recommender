"""Unit coverage for the guardrails the PostgreSQL integration tests rely on
before they DROP TABLE ... CASCADE against QUIZ_TEST_POSTGRES_URL."""

import pytest

from tests.postgres_test_safety import (
    DESTRUCTIVE_CONFIRMATION_ENV_VAR,
    REMOTE_HOST_ENV_VAR,
    require_safe_postgres_target,
)

LOCAL_DSN = "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
REMOTE_DSN = "postgresql+psycopg://user:pw@db.example-project.supabase.co:5432/postgres"


def test_a_local_host_still_requires_the_destructive_confirmation_flag(monkeypatch):
    monkeypatch.delenv(DESTRUCTIVE_CONFIRMATION_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match=DESTRUCTIVE_CONFIRMATION_ENV_VAR):
        require_safe_postgres_target(LOCAL_DSN)


def test_a_local_host_with_confirmation_is_allowed(monkeypatch):
    monkeypatch.setenv(DESTRUCTIVE_CONFIRMATION_ENV_VAR, "1")
    require_safe_postgres_target(LOCAL_DSN)  # must not raise


def test_a_remote_host_is_refused_even_with_destructive_confirmation(monkeypatch):
    monkeypatch.setenv(DESTRUCTIVE_CONFIRMATION_ENV_VAR, "1")
    monkeypatch.delenv(REMOTE_HOST_ENV_VAR, raising=False)
    with pytest.raises(RuntimeError, match=REMOTE_HOST_ENV_VAR):
        require_safe_postgres_target(REMOTE_DSN)


def test_a_remote_host_needs_both_flags_to_be_allowed(monkeypatch):
    monkeypatch.setenv(REMOTE_HOST_ENV_VAR, "1")
    monkeypatch.setenv(DESTRUCTIVE_CONFIRMATION_ENV_VAR, "1")
    require_safe_postgres_target(REMOTE_DSN)  # must not raise
