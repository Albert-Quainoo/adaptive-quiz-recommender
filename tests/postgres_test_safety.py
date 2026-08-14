"""Shared guardrails for the PostgreSQL integration tests.

test_postgres_adaptive_repository.py and test_postgres_replenishment_jobs.py
both run `DROP TABLE ... CASCADE` against whatever QUIZ_TEST_POSTGRES_URL
points at, once per test, via an autouse fixture. That is exactly right for
the disposable postgres:16 service container GitHub Actions spins up -- and
exactly wrong for any DSN that happens to resolve to a shared or production
database. Two independent, explicit opt-ins are required before a single
DROP runs:

1. REMOTE_HOST_ENV_VAR, only if the DSN's host isn't local -- so a DSN typo'd
   or copy-pasted from somewhere real fails fast instead of silently
   proceeding.
2. DESTRUCTIVE_CONFIRMATION_ENV_VAR, always -- so setting the DSN alone,
   which is easy to do absent-mindedly, is never sufficient on its own to
   drop tables.
"""

import os
from urllib.parse import urlsplit

DSN_ENV_VAR = "QUIZ_TEST_POSTGRES_URL"
REMOTE_HOST_ENV_VAR = "QUIZ_ALLOW_REMOTE_POSTGRES_TESTS"
DESTRUCTIVE_CONFIRMATION_ENV_VAR = "QUIZ_CONFIRM_DESTRUCTIVE_POSTGRES_TESTS"

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def require_safe_postgres_target(dsn: str) -> None:
    host = urlsplit(dsn).hostname
    if host not in _LOCAL_HOSTS and os.getenv(REMOTE_HOST_ENV_VAR) != "1":
        raise RuntimeError(
            f"{DSN_ENV_VAR} points at a non-local host ({host!r}). These tests "
            f"run DROP TABLE ... CASCADE. Set {REMOTE_HOST_ENV_VAR}=1 only if "
            "you are certain this DSN is a disposable database, never a "
            "shared or production one."
        )
    if os.getenv(DESTRUCTIVE_CONFIRMATION_ENV_VAR) != "1":
        raise RuntimeError(
            f"Set {DESTRUCTIVE_CONFIRMATION_ENV_VAR}=1 to confirm you want "
            f"these tests to DROP TABLE ... CASCADE against {DSN_ENV_VAR}."
        )
