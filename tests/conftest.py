"""No test in this suite talks to a real service.

Retrieval is the one part of this project that would otherwise reach the
network - a live Brave request costs quota and a live page fetch reads
whatever the web is serving today - so the transport that would carry a real
request is replaced for every test. Mock transports are untouched: they never
reach it.

A test that needs the network is a test that has to say so, by unpatching
here deliberately rather than by drifting into it.
"""

import httpx
import pytest
import streamlit


@pytest.fixture(autouse=True)
def no_live_requests(monkeypatch):
    def refuse(self, request: httpx.Request) -> httpx.Response:
        raise RuntimeError(
            f"the test suite tried to reach {request.url.host} over the network."
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)


@pytest.fixture(autouse=True)
def no_real_secrets(monkeypatch):
    """The real .streamlit/secrets.toml (git-ignored, present on developer and
    CI machines that have configured production access) carries the live
    Supabase QUIZ_DATABASE_URL. app/main.py reads it via `dict(st.secrets)`,
    and `AppSettings.from_sources` falls back to it whenever QUIZ_DATABASE_URL
    is unset in the environment -- which every AppTest-based test in this
    suite deliberately leaves unset, expecting the SQLite path they pass via
    QUIZ_DATABASE_PATH to win. Replacing the `streamlit.secrets` singleton
    with an empty dict here means that file is never opened by any test, so
    no test can silently start writing to production no matter what secrets
    happen to be configured on the machine running it.
    """
    monkeypatch.setattr(streamlit, "secrets", {})
