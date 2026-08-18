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


@pytest.fixture(autouse=True)
def no_default_nli_model(monkeypatch):
    """authoring/review/equivalence_gate.py's option-equivalence gate runs on every
    review_candidate() call that passes deterministic checks -- including in the
    hundreds of existing tests in this suite that know nothing about it. Its NLI
    detector's default scorer (authoring.review.equivalence_nli.get_default_scorer)
    lazily downloads a real ~90MB ONNX model from Hugging Face Hub on first use, which
    this suite must never do implicitly (same reasoning as no_live_requests above).

    Replaced here with authoring.review.equivalence_nli.FakeNliScorer's default
    behavior (low entailment / high neutral for anything unconfigured) -- every
    existing test's equivalence-gate evidence is deterministically "not_equivalent"/
    "not_applicable", never spuriously escalating a candidate it isn't testing. A test
    that specifically exercises the equivalence gate injects its own FakeNliScorer
    (with configured scores) or review_candidate(equivalence_nli_scorer=...) directly,
    which bypasses get_default_scorer() entirely and is unaffected by this fixture. A
    test that specifically needs the real pinned model constructs
    authoring.review.equivalence_nli.NliScorer() itself and is also unaffected.
    """
    from authoring.review.equivalence_nli import FakeNliScorer

    # Patches the module-level singleton state itself, not the get_default_scorer()
    # function object: authoring/review/equivalence_gate.py does
    # `from authoring.review.equivalence_nli import get_default_scorer`, which binds
    # its own local name to the function at import time -- patching
    # equivalence_nli.get_default_scorer afterward would not affect that already-bound
    # reference. get_default_scorer()'s body reads the module global at call time, so
    # pre-seeding it here is intercepted regardless of which module calls the function.
    monkeypatch.setattr("authoring.review.equivalence_nli._default_scorer", FakeNliScorer())
