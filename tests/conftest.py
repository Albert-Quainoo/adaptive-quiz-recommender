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


@pytest.fixture(autouse=True)
def no_live_requests(monkeypatch):
    def refuse(self, request: httpx.Request) -> httpx.Response:
        raise RuntimeError(
            f"the test suite tried to reach {request.url.host} over the network."
        )

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
