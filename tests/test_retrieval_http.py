"""The two adapters that would touch the network, driven by a mock transport.

Narrow on purpose: these tests exist to pin down redirect handling, size and
content-type limits, and the shape of one provider's JSON. Everything above
them is tested against fakes, and nothing here opens a socket.
"""

import httpx
import pytest

from authoring.retrieval.brave import (
    API_KEY_VARIABLE,
    BraveSearchProvider,
    MissingCredentials,
)
from authoring.retrieval.diagnostics import (
    SEARCH_REQUEST_FAILED,
    SEARCH_RESPONSE_UNUSABLE,
    RetrievalDiagnostics,
)
from authoring.retrieval.fetcher import HttpPageFetcher, visible_text
from authoring.retrieval.safety import (
    OversizedResponse,
    UnreadableSource,
    UnsafeSource,
    UnsupportedContentType,
)

ALLOWED = ("aima.cs.berkeley.edu",)

THREE_DOMAINS = ("aima.cs.berkeley.edu", "ai.berkeley.edu", "ocw.mit.edu")

PAGE = "https://aima.cs.berkeley.edu/heuristics.html"


def client_for(handler, requested: list[str] | None = None) -> httpx.Client:
    def record(request: httpx.Request) -> httpx.Response:
        if requested is not None:
            requested.append(str(request.url))

        return handler(request)

    return httpx.Client(transport=httpx.MockTransport(record), follow_redirects=False)


def serving(responses: dict[str, httpx.Response], requested: list[str] | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        return responses[str(request.url)]

    return client_for(handler, requested)


def html(body: str) -> httpx.Response:
    return httpx.Response(200, text=body, headers={"content-type": "text/html"})


def redirect_to(location: str) -> httpx.Response:
    return httpx.Response(302, headers={"location": location})


# The Brave adapter


def test_a_missing_api_key_names_the_variable():
    with pytest.raises(MissingCredentials, match=API_KEY_VARIABLE):
        BraveSearchProvider.from_environment(environ={})


def test_a_blank_api_key_is_treated_as_missing():
    with pytest.raises(MissingCredentials, match=API_KEY_VARIABLE):
        BraveSearchProvider.from_environment(environ={API_KEY_VARIABLE: "   "})


def test_the_key_is_read_from_the_environment():
    provider = BraveSearchProvider.from_environment(
        environ={API_KEY_VARIABLE: "test-key"}
    )

    assert provider.api_key == "test-key"


def results_for(count: int, domain: str = "aima.cs.berkeley.edu") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "web": {
                "results": [
                    {
                        "title": f"Heuristics {index}",
                        "url": f"https://{domain}/page{index}.html",
                        "description": "A heuristic estimates cost.",
                    }
                    for index in range(count)
                ]
            }
        },
    )


def search_with(handler, domains=ALLOWED, limit=5, query="heuristics"):
    diagnostics = RetrievalDiagnostics()
    results = BraveSearchProvider("test-key", client=client_for(handler)).search(
        query, limit, domains, diagnostics
    )

    return results, diagnostics


def test_the_search_carries_the_key_and_the_query():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["token"] = request.headers["x-subscription-token"]
        seen["query"] = request.url.params["q"]
        seen["count"] = request.url.params["count"]

        return httpx.Response(200, json={"web": {"results": []}})

    search_with(handler, limit=3)

    assert seen == {
        "token": "test-key",
        "query": "site:aima.cs.berkeley.edu heuristics",
        "count": "2",  # the per-domain cap, not the whole limit
    }


def test_the_allowed_domains_constrain_the_query():
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        queries.append(request.url.params["q"])

        return httpx.Response(200, json={"web": {"results": []}})

    search_with(handler, domains=THREE_DOMAINS)

    assert queries == [
        "site:aima.cs.berkeley.edu heuristics",
        "site:ai.berkeley.edu heuristics",
        "site:ocw.mit.edu heuristics",
    ]


def test_domains_are_tried_in_the_order_given():
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        queries.append(request.url.params["q"])

        return httpx.Response(200, json={"web": {"results": []}})

    search_with(handler, domains=tuple(reversed(THREE_DOMAINS)))

    assert queries == [
        "site:ocw.mit.edu heuristics",
        "site:ai.berkeley.edu heuristics",
        "site:aima.cs.berkeley.edu heuristics",
    ]


def test_the_search_stops_once_the_limit_is_filled():
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        queries.append(request.url.params["q"])

        return results_for(5, request.url.params["q"].split()[0][5:])

    results, diagnostics = search_with(handler, domains=THREE_DOMAINS, limit=4)

    assert len(results) == 4
    assert len(queries) == 2  # two domains at two each; the third is untouched
    assert diagnostics.domains_queried == list(THREE_DOMAINS[:2])


def test_no_single_domain_supplies_the_whole_limit():
    """One rich site took all twenty-one candidates of the first ordered run."""

    def handler(request: httpx.Request) -> httpx.Response:
        return results_for(5, request.url.params["q"].split()[0][5:])

    results, _ = search_with(handler, domains=THREE_DOMAINS, limit=5)
    domains = {result.url.split("/")[2] for result in results}

    assert len(results) == 5
    assert domains == set(THREE_DOMAINS)


def test_a_thin_domain_falls_through_to_the_next():
    counts = iter([1, 1, 3])

    def handler(request: httpx.Request) -> httpx.Response:
        return results_for(next(counts), request.url.params["q"].split()[0][5:])

    results, diagnostics = search_with(handler, domains=THREE_DOMAINS, limit=5)

    assert len(results) == 4  # 1 + 1 + the two the cap allows
    assert diagnostics.search_requests_made == 3
    assert diagnostics.domains_queried == list(THREE_DOMAINS)


def test_only_the_shortfall_is_asked_for():
    counts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        counts.append(request.url.params["count"])

        return results_for(2, request.url.params["q"].split()[0][5:])

    search_with(handler, domains=THREE_DOMAINS, limit=5)

    assert counts == ["2", "2", "1"]  # the cap, then the shortfall


def test_provider_results_become_search_results_without_markup():
    payload = {
        "web": {
            "results": [
                {
                    "title": "<strong>Heuristic</strong> function",
                    "url": PAGE,
                    "description": "A <strong>heuristic</strong> estimates cost.",
                },
                {"title": "No url", "description": "dropped"},
            ]
        }
    }

    results, _ = search_with(lambda request: httpx.Response(200, json=payload))

    assert len(results) == 1
    assert results[0].title == "Heuristic function"
    assert results[0].snippet == "A heuristic estimates cost."
    assert results[0].url == PAGE


def test_a_failed_search_is_counted_rather_than_raised():
    results, diagnostics = search_with(
        lambda request: httpx.Response(429), domains=THREE_DOMAINS
    )

    assert results == []
    assert diagnostics.search_requests_made == 3
    assert diagnostics.errors[SEARCH_REQUEST_FAILED] == 3


def test_one_failing_domain_does_not_lose_the_others():
    def handler(request: httpx.Request) -> httpx.Response:
        if "aima" in request.url.params["q"]:
            return httpx.Response(500)

        return results_for(2, "ai.berkeley.edu")

    results, diagnostics = search_with(handler, domains=THREE_DOMAINS, limit=5)

    assert len(results) == 2
    assert diagnostics.errors[SEARCH_REQUEST_FAILED] == 1


def test_unusable_json_is_counted_rather_than_raised():
    results, diagnostics = search_with(
        lambda request: httpx.Response(200, text="not json")
    )

    assert results == []
    assert diagnostics.errors[SEARCH_RESPONSE_UNUSABLE] == 1


def test_a_search_failure_records_no_message_from_the_server():
    _, diagnostics = search_with(
        lambda request: httpx.Response(403, text="key sk-live-secret is invalid")
    )

    assert list(diagnostics.errors) == [SEARCH_REQUEST_FAILED]
    assert "secret" not in str(diagnostics.errors)


# The page fetcher


def test_a_page_comes_back_as_visible_text():
    fetcher = HttpPageFetcher(
        ALLOWED,
        client=serving(
            {PAGE: html("<html><body><h1>Heuristics</h1><p>Estimate cost.</p></body></html>")}
        ),
    )

    page = fetcher.fetch(PAGE)

    assert "Heuristics" in page.text
    assert "Estimate cost." in page.text
    assert "<" not in page.text
    assert page.redirects == ()


def test_scripts_and_styles_are_not_page_text():
    assert "alert" not in visible_text("<script>alert(1)</script><p>Real text</p>")
    assert "Real text" in visible_text("<script>alert(1)</script><p>Real text</p>")


def test_a_redirect_inside_the_allowlist_is_followed_and_recorded():
    target = "https://aima.cs.berkeley.edu/final.html"
    fetcher = HttpPageFetcher(
        ALLOWED, client=serving({PAGE: redirect_to(target), target: html("<p>Cost.</p>")})
    )

    page = fetcher.fetch(PAGE)

    assert page.url == target
    assert page.redirects == (PAGE,)


def test_a_relative_redirect_resolves_against_the_current_hop():
    target = "https://aima.cs.berkeley.edu/final.html"
    fetcher = HttpPageFetcher(
        ALLOWED,
        client=serving({PAGE: redirect_to("/final.html"), target: html("<p>Cost.</p>")}),
    )

    assert fetcher.fetch(PAGE).url == target


def test_a_redirect_off_the_allowlist_is_never_requested():
    requested: list[str] = []
    fetcher = HttpPageFetcher(
        ALLOWED,
        client=serving({PAGE: redirect_to("https://example.com/x")}, requested),
    )

    with pytest.raises(UnsafeSource, match="outside the allowed domains"):
        fetcher.fetch(PAGE)

    assert requested == [PAGE]


def test_a_redirect_into_the_private_network_is_never_requested():
    requested: list[str] = []
    fetcher = HttpPageFetcher(
        ALLOWED,
        client=serving({PAGE: redirect_to("http://169.254.169.254/latest")}, requested),
    )

    with pytest.raises(UnsafeSource, match="not a public address"):
        fetcher.fetch(PAGE)

    assert requested == [PAGE]


def test_a_redirect_chain_stops_at_the_limit():
    hops = [f"https://aima.cs.berkeley.edu/{index}" for index in range(6)]
    responses = {url: redirect_to(hops[index + 1]) for index, url in enumerate(hops[:-1])}
    responses[hops[-1]] = html("<p>Cost.</p>")

    fetcher = HttpPageFetcher(ALLOWED, client=serving(responses), max_redirects=2)

    with pytest.raises(UnsafeSource, match="redirected more than 2 times"):
        fetcher.fetch(hops[0])


def test_a_redirect_without_a_location_is_refused():
    fetcher = HttpPageFetcher(ALLOWED, client=serving({PAGE: httpx.Response(302)}))

    with pytest.raises(UnsafeSource, match="without a location"):
        fetcher.fetch(PAGE)


def test_a_declared_oversized_body_is_refused_before_reading():
    response = httpx.Response(
        200,
        text="x" * 50,
        headers={"content-type": "text/plain", "content-length": "999999"},
    )
    fetcher = HttpPageFetcher(ALLOWED, client=serving({PAGE: response}), max_bytes=100)

    with pytest.raises(OversizedResponse, match="declares more than 100 bytes"):
        fetcher.fetch(PAGE)


def test_a_body_that_outgrows_the_limit_while_reading_is_refused():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=iter([b"x" * 80, b"y" * 80]),
            headers={"content-type": "text/plain"},
        )

    fetcher = HttpPageFetcher(ALLOWED, client=client_for(handler), max_bytes=100)

    with pytest.raises(OversizedResponse, match="returned more than 100 bytes"):
        fetcher.fetch(PAGE)


@pytest.mark.parametrize(
    "content_type", ["application/pdf", "image/png", "application/json", ""]
)
def test_a_non_textual_body_is_refused(content_type):
    response = httpx.Response(200, content=b"...", headers={"content-type": content_type})
    fetcher = HttpPageFetcher(ALLOWED, client=serving({PAGE: response}))

    with pytest.raises(UnsupportedContentType, match="not readable text"):
        fetcher.fetch(PAGE)


def test_plain_text_is_readable():
    response = httpx.Response(200, text="Estimate cost.", headers={"content-type": "text/plain"})
    fetcher = HttpPageFetcher(ALLOWED, client=serving({PAGE: response}))

    assert fetcher.fetch(PAGE).text.strip() == "Estimate cost."


def test_an_error_status_is_unreadable():
    fetcher = HttpPageFetcher(ALLOWED, client=serving({PAGE: httpx.Response(404)}))

    with pytest.raises(UnreadableSource, match="status 404"):
        fetcher.fetch(PAGE)


def test_a_timeout_is_unreadable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("too slow", request=request)

    fetcher = HttpPageFetcher(ALLOWED, client=client_for(handler))

    with pytest.raises(UnreadableSource, match="timed out"):
        fetcher.fetch(PAGE)


def test_a_failure_carries_a_countable_category():
    """Eight failures with no cause was the gap the third live run exposed."""
    cases = {
        httpx.Response(404): "fetch_failed_404",
        httpx.Response(503): "fetch_failed_5xx",
        httpx.Response(429): "fetch_failed_429",
        httpx.Response(200, content=b"%PDF", headers={"content-type": "application/pdf"}):
            "unsupported_pdf",
    }

    for response, category in cases.items():
        fetcher = HttpPageFetcher(ALLOWED, client=serving({PAGE: response}))

        with pytest.raises(UnsafeSource) as raised:
            fetcher.fetch(PAGE)

        assert raised.value.category == category


def test_a_timeout_and_a_refused_connection_are_told_apart():
    def timing_out(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("too slow", request=request)

    def refusing(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    for handler, category in ((timing_out, "fetch_failed_timeout"),
                              (refusing, "fetch_failed_connection")):
        with pytest.raises(UnsafeSource) as raised:
            HttpPageFetcher(ALLOWED, client=client_for(handler)).fetch(PAGE)

        assert raised.value.category == category


def test_a_category_never_carries_server_text():
    response = httpx.Response(403, text="key sk-live-secret rejected for user bob")
    fetcher = HttpPageFetcher(ALLOWED, client=serving({PAGE: response}))

    with pytest.raises(UnsafeSource) as raised:
        fetcher.fetch(PAGE)

    assert raised.value.category == "fetch_failed_403"
    assert "secret" not in raised.value.category
    assert "bob" not in raised.value.category


def test_entity_encoded_markup_does_not_survive_as_page_text():
    """`data-bs-toggle="collapse">` reached a passage in the third live run."""
    body = "<p>Real text</p> &lt;div class=&quot;nav&quot; data-bs-toggle=&quot;collapse&quot;&gt; Menu"

    assert "data-bs-toggle" not in visible_text(body)
    assert "Real text" in visible_text(body)
    assert "Menu" in visible_text(body)


def test_comparisons_in_prose_are_not_eaten_as_tags():
    assert "5 and y" in visible_text("<p>if x &lt; 5 and y &gt; 3 then stop</p>")


def test_an_unsafe_first_url_is_never_requested():
    requested: list[str] = []
    fetcher = HttpPageFetcher(ALLOWED, client=serving({}, requested))

    with pytest.raises(UnsafeSource):
        fetcher.fetch("http://127.0.0.1/x")

    assert requested == []
