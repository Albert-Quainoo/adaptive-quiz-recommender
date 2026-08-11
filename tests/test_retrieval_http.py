"""The two adapters that would touch the network, driven by a mock transport.

Narrow on purpose: these tests exist to pin down redirect handling, size and
content-type limits, and the shape of one provider's JSON. Everything above
them is tested against fakes, and nothing here opens a socket.
"""

from collections import Counter
from itertools import islice

import httpx
import pytest

from authoring.retrieval.brave import (
    API_KEY_VARIABLE,
    DOCUMENT_EXCLUSIONS,
    ENDPOINT,
    FIRST_RETRY_WAIT,
    MAX_PAGES_PER_STEP,
    MAX_RETRY_WAIT,
    RATE_LIMIT_RETRIES,
    RESULTS_PER_REQUEST,
    BraveSearchProvider,
    MissingCredentials,
)
from authoring.retrieval.diagnostics import (
    SEARCH_RATE_LIMITED,
    SEARCH_REQUEST_FAILED,
    SEARCH_RESPONSE_UNUSABLE,
    RetrievalDiagnostics,
    explain,
)
from authoring.retrieval.fetcher import HttpPageFetcher, visible_text
from authoring.retrieval.safety import (
    UNSUPPORTED_DOCUMENT_SUFFIXES,
    OversizedResponse,
    UnreadableSource,
    UnsafeSource,
    UnsupportedContentType,
)
from authoring.retrieval.search import RetrievalBudget, SearchStep

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


def results_for(
    count: int, domain: str = "aima.cs.berkeley.edu", more: bool = False, page: int = 0
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "query": {"more_results_available": more},
            "web": {
                "results": [
                    {
                        "title": f"Heuristics {index}",
                        "url": f"https://{domain}/page{page}-{index}.html",
                        "description": "A heuristic estimates cost.",
                    }
                    for index in range(count)
                ]
            },
        },
    )


def domain_asked(request: httpx.Request) -> str:
    """The domain a request was constrained to, read back off its query."""
    return request.url.params["q"].split()[0][len("site:") :]


def site_asked(request: httpx.Request) -> str:
    """The whole site: operator a request carried, path and all."""
    return request.url.params["q"].split()[0]


def search_with(
    handler,
    domains=ALLOWED,
    take=None,
    query="heuristics",
    budget=None,
    waits=None,
    schedule=None,
) -> tuple[list, RetrievalDiagnostics]:
    """Drain the provider's stream, or take only the first `take` results.

    Taking a few and walking away is what the retrieval loop does once its
    target is full, so it is also how these tests check that nothing further
    was requested.

    Waiting is recorded rather than done: a test that sleeps for a real
    second to prove it waited is a test nobody will keep running.

    The schedule defaults to one query angle across the given domains, which
    is the shape these tests were written against; pass one to say otherwise.
    build_search_schedule is what decides the real one, and is tested where it
    lives.
    """
    diagnostics = RetrievalDiagnostics()
    budget = budget if budget is not None else RetrievalBudget()
    provider = BraveSearchProvider(
        "test-key",
        client=client_for(handler),
        sleep=(waits if waits is not None else []).append,
    )

    if schedule is None:
        schedule = [SearchStep(query, domain) for domain in domains]

    stream = provider.search(schedule, diagnostics, budget)

    found = list(islice(stream, take)) if take is not None else list(stream)

    return [result for _, result in found], diagnostics


def test_the_search_carries_the_key_and_the_query():
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["token"] = request.headers["x-subscription-token"]
        seen["query"] = request.url.params["q"]
        seen["count"] = request.url.params["count"]
        seen["offset"] = request.url.params["offset"]

        return httpx.Response(200, json={"web": {"results": []}})

    search_with(handler)

    assert seen == {
        "token": "test-key",
        "query": f"site:aima.cs.berkeley.edu heuristics {DOCUMENT_EXCLUSIONS}",
        "count": str(RESULTS_PER_REQUEST),  # over-fetched, so rejects have replacements
        "offset": "0",
    }


def test_the_query_asks_the_index_to_leave_out_unreadable_documents():
    for suffix in UNSUPPORTED_DOCUMENT_SUFFIXES:
        assert f"-filetype:{suffix.lstrip('.')}" in DOCUMENT_EXCLUSIONS


def test_the_scheduled_domains_constrain_the_query():
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        queries.append(domain_asked(request))

        return httpx.Response(200, json={"web": {"results": []}})

    search_with(handler, domains=THREE_DOMAINS)

    assert queries == list(THREE_DOMAINS)


def test_steps_are_worked_in_the_order_the_schedule_sets():
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        queries.append(domain_asked(request))

        return httpx.Response(200, json={"web": {"results": []}})

    search_with(handler, domains=tuple(reversed(THREE_DOMAINS)))

    assert queries == list(reversed(THREE_DOMAINS))


def test_nothing_further_is_requested_once_the_caller_stops_taking():
    """The caller stops when its target is full; the provider stops with it."""
    queries: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        queries.append(domain_asked(request))

        return results_for(10, domain_asked(request), more=True)

    results, diagnostics = search_with(handler, domains=THREE_DOMAINS, take=4)

    assert len(results) == 4
    assert queries == list(THREE_DOMAINS[:2])  # two domains at two each
    assert diagnostics.search_requests_made == 2


def test_no_single_domain_supplies_every_result_while_others_are_untried():
    """One rich site took all twenty-one candidates of the first ordered run."""

    def handler(request: httpx.Request) -> httpx.Response:
        return results_for(10, domain_asked(request))

    results, _ = search_with(handler, domains=THREE_DOMAINS, take=6)
    domains = [result.url.split("/")[2] for result in results]

    # Two each, in priority order, before any of them is asked for a third.
    assert domains == [domain for domain in THREE_DOMAINS for _ in range(2)]


def test_a_domain_may_supply_everything_once_the_others_are_exhausted():
    def handler(request: httpx.Request) -> httpx.Response:
        domain = domain_asked(request)

        if domain == "aima.cs.berkeley.edu":
            return results_for(10, domain)

        return results_for(0, domain)

    results, diagnostics = search_with(handler, domains=THREE_DOMAINS, take=6)
    domains = {result.url.split("/")[2] for result in results}

    assert len(results) == 6
    assert domains == {"aima.cs.berkeley.edu"}
    assert diagnostics.domains_queried == list(THREE_DOMAINS)  # all three were asked


def test_a_thin_domain_falls_through_to_the_next():
    counts = {"aima.cs.berkeley.edu": 1, "ai.berkeley.edu": 1, "ocw.mit.edu": 3}

    def handler(request: httpx.Request) -> httpx.Response:
        return results_for(counts[domain_asked(request)], domain_asked(request))

    results, diagnostics = search_with(handler, domains=THREE_DOMAINS)

    assert len(results) == 5  # 1 + 1 + 3, over as many turns as it takes
    assert diagnostics.search_requests_made == 3
    assert diagnostics.domains_queried == list(THREE_DOMAINS)


def test_a_page_is_bought_once_and_handed_out_over_several_turns():
    """Over-fetching is the point: a reject is replaced without a new request."""

    def handler(request: httpx.Request) -> httpx.Response:
        return results_for(10, domain_asked(request))

    results, diagnostics = search_with(handler, domains=ALLOWED)

    assert len(results) == 10
    assert diagnostics.search_requests_made == 1


# Narrowing the request to a scope


CS50_AI_STEP = SearchStep("state space", "cs50.harvard.edu", "/ai/")


def sites_asked(handler_results):
    """Run one schedule and report the site: operator of every request made."""
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(site_asked(request))

        return handler_results(site_asked(request))

    return asked, handler


def test_a_scoped_step_asks_the_index_for_the_path():
    """One domain and two courses; only one of them is being asked about."""
    asked, handler = sites_asked(lambda site: results_for(2, "cs50.harvard.edu"))

    search_with(handler, schedule=[CS50_AI_STEP])

    assert asked[0] == "site:cs50.harvard.edu/ai/"


def test_an_unscoped_step_still_asks_for_the_whole_domain():
    asked, handler = sites_asked(lambda site: results_for(2))

    search_with(handler, schedule=[SearchStep("heuristics", "aima.cs.berkeley.edu")])

    assert asked == ["site:aima.cs.berkeley.edu"]


def test_a_scope_the_index_will_not_honour_is_asked_again_whole():
    """A path in site: is advisory, and a run that trusted it could find nothing.

    Nothing here can tell an unhonoured operator from a genuinely empty
    course, and it does not need to: both answers are the same request.
    """
    def answer(site: str) -> httpx.Response:
        if "/ai/" in site:
            return httpx.Response(200, json={"web": {"results": []}})

        return results_for(2, "cs50.harvard.edu")

    asked, handler = sites_asked(answer)
    results, _ = search_with(handler, schedule=[CS50_AI_STEP])

    assert asked == ["site:cs50.harvard.edu/ai/", "site:cs50.harvard.edu"]
    assert len(results) == 2


def test_a_scope_that_works_is_never_widened():
    asked, handler = sites_asked(lambda site: results_for(2, "cs50.harvard.edu"))

    search_with(handler, schedule=[CS50_AI_STEP])

    assert asked == ["site:cs50.harvard.edu/ai/"]


def test_one_domain_costs_one_wasted_request_however_often_it_is_asked():
    """Which is what keeps the ceiling a ceiling.

    Three angles on a scope the index ignores would be three wasted requests
    out of ten if each step found out for itself. The first one to find out
    tells the others.
    """
    def answer(site: str) -> httpx.Response:
        if "/ai/" in site:
            return httpx.Response(200, json={"web": {"results": []}})

        return results_for(1, "cs50.harvard.edu")

    schedule = [
        SearchStep(angle, "cs50.harvard.edu", "/ai/")
        for angle in ("state space", "search tree", "problem formulation")
    ]
    asked, handler = sites_asked(answer)

    search_with(handler, schedule=schedule)

    assert asked.count("site:cs50.harvard.edu/ai/") == 1
    assert asked.count("site:cs50.harvard.edu") == 3


def test_a_widened_step_that_finds_nothing_either_is_not_asked_again():
    """The fallback is one step down, not a loop."""
    asked, handler = sites_asked(
        lambda site: httpx.Response(200, json={"web": {"results": []}})
    )

    results, diagnostics = search_with(handler, schedule=[CS50_AI_STEP])

    assert results == []
    assert asked == ["site:cs50.harvard.edu/ai/", "site:cs50.harvard.edu"]
    assert diagnostics.search_requests_made == 2


def test_widening_cannot_outspend_the_request_budget():
    asked, handler = sites_asked(
        lambda site: httpx.Response(200, json={"web": {"results": []}})
    )
    schedule = [
        SearchStep("angle", f"domain{index}.edu", "/scope/") for index in range(6)
    ]
    budget = RetrievalBudget(max_requests=4)

    search_with(handler, schedule=schedule, budget=budget)

    assert len(asked) == 4
    assert budget.requests_made == 4


# Pagination


def test_a_second_page_is_asked_for_only_when_the_first_runs_out():
    offsets: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        offsets.append(request.url.params["offset"])

        return results_for(
            2, domain_asked(request), more=True, page=int(request.url.params["offset"])
        )

    results, diagnostics = search_with(handler, domains=ALLOWED, take=5)

    assert offsets == ["0", "1", "2"]
    assert len(results) == 5
    assert diagnostics.paginated_requests == 2


def test_pagination_stops_when_no_more_results_are_available():
    offsets: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        offsets.append(request.url.params["offset"])

        return results_for(2, domain_asked(request), more=False)

    results, diagnostics = search_with(handler, domains=ALLOWED, take=10)

    assert offsets == ["0"]  # the flag said there was nothing behind it
    assert len(results) == 2
    assert diagnostics.paginated_requests == 0


def test_a_page_that_promises_more_but_delivers_nothing_ends_the_domain():
    offsets: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        offsets.append(request.url.params["offset"])

        return results_for(0, domain_asked(request), more=True)

    results, _ = search_with(handler, domains=ALLOWED, take=10)

    assert results == []
    assert offsets == ["0"]


def test_pagination_stops_at_the_page_limit():
    def handler(request: httpx.Request) -> httpx.Response:
        return results_for(
            1, domain_asked(request), more=True, page=int(request.url.params["offset"])
        )

    results, diagnostics = search_with(handler, domains=ALLOWED, take=99)

    assert diagnostics.search_requests_made == MAX_PAGES_PER_STEP
    assert len(results) == MAX_PAGES_PER_STEP


def test_the_request_budget_stops_the_search():
    def handler(request: httpx.Request) -> httpx.Response:
        return results_for(
            1, domain_asked(request), more=True, page=int(request.url.params["offset"])
        )

    budget = RetrievalBudget(max_requests=2)
    _, diagnostics = search_with(
        handler, domains=THREE_DOMAINS, take=99, budget=budget
    )

    assert diagnostics.search_requests_made == 2
    assert budget.requests_made == 2
    assert not budget.may_request()


def test_the_same_url_is_never_yielded_twice():
    def handler(request: httpx.Request) -> httpx.Response:
        return results_for(2, "aima.cs.berkeley.edu")  # every domain answers alike

    results, _ = search_with(handler, domains=THREE_DOMAINS)

    assert len({result.url for result in results}) == len(results) == 2


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
        lambda request: httpx.Response(500), domains=THREE_DOMAINS
    )

    assert results == []
    assert diagnostics.search_requests_made == 3
    assert diagnostics.errors[SEARCH_REQUEST_FAILED] == 3


def test_one_failing_domain_does_not_lose_the_others():
    def handler(request: httpx.Request) -> httpx.Response:
        if "aima" in request.url.params["q"]:
            return httpx.Response(500)

        return results_for(2, "ai.berkeley.edu")

    results, diagnostics = search_with(handler, domains=THREE_DOMAINS)

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


# Being told to slow down


def throttling(times: int, retry_after: str | None = None):
    """A handler that returns 429 the first `times` times, then answers."""
    seen = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["count"] += 1

        if seen["count"] > times:
            return results_for(2, domain_asked(request))

        headers = {"retry-after": retry_after} if retry_after else {}

        return httpx.Response(429, headers=headers)

    return handler


def test_a_throttled_request_is_waited_out_and_retried():
    waits: list[float] = []
    results, diagnostics = search_with(throttling(1), waits=waits)

    assert len(results) == 2  # the retry got the page
    assert waits == [FIRST_RETRY_WAIT]
    assert diagnostics.rate_limit_retries == 1
    assert diagnostics.errors == Counter()


def test_the_wait_backs_off_between_retries():
    waits: list[float] = []
    search_with(throttling(2), waits=waits)

    assert waits == [FIRST_RETRY_WAIT, FIRST_RETRY_WAIT * 2]


def test_retry_after_is_respected_when_the_server_sends_one():
    waits: list[float] = []
    results, _ = search_with(throttling(1, retry_after="3"), waits=waits)

    assert waits == [3.0]
    assert len(results) == 2


def test_a_retry_after_a_server_could_hang_the_run_with_is_capped():
    waits: list[float] = []
    search_with(throttling(1, retry_after="86400"), waits=waits)

    assert waits == [MAX_RETRY_WAIT]


def test_an_unreadable_retry_after_falls_back_to_the_backoff():
    waits: list[float] = []
    search_with(throttling(1, retry_after="Wed, 05 Aug 2026 12:00:00 GMT"), waits=waits)

    assert waits == [FIRST_RETRY_WAIT]


def test_a_request_throttled_past_its_retries_is_given_up_on():
    waits: list[float] = []
    results, diagnostics = search_with(throttling(99), waits=waits)

    assert results == []
    assert len(waits) == RATE_LIMIT_RETRIES  # tried, then stopped trying
    assert diagnostics.errors[SEARCH_RATE_LIMITED] == 1
    assert diagnostics.errors[SEARCH_REQUEST_FAILED] == 0


def test_retries_do_not_spend_the_request_budget_twice():
    """The retries are what one request cost, not requests of their own."""
    budget = RetrievalBudget(max_requests=2)
    _, diagnostics = search_with(
        throttling(1), domains=THREE_DOMAINS, budget=budget
    )

    assert budget.requests_made == 2
    assert diagnostics.search_requests_made == 2


def test_a_throttled_run_says_so_rather_than_blaming_the_key():
    _, diagnostics = search_with(throttling(99), domains=THREE_DOMAINS)

    assert "rate limited" in explain(diagnostics)
    assert "API key" not in explain(diagnostics)


def test_no_test_can_reach_the_live_search_api():
    """The conftest guard, checked rather than taken on trust.

    Every other test here drives a mock transport, which is a promise about
    how the tests are written. This one is the promise that a test written
    the other way would fail loudly instead of quietly spending quota.
    """
    provider = BraveSearchProvider("test-key")  # a real client, no mock transport
    diagnostics = RetrievalDiagnostics()

    assert provider.endpoint == ENDPOINT
    assert ENDPOINT.startswith("https://api.search.brave.com/")

    with pytest.raises(RuntimeError, match="over the network"):
        provider.request("site:aima.cs.berkeley.edu heuristics", diagnostics)

    with pytest.raises(RuntimeError, match="over the network"):
        HttpPageFetcher(ALLOWED).fetch(PAGE)


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


def test_expandable_teaser_and_control_labels_are_not_reference_text():
    body = (
        "<p>Lead sentence.</p>"
        "<div>Truncated duplicate text … <button>Show more</button></div>"
        "<p>Full clean course description.</p>"
        "<button>Show less</button>"
    )

    cleaned = visible_text(body)

    assert "Truncated duplicate" not in cleaned
    assert "Show more" not in cleaned
    assert "Show less" not in cleaned
    assert "Full clean course description." in cleaned


def test_removed_inline_markup_does_not_leave_space_before_punctuation():
    cleaned = visible_text("<p>intelligent agents <span></span>, which act</p>")

    assert "agents ," not in cleaned
    assert "agents," in cleaned


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
