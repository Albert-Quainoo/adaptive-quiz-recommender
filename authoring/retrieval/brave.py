"""One concrete search provider: the Brave Search API.

Everything Brave-specific lives here - its endpoint, its header, the shape of
its JSON. The rest of the package only knows the SearchProvider protocol, so
swapping providers means adding a sibling of this file and nothing else.
"""

import os
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace

import httpx

from authoring.retrieval.diagnostics import (
    SEARCH_RATE_LIMITED,
    SEARCH_REQUEST_FAILED,
    SEARCH_RESPONSE_UNUSABLE,
    RetrievalDiagnostics,
)
from authoring.retrieval.models import SearchResult
from authoring.retrieval.safety import UNSUPPORTED_DOCUMENT_SUFFIXES
from authoring.retrieval.search import RetrievalBudget, SearchStep

ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
API_KEY_VARIABLE = "BRAVE_SEARCH_API_KEY"

DEFAULT_TIMEOUT = 10.0

# How many results one scheduled step may contribute before the next step is
# tried. A step is one domain, so this is what stops a rich site crowding
# every other source out of the review queue. It is a turn, not a quota: the
# step comes round again once the others have had theirs.
PER_STEP_LIMIT = 2

# Deliberately more than one turn's worth. Rejected results have to be
# replaced from somewhere, and replacing them out of a page already paid for
# is cheaper than asking for another.
RESULTS_PER_REQUEST = 10

# Brave pages by offset, counted in pages of `count` results, and stops
# accepting one above 9. Three is well inside that and is already 30 results
# deep on one query and one domain: past there the index is not answering the
# question.
MAX_PAGES_PER_STEP = 3

# Brave supports filetype:, so the documents this milestone cannot read can be
# asked out of the results instead of being discovered one wasted fetch at a
# time. It is a preference and not a boundary - the operator is advisory, the
# index is stale, and a URL says nothing binding about what a server will
# send - so is_unsupported_document still checks every URL that comes back,
# and the fetcher still checks every content type behind it.
DOCUMENT_EXCLUSIONS = " ".join(
    f"-filetype:{suffix.lstrip('.')}" for suffix in UNSUPPORTED_DOCUMENT_SUFFIXES
)

TOO_MANY_REQUESTS = 429

# Being told to slow down is not a failure, so it is worth waiting out - but
# only briefly and only twice. An unconditional delay on every request would
# tax the runs that were never throttled to spare the ones that were, and an
# unbounded wait hands a server the power to park the whole run.
RATE_LIMIT_RETRIES = 2
FIRST_RETRY_WAIT = 1.0
MAX_RETRY_WAIT = 5.0

MARKUP = re.compile(r"<[^>]*>")


class MissingCredentials(RuntimeError):
    pass


def plain(text: str) -> str:
    """Brave marks matched terms with <strong>; a title is not markup."""
    return " ".join(MARKUP.sub("", text).split())


def retry_wait(response: httpx.Response, attempt: int, cap: float) -> float:
    """How long to wait before asking again: what the server said, within reason.

    Retry-After is honoured because the server knows its own limit better
    than a doubling guess does, but it is capped either way - it arrives from
    off the network, and a run that sleeps for as long as it is told is a run
    a header can hang.

    Only the seconds form is read. The HTTP-date form is rare from search
    APIs, and falling back to the backoff is the same answer a missing header
    gets.
    """
    header = response.headers.get("retry-after", "").strip()

    if header.isdigit():
        return min(float(header), cap)

    return min(FIRST_RETRY_WAIT * 2**attempt, cap)


def query_for(step: SearchStep) -> str:
    """The query as the index is asked it: one place, ordinary pages.

    site: takes the path as well as the domain where the step has one. That
    is what stops a request meant for CS50's AI course being answered out of
    its introductory programming one, and a request meant for MIT 6.034 being
    answered out of electromagnetic field theory - both of which the live
    pilot paid for. Narrowing here is what makes the quota buy the course
    rather than the university.

    The operator is a preference and not a boundary, exactly as the domain
    form is: the index may ignore the path, serve a stale one, or know the
    page under a different host. search below notices when it has, and every
    URL that comes back is still checked against the allowlist.
    """
    return f"site:{step.domain}{step.path} {step.query} {DOCUMENT_EXCLUSIONS}"


@dataclass
class StepCursor:
    """How far a paged search through one scheduled step has got.

    Each step keeps its own place, so the search can hand out a couple of
    results from one site, move on to the next step, and come back for more
    later without asking the index the same question twice.
    """

    step: SearchStep
    offset: int = 0
    more: bool = True
    buffered: list[SearchResult] = field(default_factory=list)

    def may_page(self, max_pages: int) -> bool:
        return self.more and self.offset < max_pages

    def take(self, count: int) -> list[SearchResult]:
        taken, self.buffered = self.buffered[:count], self.buffered[count:]

        return taken

    def drew_a_blank(self) -> bool:
        """Did the very first page of this step come back with nothing?"""
        return self.offset == 1 and not self.buffered

    def widen(self) -> None:
        """Ask this step again for the whole domain, from its first page."""
        self.step = replace(self.step, path="")
        self.offset = 0
        self.more = True


class BraveSearchProvider:
    """Reads the web index. Never fetches the pages it names - that is the fetcher's job."""

    def __init__(
        self,
        api_key: str,
        client: httpx.Client | None = None,
        endpoint: str = ENDPOINT,
        timeout: float = DEFAULT_TIMEOUT,
        rate_limit_retries: int = RATE_LIMIT_RETRIES,
        max_retry_wait: float = MAX_RETRY_WAIT,
        sleep: Callable[[float], None] = time.sleep,
    ):
        if not api_key.strip():
            raise MissingCredentials(f"{API_KEY_VARIABLE} is empty.")

        self.api_key = api_key.strip()
        self.client = client or httpx.Client()
        self.endpoint = endpoint
        self.timeout = timeout
        self.rate_limit_retries = rate_limit_retries
        self.max_retry_wait = max_retry_wait
        # Injected so the tests can watch the run wait without waiting.
        self.sleep = sleep

    @classmethod
    def from_environment(
        cls,
        environ: Mapping[str, str] | None = None,
        client: httpx.Client | None = None,
    ) -> "BraveSearchProvider":
        environ = os.environ if environ is None else environ
        api_key = environ.get(API_KEY_VARIABLE, "")

        if not api_key.strip():
            raise MissingCredentials(
                f"{API_KEY_VARIABLE} is not set. Export a Brave Search API key, "
                f"or pass a different SearchProvider."
            )

        return cls(api_key, client=client)

    def request(
        self,
        query: str,
        diagnostics: RetrievalDiagnostics,
        domain: str = "",
        offset: int = 0,
        count: int = RESULTS_PER_REQUEST,
    ) -> tuple[list[SearchResult], bool]:
        """One result page, and whether Brave says there is another behind it.

        Failures are counted by category, never by message: a provider
        message can carry back whatever a server chose to send. A failure
        also reports no further pages - there is no sense paging past a
        request that did not work.

        Being throttled is the one failure worth retrying, so a 429 is waited
        out rather than counted, up to rate_limit_retries times. One request
        is what the budget is charged either way: the retries are the cost of
        that request, not extra requests to spend.
        """
        for attempt in range(self.rate_limit_retries + 1):
            try:
                response = self.client.get(
                    self.endpoint,
                    params={"q": query, "count": count, "offset": offset},
                    headers={
                        "Accept": "application/json",
                        "X-Subscription-Token": self.api_key,
                    },
                    timeout=self.timeout,
                )
            except httpx.HTTPError:
                diagnostics.record_error(SEARCH_REQUEST_FAILED, domain)

                return [], False

            if response.status_code != TOO_MANY_REQUESTS:
                break

            if attempt == self.rate_limit_retries:
                diagnostics.record_error(SEARCH_RATE_LIMITED, domain)

                return [], False

            diagnostics.rate_limit_retries += 1
            self.sleep(retry_wait(response, attempt, self.max_retry_wait))

        try:
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError:
            diagnostics.record_error(SEARCH_REQUEST_FAILED, domain)

            return [], False
        except ValueError:
            diagnostics.record_error(SEARCH_RESPONSE_UNUSABLE, domain)

            return [], False

        results = []

        for entry in payload.get("web", {}).get("results", []):
            url = entry.get("url", "")
            title = plain(entry.get("title", ""))

            if not url or not title:
                continue

            results.append(
                SearchResult(
                    title=title,
                    url=url,
                    snippet=plain(entry.get("description", "")),
                )
            )

        more = bool(payload.get("query", {}).get("more_results_available"))

        # A page that came back empty ends the domain whatever the flag says,
        # so a provider promising more it will not deliver cannot spend the
        # rest of the budget on empty pages.
        return results, more and bool(results)

    def load(
        self,
        cursor: StepCursor,
        diagnostics: RetrievalDiagnostics,
        budget: RetrievalBudget,
    ) -> None:
        """Buy one more result page for this step, and record the spend."""
        budget.spend_request()
        diagnostics.record_query(cursor.step.domain, cursor.offset)

        results, more = self.request(
            query_for(cursor.step),
            diagnostics,
            cursor.step.domain,
            offset=cursor.offset,
        )

        cursor.buffered.extend(results)
        cursor.more = more
        cursor.offset += 1

    def search(
        self,
        schedule: Sequence[SearchStep],
        diagnostics: RetrievalDiagnostics,
        budget: RetrievalBudget,
        per_step: int = PER_STEP_LIMIT,
        max_pages: int = MAX_PAGES_PER_STEP,
    ) -> Iterator[tuple[SearchStep, SearchResult]]:
        """Take turns along the schedule, paging deeper each time round.

        Brave indexes the whole web, so an unconstrained query spends the
        run's quota on pages the allowlist will throw away - which is exactly
        how a working API key produced no candidates at all. site: asks the
        index the question the allowlist is going to ask anyway.

        Steps are tried in the order the schedule sets, but only per_step
        results at a time: the first ordered run took all twenty-one
        candidates from one site. Coming round again is what makes the order
        a preference rather than a monopoly - one domain can still supply
        everything, but only once the rest have been asked and had nothing to
        offer.

        Nothing is requested until the caller asks for the result it would
        hold, so a caller that has what it needs pays for no further page,
        and a schedule is a ceiling on what may be asked rather than a
        promise to ask it.

        A scoped step whose first page comes back empty is asked again for
        the whole domain, and every other step on that domain is widened with
        it. A path in site: is advisory - the index may not honour it, or may
        hold the pages under a host the scope does not name - and a run that
        found nothing because of that would be worse than one that searched
        too broadly. Widening costs one wasted request per domain, whatever
        the schedule asks of it, so the ceiling still holds.
        """
        cursors = [StepCursor(step) for step in schedule]
        seen: set[str] = set()
        widened: set[str] = set()

        while True:
            working = False

            for cursor in cursors:
                if not cursor.buffered:
                    if not cursor.may_page(max_pages) or not budget.may_request():
                        continue

                    if cursor.step.path and cursor.step.domain in widened:
                        cursor.widen()

                    self.load(cursor, diagnostics, budget)

                    # The path bought nothing. Widening leaves this step
                    # ready to be asked again, so there is still work here
                    # even though it has no result to offer this time round.
                    if cursor.step.path and cursor.drew_a_blank():
                        widened.add(cursor.step.domain)
                        cursor.widen()
                        working = True

                        continue

                for result in cursor.take(per_step):
                    working = True

                    if result.url in seen:
                        continue

                    seen.add(result.url)

                    yield cursor.step, result

            # Every step is either paged out, empty or unaffordable.
            if not working:
                return
