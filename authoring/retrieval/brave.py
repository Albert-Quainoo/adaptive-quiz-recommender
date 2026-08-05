"""One concrete search provider: the Brave Search API.

Everything Brave-specific lives here - its endpoint, its header, the shape of
its JSON. The rest of the package only knows the SearchProvider protocol, so
swapping providers means adding a sibling of this file and nothing else.
"""

import os
import re
from collections.abc import Mapping, Sequence

import httpx

from authoring.retrieval.diagnostics import (
    SEARCH_REQUEST_FAILED,
    SEARCH_RESPONSE_UNUSABLE,
    RetrievalDiagnostics,
)
from authoring.retrieval.models import SearchResult

ENDPOINT = "https://api.search.brave.com/res/v1/web/search"
API_KEY_VARIABLE = "BRAVE_SEARCH_API_KEY"

DEFAULT_TIMEOUT = 10.0

# How many results one domain may contribute to a single query, so that a
# rich site cannot crowd every other source out of the review queue.
PER_DOMAIN_LIMIT = 2

MARKUP = re.compile(r"<[^>]*>")


class MissingCredentials(RuntimeError):
    pass


def plain(text: str) -> str:
    """Brave marks matched terms with <strong>; a title is not markup."""
    return " ".join(MARKUP.sub("", text).split())


class BraveSearchProvider:
    """Reads the web index. Never fetches the pages it names - that is the fetcher's job."""

    def __init__(
        self,
        api_key: str,
        client: httpx.Client | None = None,
        endpoint: str = ENDPOINT,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        if not api_key.strip():
            raise MissingCredentials(f"{API_KEY_VARIABLE} is empty.")

        self.api_key = api_key.strip()
        self.client = client or httpx.Client()
        self.endpoint = endpoint
        self.timeout = timeout

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
        limit: int,
        diagnostics: RetrievalDiagnostics,
        domain: str = "",
    ) -> list[SearchResult]:
        """One search request. Failures are counted by category, never by message.

        A provider message can carry back whatever a server chose to send, so
        only fixed category names are recorded.
        """
        try:
            response = self.client.get(
                self.endpoint,
                params={"q": query, "count": limit},
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": self.api_key,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPError:
            diagnostics.record_error(SEARCH_REQUEST_FAILED, domain)

            return []
        except ValueError:
            diagnostics.record_error(SEARCH_RESPONSE_UNUSABLE, domain)

            return []

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

        return results[:limit]

    def search(
        self,
        query: str,
        limit: int,
        allowed_domains: Sequence[str],
        diagnostics: RetrievalDiagnostics,
        per_domain: int = PER_DOMAIN_LIMIT,
    ) -> list[SearchResult]:
        """Search the allowed domains in turn, stopping once limit is filled.

        Brave indexes the whole web, so an unconstrained query spends the
        run's quota on pages the allowlist will throw away - which is exactly
        how a working API key produced no candidates at all. site: asks the
        index the question the allowlist is going to ask anyway.

        Domains are tried in the order given, which makes the order a priority
        order. No single domain may supply more than per_domain results,
        though: the first ordered run took all twenty-one candidates from one
        site, and a second opinion is worth one extra request.
        """
        collected: list[SearchResult] = []
        seen: set[str] = set()

        for domain in allowed_domains:
            if len(collected) >= limit:
                break

            domain = domain.strip().lower().strip(".")

            if not domain:
                continue

            diagnostics.record_query(domain)
            wanted = min(per_domain, limit - len(collected))

            for result in self.request(
                f"site:{domain} {query}", wanted, diagnostics, domain
            ):
                if result.url in seen:
                    continue

                seen.add(result.url)
                collected.append(result)

        return collected[:limit]
