"""The production page fetcher.

Redirects are followed by hand rather than by httpx, because a redirect is a
server telling this process where to connect next: every hop is re-checked
against the same rules as the original URL, before the request is made and
again after it resolves. An allowed first hop earns nothing.
"""

import re
from html import unescape

import httpx

from authoring.retrieval.diagnostics import media_category, status_category
from authoring.retrieval.safety import (
    MAX_PAGE_BYTES,
    MAX_REDIRECTS,
    OversizedResponse,
    UnreadableSource,
    UnsafeSource,
    UnsupportedContentType,
    check_url,
    domain_is_allowed,
)
from authoring.retrieval.search import FetchedPage

DEFAULT_TIMEOUT = 10.0

TEXTUAL_CONTENT_TYPES = (
    "text/html",
    "text/plain",
    "application/xhtml+xml",
)

STRIPPED_ELEMENTS = re.compile(
    r"<(script|style|noscript)\b[^>]*>.*?</\1>", re.DOTALL | re.IGNORECASE
)
# A tag opens with a letter, a slash or a bang - never with a space or a
# digit - so "x < 5 and y > 3" in prose survives this where <[^>]*> ate it.
TAG = re.compile(r"<\s*/?[a-zA-Z!][^>]*>")
EXPANDABLE_TEASER = re.compile(
    r"(^|(?<=[.!?]))[^.!?]*(?:…|\.\.\.)\s*Show more\s*",
    re.IGNORECASE,
)
EXPANSION_CONTROL = re.compile(r"\bShow (?:more|less)\b", re.IGNORECASE)
SPACE_BEFORE_PUNCTUATION = re.compile(r"\s+([,.;:!?])")


def visible_text(body: str) -> str:
    """The reading text of a page, without its markup or its scripts.

    Tags are stripped twice, either side of unescaping. A page that writes
    its markup as entities - &lt;div data-bs-toggle="collapse"&gt; - passes
    the first strip untouched and becomes markup again when unescaped, which
    is how `data-bs-toggle="collapse">` ended up quoted in a passage.
    """
    without_tags = TAG.sub(" ", STRIPPED_ELEMENTS.sub(" ", body))
    visible = TAG.sub(" ", unescape(without_tags))
    without_teasers = EXPANDABLE_TEASER.sub(" ", visible)

    without_controls = EXPANSION_CONTROL.sub(" ", without_teasers)
    return SPACE_BEFORE_PUNCTUATION.sub(r"\1", without_controls)


def content_type_of(response: httpx.Response) -> str:
    return response.headers.get("content-type", "").split(";")[0].strip().lower()


class HttpPageFetcher:
    """Fetches pages over HTTP within the limits the retrieval rules set."""

    def __init__(
        self,
        allowed_domains: tuple[str, ...],
        client: httpx.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_redirects: int = MAX_REDIRECTS,
        max_bytes: int = MAX_PAGE_BYTES,
    ):
        self.allowed_domains = tuple(allowed_domains)
        self.client = client or httpx.Client(follow_redirects=False)
        self.timeout = timeout
        self.max_redirects = max_redirects
        self.max_bytes = max_bytes

    def check_destination(self, url: str) -> None:
        check_url(url)

        if not domain_is_allowed(url, self.allowed_domains):
            raise UnsafeSource(f"{url} is outside the allowed domains.")

    def read_body(self, response: httpx.Response) -> str:
        declared = response.headers.get("content-length")

        if declared and declared.isdigit() and int(declared) > self.max_bytes:
            raise OversizedResponse(
                f"{response.url} declares more than {self.max_bytes} bytes."
            )

        body = bytearray()

        for chunk in response.iter_bytes():
            body.extend(chunk)

            if len(body) > self.max_bytes:
                raise OversizedResponse(
                    f"{response.url} returned more than {self.max_bytes} bytes."
                )

        return bytes(body).decode(response.charset_encoding or "utf-8", errors="replace")

    def request(self, url: str) -> tuple[str | None, str]:
        """One request: either where it points next, or the body it returned."""
        try:
            with self.client.stream(
                "GET", url, timeout=self.timeout, follow_redirects=False
            ) as response:
                if response.is_redirect:
                    location = response.headers.get("location", "").strip()

                    if not location:
                        raise UnsafeSource(f"{url} redirected without a location.")

                    return str(response.url.join(location)), ""

                if response.status_code >= 400:
                    raise UnreadableSource(
                        f"{url} returned status {response.status_code}.",
                        status_category(response.status_code),
                    )

                content_type = content_type_of(response)

                if content_type not in TEXTUAL_CONTENT_TYPES:
                    raise UnsupportedContentType(
                        f"{url} served {content_type or 'no content type'}, "
                        f"which is not readable text.",
                        media_category(content_type),
                    )

                return None, self.read_body(response)
        except httpx.TimeoutException as error:
            raise UnreadableSource(
                f"{url} timed out.", "fetch_failed_timeout"
            ) from error
        except httpx.TransportError as error:
            raise UnreadableSource(
                f"{url} could not be reached.", "fetch_failed_connection"
            ) from error
        except httpx.HTTPError as error:
            raise UnreadableSource(
                f"{url} could not be read: {error}", "fetch_failed_other"
            ) from error

    def fetch(self, url: str) -> FetchedPage:
        redirects: list[str] = []
        current = url

        while True:
            self.check_destination(current)

            target, body = self.request(current)

            if target is None:
                return FetchedPage(
                    url=current, text=visible_text(body), redirects=tuple(redirects)
                )

            if len(redirects) >= self.max_redirects:
                raise UnsafeSource(
                    f"{url} redirected more than {self.max_redirects} times."
                )

            redirects.append(current)
            current = target
