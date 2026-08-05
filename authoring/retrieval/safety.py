"""What a retrieval run is allowed to connect to and read.

A search provider decides which URLs this code fetches, so those URLs are
untrusted input. Every check here runs before the first request and again on
each redirect hop, because a redirect is a second chance to point the fetcher
at a loopback or private-network address.
"""

import ipaddress
import re
from collections.abc import Sequence
from urllib.parse import urlsplit, urlunsplit

ALLOWED_SCHEMES = ("http", "https")
DEFAULT_PORTS = {"http": 80, "https": 443}

MAX_REDIRECTS = 3
MAX_PAGE_BYTES = 2_000_000

# Documents this milestone has no extractor for. The fetcher recognises most
# of them from their content type, but only after a request has been spent on
# them - and by then the result has already taken a slot that nothing else can
# refill. The suffix is the cheap half of the same check, made before the
# fetch rather than after it.
UNSUPPORTED_DOCUMENT_SUFFIXES = (".pdf", ".ppt", ".pptx", ".doc", ".docx", ".zip")

# A filename with one of those suffixes, named in a page's own title. The dot
# and the name in front of it are what separate a wrapper from a page that
# merely links to documents: MIT's lecture-notes pages list "( PDF )" beside
# every entry and are perfectly readable prose, while a page titled
# "MIT16_30F10_lec09.pdf" is a download behind an HTML frame.
DOCUMENT_FILENAME = re.compile(
    r"[\w.-]+\.(?:"
    + "|".join(suffix.lstrip(".") for suffix in UNSUPPORTED_DOCUMENT_SUFFIXES)
    + r")(?!\w)",
    re.IGNORECASE,
)


class UnsafeSource(ValueError):
    """A URL or response that a retrieval run must not use.

    The category is a fixed name the diagnostics can count, set by whoever
    raises it. It exists so a run can say "eight 404s" instead of "eight
    failures" without ever recording a word the server chose.
    """

    def __init__(self, message: str, category: str = ""):
        super().__init__(message)
        self.category = category


class UnreadableSource(UnsafeSource):
    """A source that could not be read: an error status, a timeout, a binary body.

    A subclass so that one except clause in the retrieval loop covers both
    "must not read this" and "could not read this", while the messages stay
    honest about which happened.
    """


class UnsupportedContentType(UnreadableSource):
    """A response that is not text this pipeline can read."""


class OversizedResponse(UnsafeSource):
    """A response larger than a reference passage could ever need to be."""


def host_of(url: str) -> str:
    """The host in one comparable form.

    The trailing dot of a fully qualified name and a unicode spelling of an
    ASCII domain both name the same host, so both are folded away here rather
    than left for the allowlist to be fooled by.
    """
    host = urlsplit(url).hostname

    if not host:
        raise UnsafeSource(f"{url!r} has no host.")

    host = host.lower().rstrip(".")

    if not host:
        raise UnsafeSource(f"{url!r} has no host.")

    if not host.isascii():
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError as error:
            raise UnsafeSource(f"{host} is not a usable hostname.") from error

    return host


def is_unsupported_document(url: str) -> bool:
    """A URL naming a document type this pipeline cannot read as text.

    The name is a hint, not a guarantee - a .pdf path can serve HTML and an
    extensionless path can serve a PDF - so the fetcher still checks what
    actually came back. This only avoids the fetches that are obviously
    pointless.
    """
    return urlsplit(url).path.lower().endswith(UNSUPPORTED_DOCUMENT_SUFFIXES)


def titled_as_document(title: str) -> bool:
    """A page that names a document as what it is, whatever it served.

    The live pilot collected "MIT16_30F10_lec09.pdf | Feedback Control
    Systems": an ordinary HTML page around a download link, so the URL suffix
    said nothing, the content type said text/html, and the passage a reviewer
    got was the course's navigation menu. What the page calls itself is the
    only place the document shows through, and it arrives with the search
    result - so this is checked before the fetch rather than after it.

    The title is the page's own <title>, which is metadata a server chose,
    and it is only ever read for this yes-or-no answer.
    """
    return bool(DOCUMENT_FILENAME.search(title))


def check_url(url: str) -> None:
    """Raise unless the URL is a plain public web address."""
    parts = urlsplit(url)

    if parts.scheme.lower() not in ALLOWED_SCHEMES:
        raise UnsafeSource(
            f"{parts.scheme or 'missing'} is not a supported URL scheme."
        )

    if parts.username or parts.password:
        raise UnsafeSource(f"{url} carries credentials.")

    host = host_of(url)

    if host == "localhost" or host.endswith(".localhost"):
        raise UnsafeSource(f"{host} is a loopback name.")

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return

    if not address.is_global:
        raise UnsafeSource(f"{host} is not a public address.")


def canonical_url(url: str) -> str:
    """The form two URLs share when they name the same page.

    Deduplication keys on this, so a trailing slash, a fragment, a default
    port or a www prefix cannot smuggle the same passage in twice.
    """
    parts = urlsplit(url)
    scheme = parts.scheme.lower()
    host = host_of(url)

    if host.startswith("www."):
        host = host[4:]

    netloc = host

    if parts.port and parts.port != DEFAULT_PORTS.get(scheme):
        netloc = f"{host}:{parts.port}"

    path = parts.path.rstrip("/")

    return urlunsplit((scheme, netloc, path, parts.query, ""))


def domain_is_allowed(url: str, allowed_domains: Sequence[str]) -> bool:
    """Match whole labels, so no lookalike host can wear an allowed suffix.

    Comparing label lists rather than string endings is the whole point:
    evilberkeley.edu and berkeley.edu.attacker.example both end in text that
    looks allowed, and neither is.
    """
    labels = host_of(url).split(".")

    for domain in allowed_domains:
        wanted = domain.strip().lower().strip(".").split(".")

        if not any(wanted):
            continue

        if labels[-len(wanted):] == wanted:
            return True

    return False
