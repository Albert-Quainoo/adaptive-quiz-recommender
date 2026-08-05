"""Where a retrieval run's results went.

A run that ends with nothing has to be able to say why: whether the searches
returned nothing, whether the allowlist dropped everything, or whether every
page failed to read. Counting each outcome is the difference between "zero
candidates" and a diagnosis.

Only counts and fixed category names are recorded here. Nothing that came off
the network - no query, no URL, no passage - and no credential ever lands in
a diagnostic.
"""

from collections import Counter
from dataclasses import dataclass, field

# Fixed category names. Provider and fetch failures are counted under these
# rather than by their messages, so no server-supplied text can reach a log.
SEARCH_REQUEST_FAILED = "search_request_failed"
SEARCH_RESPONSE_UNUSABLE = "search_response_unusable"
FETCH_FAILED = "fetch_failed"
UNSUPPORTED_MEDIA = "unsupported_media"
INVALID_RESULT_URL = "invalid_result_url"

# Failures carry a narrower name where one is known: an HTTP status, a
# timeout, a media family. A status code is not something a server wrote in
# prose, so counting it leaks nothing and answers "why did eight fetches
# fail" - which the bare category could not.
HTTP_STATUS_CATEGORIES = {
    401: "fetch_failed_401",
    403: "fetch_failed_403",
    404: "fetch_failed_404",
    410: "fetch_failed_410",
    429: "fetch_failed_429",
}

MEDIA_FAMILIES = {
    "application/pdf": "unsupported_pdf",
    "application/zip": "unsupported_archive",
    "application/octet-stream": "unsupported_binary",
}


def status_category(status: int) -> str:
    if status in HTTP_STATUS_CATEGORIES:
        return HTTP_STATUS_CATEGORIES[status]

    return f"fetch_failed_{status // 100}xx"


def media_category(content_type: str) -> str:
    if content_type in MEDIA_FAMILIES:
        return MEDIA_FAMILIES[content_type]

    family = content_type.split("/")[0] if "/" in content_type else ""

    return f"unsupported_{family}" if family else "unsupported_unknown"


@dataclass
class RetrievalDiagnostics:
    search_requests_made: int = 0
    search_results_received: int = 0
    rejected_by_allowlist: int = 0
    rejected_as_unsafe: int = 0
    fetch_failures: int = 0
    unsupported_content_type: int = 0
    oversized_response: int = 0
    empty_or_short_passage: int = 0
    rejected_as_non_prose: int = 0
    duplicate_url: int = 0
    duplicate_passage: int = 0
    candidates_created: int = 0

    errors: Counter = field(default_factory=Counter)
    domains_queried: list[str] = field(default_factory=list)
    failures_by_domain: dict[str, Counter] = field(default_factory=dict)

    def record_error(self, category: str, domain: str = "") -> None:
        """Count a failure, and where it happened when that is ours to know.

        The domain is only ever one this run chose to query - never a host a
        server redirected us to - so attributing a failure tells us which of
        our own sources has gone stale without recording anyone else's.
        """
        self.errors[category] += 1

        if domain:
            self.failures_by_domain.setdefault(domain, Counter())[category] += 1

    def record_query(self, domain: str) -> None:
        self.search_requests_made += 1

        if domain not in self.domains_queried:
            self.domains_queried.append(domain)


COUNTS = (
    ("search requests made", "search_requests_made"),
    ("search results received", "search_results_received"),
    ("rejected by allowlist", "rejected_by_allowlist"),
    ("rejected as unsafe", "rejected_as_unsafe"),
    ("fetch failures", "fetch_failures"),
    ("unsupported content type", "unsupported_content_type"),
    ("oversized responses", "oversized_response"),
    ("empty or short passages", "empty_or_short_passage"),
    ("rejected as non-prose", "rejected_as_non_prose"),
    ("duplicate urls", "duplicate_url"),
    ("duplicate passages", "duplicate_passage"),
    ("candidates created", "candidates_created"),
)


def explain(diagnostics: RetrievalDiagnostics) -> str:
    """The one line worth reading when a run produced nothing."""
    if diagnostics.candidates_created:
        return ""

    if not diagnostics.search_requests_made:
        return "No searches were issued: the run had no query or no allowed domain."

    if diagnostics.errors[SEARCH_REQUEST_FAILED] >= diagnostics.search_requests_made:
        return "Every search request failed: check the API key, quota and network."

    if not diagnostics.search_results_received:
        return "The searches returned no results on the allowed domains."

    if diagnostics.rejected_by_allowlist >= diagnostics.search_results_received:
        return "Every result was off the allowlist, so nothing was fetched."

    if diagnostics.fetch_failures or diagnostics.unsupported_content_type:
        return "Results were found but their pages could not be read."

    if diagnostics.rejected_as_non_prose:
        return "Results were source files or code pages, not reference prose."

    if diagnostics.empty_or_short_passage:
        return "Pages were read but held too little text to quote."

    return "Results were found but none survived the retrieval checks."


def summary(diagnostics: RetrievalDiagnostics) -> str:
    lines = ["Retrieval diagnostics"]
    lines += [
        f"  {label:<26}{getattr(diagnostics, attribute):>5}"
        for label, attribute in COUNTS
    ]

    if diagnostics.domains_queried:
        lines.append(f"  domains queried: {', '.join(diagnostics.domains_queried)}")

    if diagnostics.errors:
        counted = ", ".join(
            f"{category} {count}"
            for category, count in sorted(diagnostics.errors.items())
        )
        lines.append(f"  errors: {counted}")

    if diagnostics.failures_by_domain:
        lines.append("  failures by domain:")
        width = max(len(domain) for domain in diagnostics.failures_by_domain)

        for domain in sorted(diagnostics.failures_by_domain):
            failures = diagnostics.failures_by_domain[domain]
            counted = "  ".join(
                f"{category} x{count}" for category, count in sorted(failures.items())
            )
            lines.append(f"    {domain:<{width}}  {counted}")

    reason = explain(diagnostics)

    if reason:
        lines.append(f"  {reason}")

    return "\n".join(lines)
