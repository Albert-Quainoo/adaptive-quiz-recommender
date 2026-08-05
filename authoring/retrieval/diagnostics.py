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
# A request that was still being refused after every retry it was allowed.
# Told apart from a plain failure because the answer is different: wait, or
# raise the plan's rate limit - not "check the key".
SEARCH_RATE_LIMITED = "search_rate_limited"
FETCH_FAILED = "fetch_failed"
UNSUPPORTED_MEDIA = "unsupported_media"
INVALID_RESULT_URL = "invalid_result_url"

# A document recognised from its URL and never fetched. Deliberately distinct
# from the unsupported_* media categories, which are what a fetch discovered:
# the two answer different questions, one about the search results and one
# about the pages behind them.
UNSUPPORTED_DOCUMENT = "unsupported_document"

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
    # search_requests_made counts every result page asked of the provider;
    # paginated_requests counts the ones asked at a non-zero offset, which is
    # how far past the first page the run had to go to fill its target.
    search_requests_made: int = 0
    paginated_requests: int = 0
    # Waits that bought a working request. A request given up on after its
    # retries is counted under SEARCH_RATE_LIMITED instead.
    rate_limit_retries: int = 0
    search_results_received: int = 0
    rejected_by_allowlist: int = 0
    rejected_as_unsafe: int = 0
    # Recognised from the URL and never fetched, as against
    # unsupported_content_type, which is what a fetched response turned out
    # to be.
    unsupported_document_skipped: int = 0
    fetch_failures: int = 0
    unsupported_content_type: int = 0
    oversized_response: int = 0
    empty_or_short_passage: int = 0
    rejected_as_non_prose: int = 0
    # Read, quotable, and about something else. Counted apart from every
    # other rejection because it is the only one that says the sources are
    # fine and the question reached them wrong.
    rejected_as_irrelevant: int = 0
    duplicate_url: int = 0
    duplicate_passage: int = 0
    candidates_created: int = 0
    # Standing candidates an earlier run left in the store, which counted
    # towards this run's targets instead of being looked for again.
    candidates_already_held: int = 0

    # How each skill's search ended. Exactly one of these is counted per
    # skill, so a run of three skills adds three between them - which is what
    # separates "five candidates because that is all there was" from "five
    # candidates because the budget ran out".
    targets_reached: int = 0
    searches_exhausted: int = 0
    request_budgets_exhausted: int = 0
    fetch_budgets_exhausted: int = 0

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

    def absorb(self, other: "RetrievalDiagnostics") -> None:
        """Add one skill's record into a running total.

        The counts are merged through COUNTS rather than named again here, so
        a count that is reported is a count that is totalled - the two cannot
        drift apart and leave a run summing fifteen of its eighteen numbers.
        """
        for _, attribute in COUNTS:
            total = getattr(self, attribute) + getattr(other, attribute)
            setattr(self, attribute, total)

        self.errors += other.errors

        for domain in other.domains_queried:
            if domain not in self.domains_queried:
                self.domains_queried.append(domain)

        for domain, failures in other.failures_by_domain.items():
            self.failures_by_domain.setdefault(domain, Counter()).update(failures)

    def record_query(self, domain: str, offset: int = 0) -> None:
        self.search_requests_made += 1

        if offset:
            self.paginated_requests += 1

        if domain not in self.domains_queried:
            self.domains_queried.append(domain)


COUNTS = (
    ("search requests made", "search_requests_made"),
    ("paginated requests", "paginated_requests"),
    ("rate limit retries", "rate_limit_retries"),
    ("search results received", "search_results_received"),
    ("rejected by allowlist", "rejected_by_allowlist"),
    ("rejected as unsafe", "rejected_as_unsafe"),
    ("unsupported documents", "unsupported_document_skipped"),
    ("fetch failures", "fetch_failures"),
    ("unsupported content type", "unsupported_content_type"),
    ("oversized responses", "oversized_response"),
    ("empty or short passages", "empty_or_short_passage"),
    ("rejected as non-prose", "rejected_as_non_prose"),
    ("rejected as irrelevant", "rejected_as_irrelevant"),
    ("duplicate urls", "duplicate_url"),
    ("duplicate passages", "duplicate_passage"),
    ("candidates created", "candidates_created"),
    ("already held", "candidates_already_held"),
    ("targets reached", "targets_reached"),
    ("searches exhausted", "searches_exhausted"),
    ("request budgets spent", "request_budgets_exhausted"),
    ("fetch budgets spent", "fetch_budgets_exhausted"),
)

# Why a skill stopped short of its usable-candidate target. Reported so that
# a thin run can be told from a curtailed one without reading the counts.
SHORTFALLS = (
    ("search exhausted", "searches_exhausted"),
    ("search request budget spent", "request_budgets_exhausted"),
    ("page fetch budget spent", "fetch_budgets_exhausted"),
)


def explain(diagnostics: RetrievalDiagnostics) -> str:
    """The one line worth reading when a run produced nothing."""
    if diagnostics.candidates_created:
        return ""

    if diagnostics.candidates_already_held and not diagnostics.search_requests_made:
        return (
            "Nothing was searched for: the store already holds every skill's "
            "target. Review what is pending, or raise the limit."
        )

    if not diagnostics.search_requests_made:
        return "No searches were issued: the run had no query or no allowed domain."

    if diagnostics.errors[SEARCH_RATE_LIMITED] >= diagnostics.search_requests_made:
        return (
            "Every search request was rate limited: the run is asking faster "
            "than the plan allows."
        )

    if diagnostics.errors[SEARCH_REQUEST_FAILED] >= diagnostics.search_requests_made:
        return "Every search request failed: check the API key, quota and network."

    if not diagnostics.search_results_received:
        return "The searches returned no results on the allowed domains."

    if diagnostics.rejected_by_allowlist >= diagnostics.search_results_received:
        return "Every result was off the allowlist, so nothing was fetched."

    if diagnostics.unsupported_document_skipped >= diagnostics.search_results_received:
        return "Every result was a PDF, slide deck or other unreadable document."

    if diagnostics.fetch_failures or diagnostics.unsupported_content_type:
        return "Results were found but their pages could not be read."

    if diagnostics.rejected_as_non_prose:
        return (
            "Results were source files, code pages or navigation, not "
            "reference prose."
        )

    if diagnostics.rejected_as_irrelevant:
        return (
            "Pages were read but none was about the skill that searched for "
            "them: check the source scopes, or the query anchor."
        )

    if diagnostics.empty_or_short_passage:
        return "Pages were read but held too little text to quote."

    return "Results were found but none survived the retrieval checks."


def summary(diagnostics: RetrievalDiagnostics, label: str = "") -> str:
    """The counts, and the one line worth reading under them.

    The label names whose record this is - a skill id, for a per-skill
    report. It is one of ours, never anything off the network.
    """
    lines = [f"Retrieval diagnostics{f': {label}' if label else ''}"]
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

    short = [
        f"{label} x{getattr(diagnostics, attribute)}"
        for label, attribute in SHORTFALLS
        if getattr(diagnostics, attribute)
    ]

    if short:
        lines.append(f"  stopped short of target: {', '.join(short)}")

    reason = explain(diagnostics)

    if reason:
        lines.append(f"  {reason}")

    return "\n".join(lines)
