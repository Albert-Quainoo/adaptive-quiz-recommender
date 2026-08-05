"""Turning a taxonomy skill into pending reference candidates.

The provider and the fetcher are injected protocols rather than one named
search service: the taxonomy should outlive whichever service is in use, and
the tests need a run that touches no network at all.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from authoring.retrieval.diagnostics import (
    FETCH_FAILED,
    INVALID_RESULT_URL,
    UNSUPPORTED_MEDIA,
    RetrievalDiagnostics,
)
from authoring.retrieval.models import (
    MIN_PASSAGE_CHARS,
    ReferenceCandidate,
    SearchResult,
    new_candidate,
    utc_now,
)
from authoring.retrieval.passage import (
    is_code_dense,
    looks_like_source,
    select_passage,
)
from authoring.retrieval.safety import (
    MAX_PAGE_BYTES,
    MAX_REDIRECTS,
    OversizedResponse,
    UnreadableSource,
    UnsafeSource,
    UnsupportedContentType,
    canonical_url,
    check_url,
    domain_is_allowed,
    host_of,
)
from taxonomy.schemas import SkillDefinition

SEARCH_LIMIT = 5


class RetrievalError(ValueError):
    pass


@dataclass(frozen=True)
class FetchedPage:
    """A page as read: where the fetcher landed, and how it got there."""

    url: str
    text: str
    redirects: tuple[str, ...] = ()


class SearchProvider(Protocol):
    """A provider searches within the allowed domains, not across the web.

    allowed_domains is part of the contract rather than a filter applied
    afterwards: a provider that searches the whole web and lets the caller
    discard the ineligible results spends its quota discovering pages that
    could never be used. The caller re-checks every URL regardless - see
    retrieve_candidates - because a provider is not a security boundary.
    """

    def search(
        self,
        query: str,
        limit: int,
        allowed_domains: Sequence[str],
        diagnostics: RetrievalDiagnostics,
    ) -> Sequence[SearchResult]: ...


class PageFetcher(Protocol):
    def fetch(self, url: str) -> FetchedPage: ...


def build_search_queries(skill: SkillDefinition) -> list[str]:
    """Query the skill from each angle the taxonomy already describes it by.

    The name alone is ambiguous across courses, so every query carries it plus
    one of the fields that places it: its topic, its subtopic, or the learning
    objective the reference has to be able to support.
    """
    queries = [
        f"{skill.name} {skill.topic}",
        f"{skill.name} {skill.subtopic}",
        f"{skill.name} {skill.learning_objective}",
    ]

    unique: list[str] = []

    for query in queries:
        normalised = " ".join(query.split())

        if normalised and normalised not in unique:
            unique.append(normalised)

    return unique


def read_page(
    fetcher: PageFetcher, url: str, allowed_domains: Sequence[str]
) -> FetchedPage:
    page = fetcher.fetch(url)

    if len(page.redirects) > MAX_REDIRECTS:
        raise UnsafeSource(f"{url} redirected {len(page.redirects)} times.")

    for hop in (*page.redirects, page.url):
        check_url(hop)

        if not domain_is_allowed(hop, allowed_domains):
            raise UnsafeSource(f"{hop} is outside the allowed domains.")

    if len(page.text.encode("utf-8")) > MAX_PAGE_BYTES:
        raise OversizedResponse(
            f"{page.url} returned more than {MAX_PAGE_BYTES} bytes."
        )

    return page


def retrieve_candidates(
    skill: SkillDefinition,
    provider: SearchProvider,
    fetcher: PageFetcher,
    allowed_domains: Sequence[str],
    limit: int = SEARCH_LIMIT,
    clock: Callable[[], datetime] = utc_now,
    diagnostics: RetrievalDiagnostics | None = None,
    min_passage_chars: int = MIN_PASSAGE_CHARS,
) -> list[ReferenceCandidate]:
    """Search, read and collect candidates for one skill. All come back pending.

    A result this run cannot vouch for is skipped rather than raised on: one
    unusable hit should not lose the rest of the run's reading. Every skip is
    counted, though - a run that quietly discards everything and reports zero
    is indistinguishable from a run that searched nothing.

    The provider searches within allowed_domains, and every URL it returns is
    checked against that same list here. The provider narrows what is
    discovered; only these checks decide what is read.
    """
    if not [domain for domain in allowed_domains if domain.strip()]:
        raise RetrievalError(
            f"{skill.skill_id} needs an explicit allowed-domain list to retrieve from."
        )

    diagnostics = diagnostics if diagnostics is not None else RetrievalDiagnostics()

    candidates: list[ReferenceCandidate] = []
    seen_urls: set[str] = set()
    seen_hashes: set[str] = set()

    for query in build_search_queries(skill):
        for result in provider.search(query, limit, allowed_domains, diagnostics):
            diagnostics.search_results_received += 1

            # Known only once the URL has passed the allowlist, which is what
            # makes it safe to record: it is a domain this run chose.
            domain = ""

            try:
                try:
                    check_url(result.url)
                except UnsafeSource:
                    diagnostics.record_error(INVALID_RESULT_URL)
                    raise

                if not domain_is_allowed(result.url, allowed_domains):
                    diagnostics.rejected_by_allowlist += 1
                    continue

                domain = host_of(result.url)

                # Cheaper to recognise a source file by its name than to
                # fetch it and read the imports.
                if looks_like_source(result.url):
                    diagnostics.rejected_as_non_prose += 1
                    continue

                key = canonical_url(result.url)

                if key in seen_urls:
                    diagnostics.duplicate_url += 1
                    continue

                seen_urls.add(key)

                page = read_page(fetcher, result.url, allowed_domains)
            except OversizedResponse:
                diagnostics.oversized_response += 1
                continue
            except UnsupportedContentType as error:
                diagnostics.unsupported_content_type += 1
                diagnostics.record_error(error.category or UNSUPPORTED_MEDIA, domain)
                continue
            except UnreadableSource as error:
                diagnostics.fetch_failures += 1
                diagnostics.record_error(error.category or FETCH_FAILED, domain)
                continue
            except UnsafeSource:
                diagnostics.rejected_as_unsafe += 1
                continue

            if is_code_dense(page.text):
                diagnostics.rejected_as_non_prose += 1
                continue

            # Quoted around the query that found the page, not from its top.
            passage = select_passage(page.text, query)

            if len(passage) < min_passage_chars:
                diagnostics.empty_or_short_passage += 1
                continue

            candidate = new_candidate(
                skill_id=skill.skill_id,
                title=result.title,
                source_url=page.url,
                source_domain=host_of(page.url),
                passage=passage,
                retrieved_at=clock(),
            )

            if candidate.content_hash in seen_hashes:
                diagnostics.duplicate_passage += 1
                continue

            seen_hashes.add(candidate.content_hash)
            candidates.append(candidate)
            diagnostics.candidates_created += 1

    return candidates
