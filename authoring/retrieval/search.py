"""Turning a taxonomy skill into pending reference candidates.

The provider and the fetcher are injected protocols rather than one named
search service: the taxonomy should outlive whichever service is in use, and
the tests need a run that touches no network at all.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from authoring.retrieval.diagnostics import (
    FETCH_FAILED,
    INVALID_RESULT_URL,
    UNSUPPORTED_DOCUMENT,
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
from authoring.retrieval.relevance import (
    AI_CONTEXT_ANCHOR,
    MIN_RELEVANCE_SCORE,
    SourceScope,
    score_relevance,
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
    is_unsupported_document,
    titled_as_document,
)
from taxonomy.schemas import SkillDefinition

# The number of candidates a skill retains after every eligible page found
# within the budget has been ranked. It is not a cap on how many search
# results may be read: a limit on raw results is what let one PDF cost a slot
# outright, and stopping at the fifth eligible result would preserve search
# order instead of selecting the best five.
SEARCH_LIMIT = 5

# What one skill may spend reaching that target. Backfilling means every
# rejected result costs another search or another fetch, so a skill whose
# domains all serve slide decks would page through the index indefinitely
# without a ceiling on both.
#
# Ten requests, spent by the schedule below rather than on one query angle.
# Each page is over-fetched, so a skill can still see a hundred results. The
# whole taxonomy at this rate is 37 x 10 = 370 requests, well inside the
# monthly credit - and most skills stop long before ten.
MAX_SEARCH_REQUESTS_PER_SKILL = 10
MAX_PAGE_FETCHES_PER_SKILL = 25

# How many of a skill's domains are asked every query angle. Three, because
# that is what PILOT_SOURCE_SCOPES names for each skill: the places chosen for
# this skill in particular, as against the rest of the allowlist.
PREFERRED_DOMAIN_COUNT = 3

AI_FND_01_PASSAGE_QUERY = (
    "Define artificial intelligence and recognise tasks that require intelligent "
    "behaviour. field devoted building intelligent agents systems percepts "
    "environment behavior actions examples faces chess speech problem solving learning"
)


class RetrievalError(ValueError):
    pass


@dataclass
class RetrievalBudget:
    """One skill's allowance of search requests and page fetches.

    Shared across that skill's queries rather than reset per query, because
    the queries are a fallback ladder: the second one only runs because the
    first came up short, and the two together are what the budget covers.
    """

    max_requests: int = MAX_SEARCH_REQUESTS_PER_SKILL
    max_fetches: int = MAX_PAGE_FETCHES_PER_SKILL
    requests_made: int = 0
    fetches_made: int = 0

    def may_request(self) -> bool:
        return self.requests_made < self.max_requests

    def spend_request(self) -> None:
        self.requests_made += 1

    def may_fetch(self) -> bool:
        return self.fetches_made < self.max_fetches

    def spend_fetch(self) -> None:
        self.fetches_made += 1


@dataclass(frozen=True)
class FetchedPage:
    """A page as read: where the fetcher landed, and how it got there."""

    url: str
    text: str
    redirects: tuple[str, ...] = ()


@dataclass(frozen=True)
class KnownCandidates:
    """What earlier runs already produced for one skill.

    Retrieval is run more than once against the same store, and without this
    the second run rediscovers the first run's pages, spends its whole
    allowance re-reading them and adds nothing - the store drops them on the
    way in, so the quota is gone and the shelf is unchanged.
    """

    urls: frozenset[str] = frozenset()
    hashes: frozenset[str] = frozenset()
    toward_target: int = 0


def known_for(
    skill_id: str, stored: Iterable[ReferenceCandidate]
) -> KnownCandidates:
    """What the store holds for one skill: what to skip, and what still counts.

    Every stored candidate is skipped whatever its status - a page a reviewer
    has already turned down is the last page worth fetching again - but only
    the ones still standing count towards the target. A skill whose five
    candidates were all rejected has nothing to show for them and should go
    looking again; a skill with five pending should be read before it is
    asked to find more.

    Exclusion is per skill because sharing is legitimate: one page can ground
    two skills, and the store keeps them apart by candidate id.
    """
    mine = [
        candidate for candidate in stored if candidate.skill_id == skill_id
    ]

    return KnownCandidates(
        urls=frozenset(canonical_url(candidate.source_url) for candidate in mine),
        hashes=frozenset(candidate.content_hash for candidate in mine),
        toward_target=sum(
            candidate.review_status != "rejected" for candidate in mine
        ),
    )


@dataclass(frozen=True)
class SearchStep:
    """One request the run intends to make: one query angle, in one place.

    The place is a domain and, where this skill has a scope for it, the path
    on that domain worth asking. A provider narrows its request to both - see
    brave.query_for - so a step on a scoped domain spends its quota inside
    the course rather than across the whole site.
    """

    query: str
    domain: str
    path: str = ""


def build_search_schedule(
    queries: Sequence[str],
    domains: Sequence[str],
    steps: int = MAX_SEARCH_REQUESTS_PER_SKILL,
    preferred: int = PREFERRED_DOMAIN_COUNT,
    scopes: Sequence[SourceScope] = (),
) -> list[SearchStep]:
    """Decide in advance which query angle to ask of which domain.

    A skill is asked three ways - by topic, by subtopic and by learning
    objective - and the three do not find the same pages. Spending the whole
    allowance on the first angle across every domain answers one question
    thoroughly and the other two not at all, which is the wrong trade when
    the angle that would have worked is the third.

    So the preferred domains are asked all three angles first: nine requests
    covering every angle on the three sites chosen for this skill, one site
    at a time within each angle so the first three requests are three
    different sources rather than three ways of asking one. The tenth reaches
    past the preferences to the first fallback domain, which is the cheapest
    hedge against all three preferences being wrong about this skill.

    A domain this skill has a scope for is asked at that scope's path. The
    schedule asks by domain, so it is the first scope named for a domain that
    is used - which is also the one domains_for kept when it collapsed the
    scopes into a domain order, so the two agree.

    The rest of the allowlist is still the boundary every URL is checked
    against - see retrieve_candidates. This only decides what gets asked.
    """
    paths: dict[str, str] = {}

    for scope in scopes:
        paths.setdefault(scope.domain.strip().lower().strip("."), scope.path)

    ordered = [
        cleaned
        for domain in domains
        if (cleaned := domain.strip().lower().strip("."))
    ]
    leading, fallback = ordered[:preferred], ordered[preferred:]

    planned = [
        SearchStep(query, domain, paths.get(domain, ""))
        for query in queries
        for domain in leading
    ]

    if queries:
        planned += [
            SearchStep(queries[0], domain, paths.get(domain, ""))
            for domain in fallback
        ]

    return planned[:steps]


class SearchProvider(Protocol):
    """A provider works the schedule it is given, not the whole web.

    The schedule is part of the contract rather than a filter applied
    afterwards: a provider that searches the whole web and lets the caller
    discard the ineligible results spends its quota discovering pages that
    could never be used. The caller re-checks every URL regardless - see
    retrieve_candidates - because a provider is not a security boundary.

    Each result is yielded with the step that found it, one at a time, and no
    count is asked for. The caller is the only one that knows how many
    results turned into usable candidates and how many were PDFs, so it is
    the caller that decides when to stop - it simply stops consuming, and no
    further request is made. The budget is what stops the provider when the
    caller never stops asking.
    """

    def search(
        self,
        schedule: Sequence[SearchStep],
        diagnostics: RetrievalDiagnostics,
        budget: RetrievalBudget,
    ) -> Iterator[tuple[SearchStep, SearchResult]]: ...


class PageFetcher(Protocol):
    def fetch(self, url: str) -> FetchedPage: ...


def build_search_queries(skill: SkillDefinition) -> list[str]:
    """Query the skill from each angle the taxonomy already describes it by.

    The name alone is ambiguous across courses, so every query carries it plus
    one of the fields that places it: its topic, its subtopic, or the learning
    objective the reference has to be able to support.

    Every angle also carries the AI context anchor, because those fields are
    ambiguous too. "Problem formulation" and "Search and Problem Solving" are
    what this taxonomy calls its own topics; asked of the whole of MIT
    OpenCourseWare they returned a problem-solving approach to electromagnetic
    field theory, which answers the question as put. No angle searches alone.
    """
    queries = [
        f"{skill.name} {skill.topic}",
        f"{skill.name} {skill.subtopic}",
        f"{skill.name} {skill.learning_objective}",
    ]

    unique: list[str] = []

    for query in queries:
        normalised = " ".join(f"{query} {AI_CONTEXT_ANCHOR}".split())

        if normalised and normalised not in unique:
            unique.append(normalised)

    return unique


def passage_query_for(skill: SkillDefinition) -> str:
    if skill.skill_id == "AI-FND-01":
        return AI_FND_01_PASSAGE_QUERY
    return skill.learning_objective


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
    budget: RetrievalBudget | None = None,
    known: KnownCandidates | None = None,
    scopes: Sequence[SourceScope] = (),
    min_relevance: int = MIN_RELEVANCE_SCORE,
) -> list[ReferenceCandidate]:
    """Search, read and collect candidates for one skill. All come back pending.

    A result this run cannot vouch for is skipped rather than raised on: one
    unusable hit should not lose the rest of the run's reading. Every skip is
    counted, though - a run that quietly discards everything and reports zero
    is indistinguishable from a run that searched nothing.

    A skip also does not cost a slot. limit is the number of candidates kept,
    not the number of eligible candidates considered. The search runs until
    its schedule or budget is exhausted, ranks every eligible candidate it
    found, and retains the best remaining slots. Reading limit as "how many
    results to look at" is what left the first runs holding two candidates and
    three PDFs.

    What a previous run already found counts towards that target and is
    skipped on the way past, so running again looks for the shortfall rather
    than for the whole target again. A skill already holding enough plans no
    requests at all.

    The provider works the schedule built here, and every URL it returns is
    checked against the whole allowed_domains list. The schedule narrows what
    is discovered; only these checks decide what is read.

    scopes are the reviewed paths on those domains that hold this skill's
    material. When supplied they are a relevance prerequisite after fetch;
    allowed_domains remains the network boundary that alone says what may be
    fetched.
    """
    if not [domain for domain in allowed_domains if domain.strip()]:
        raise RetrievalError(
            f"{skill.skill_id} needs an explicit allowed-domain list to retrieve from."
        )

    diagnostics = diagnostics if diagnostics is not None else RetrievalDiagnostics()
    budget = budget if budget is not None else RetrievalBudget()
    known = known if known is not None else KnownCandidates()

    eligible: dict[str, ReferenceCandidate] = {}
    seen_urls: set[str] = set(known.urls)

    held = known.toward_target
    diagnostics.candidates_already_held += held

    # A scope on a domain this run may not read is a request nobody could
    # use the answer to, so it is not asked. The allowlist decides this the
    # same way it decides every result URL.
    askable = [
        scope
        for scope in scopes
        if domain_is_allowed(f"https://{scope.domain}/", allowed_domains)
    ]

    # An empty schedule is how a skill that already has enough asks for
    # nothing: no step to walk means no request to pay for.
    schedule = (
        []
        if held >= limit
        else build_search_schedule(
            build_search_queries(skill),
            allowed_domains,
            budget.max_requests,
            scopes=askable,
        )
    )

    for step, result in provider.search(schedule, diagnostics, budget):
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

            # Cheaper to recognise a source file by its name than to fetch
            # it and read the imports.
            if looks_like_source(result.url):
                diagnostics.rejected_as_non_prose += 1
                continue

            # Same reasoning, and the reason this loop exists: a PDF found
            # after the fetch has already cost a slot, while one found here
            # costs nothing but the next result.
            #
            # The title is checked alongside the URL because a document can
            # be served from an HTML wrapper, where neither the suffix nor
            # the content type gives it away and the passage a reviewer gets
            # is the site's navigation menu.
            if is_unsupported_document(result.url) or titled_as_document(result.title):
                diagnostics.unsupported_document_skipped += 1
                diagnostics.record_error(UNSUPPORTED_DOCUMENT, domain)
                continue

            key = canonical_url(result.url)

            if key in seen_urls:
                diagnostics.duplicate_url += 1
                continue

            seen_urls.add(key)

            if not budget.may_fetch():
                break

            budget.spend_fetch()
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

        # Quote the part that can support the learning objective. The search
        # angle discovers the page; it does not decide what the reviewer sees.
        passage = select_passage(page.text, passage_query_for(skill))

        # Nothing came back at all: the page holds no prose to quote, only
        # menus and headings. Counted with the source files rather than with
        # the short passages, because the two ask different questions - "was
        # there anything to read here" and "was there enough of it".
        if not passage:
            diagnostics.rejected_as_non_prose += 1
            continue

        if len(passage) < min_passage_chars:
            diagnostics.empty_or_short_passage += 1
            continue

        # Last, and only here: relevance is the one check that needs the
        # passage a reviewer would actually read. A result that fails it has
        # cost a fetch and costs no slot, which is the trade the whole
        # backfilling loop is built on.
        relevance = score_relevance(
            skill,
            page.url,
            result.title,
            result.snippet,
            passage,
            scopes,
        )

        if not relevance.is_relevant(min_relevance):
            diagnostics.rejected_as_irrelevant += 1
            continue

        candidate = new_candidate(
            skill_id=skill.skill_id,
            title=result.title,
            source_url=page.url,
            source_domain=host_of(page.url),
            passage=passage,
            retrieved_at=clock(),
            relevance_score=relevance.score,
            matched_terms=relevance.matched_terms,
        )

        if candidate.content_hash in known.hashes:
            diagnostics.duplicate_passage += 1
            continue

        existing = eligible.get(candidate.content_hash)

        if existing is not None:
            diagnostics.duplicate_passage += 1

            if (-candidate.relevance_score, candidate.source_url) < (
                -existing.relevance_score,
                existing.source_url,
            ):
                eligible[candidate.content_hash] = candidate

            continue

        eligible[candidate.content_hash] = candidate

    slots = max(limit - held, 0)
    candidates = sorted(
        eligible.values(),
        key=lambda candidate: (
            -candidate.relevance_score,
            candidate.source_url,
            candidate.candidate_id,
        ),
    )[:slots]
    diagnostics.candidates_created += len(candidates)

    record_outcome(diagnostics, budget, reached=held + len(candidates) >= limit)

    return candidates


def record_outcome(
    diagnostics: RetrievalDiagnostics, budget: RetrievalBudget, reached: bool
) -> None:
    """Say how this skill's search ended, in exactly one count.

    "Four candidates" means something different depending on whether the
    index had nothing else to offer or the run stopped paying for it, and the
    counts alone cannot tell those apart.
    """
    if reached:
        diagnostics.targets_reached += 1
    elif not budget.may_fetch():
        diagnostics.fetch_budgets_exhausted += 1
    elif not budget.may_request():
        diagnostics.request_budgets_exhausted += 1
    else:
        diagnostics.searches_exhausted += 1
