"""The first retrieval-assisted authoring run: AI-SRC-01, AI-SRC-02, AI-SRC-08.

Three generated, understand-level search skills that carry no reference
material yet, which is what makes them the honest test of the workflow.

This module plans the run and can execute it against an injected provider. It
writes nothing: references.csv stays as it is until the candidates it produces
have been read and approved, and which domains are trustworthy for a course is
a review decision, not one retrieval gets to make for itself.
"""

from collections.abc import Callable, Iterable, Sequence
from datetime import datetime
from pathlib import Path

from authoring.retrieval.diagnostics import RetrievalDiagnostics
from authoring.retrieval.models import ReferenceCandidate, utc_now
from authoring.retrieval.relevance import SourceScope
from authoring.retrieval.search import (
    SEARCH_LIMIT,
    PageFetcher,
    SearchProvider,
    build_search_queries,
    known_for,
    retrieve_candidates,
)
from taxonomy.loader import course_paths, load_skills
from taxonomy.schemas import SkillCatalogue, SkillDefinition

PILOT_SKILL_IDS = ("AI-SRC-01", "AI-SRC-02", "AI-SRC-08")

# Every domain here is a decision that its material may become course
# reference text, so each one is named individually - never a whole registry
# like .edu, and never a search or aggregation site that would let anyone else
# choose for us.
#
# The order is a priority order: a query stops as soon as it has filled its
# limit, so whatever sits at the top supplies most of what gets reviewed. The
# first live run put aima.cs.berkeley.edu first and collected a table of
# contents, a Lisp manifest and two copies of search.py - it is the book's
# code companion, not its prose. Sites that explain in sentences come first
# now, and the code companions sit below them.
PILOT_ALLOWED_DOMAINS = (
    # Explanatory prose about search, which is what the pilot skills need
    "redblobgames.com",  # reviewed by Albert: a personal site, but the
    # clearest pathfinding and heuristics writing available
    "inst.eecs.berkeley.edu",  # Berkeley CS188, the HTML textbook
    "ocw.mit.edu",  # MIT OpenCourseWare
    "cs50.harvard.edu",  # Harvard CS50 AI with Python
    # Scholarly reference
    "plato.stanford.edu",  # Stanford Encyclopedia of Philosophy
    # Open textbooks
    "d2l.ai",  # Dive into Deep Learning
    "deeplearningbook.org",  # Goodfellow, Bengio and Courville
    # Code companions and API documentation: accurate, but they describe
    # implementations rather than teach the ideas
    "aima.cs.berkeley.edu",  # Russell & Norvig, companion site
    "scikit-learn.org",
    "pytorch.org",
    # Still reviewed, still allowed, no longer preferred: the CS188 material
    # here has moved and the live pilot spent six fetches on 404s finding
    # that out. inst.eecs.berkeley.edu carries the textbook now.
    "ai.berkeley.edu",
)

# Where on those domains this skill's material actually is.
#
# A domain is too coarse a unit to say that with. ocw.mit.edu is one domain
# and two thousand courses, and searching it whole returned electromagnetic
# field theory for "problem formulation" and functional analysis for
# "heuristic search" - both real pages, both on an approved domain, neither of
# any use. cs50.harvard.edu is one domain and two courses, and the live run
# collected the introductory programming one.
#
# So the priority order is per skill and per path. The allowlist remains the
# only thing that decides what may be fetched, while these reviewed paths are
# a separate relevance prerequisite: being inside one permits a page to be
# scored, but does not prove that it covers the concept or objective.
CS188_TEXTBOOK = SourceScope("inst.eecs.berkeley.edu", "/~cs188/textbook/")
CS50_AI = SourceScope("cs50.harvard.edu", "/ai/")
MIT_6034 = SourceScope("ocw.mit.edu", "/courses/6-034-artificial-intelligence-")
REDBLOB_PATHFINDING = SourceScope("redblobgames.com", "/pathfinding/")
STANFORD_AI = SourceScope("plato.stanford.edu", "/entries/artificial-intelligence/")

PILOT_SOURCE_SCOPES: dict[str, tuple[SourceScope, ...]] = {
    # Introductory definitions of AI are covered by the same reviewed course
    # texts used for the search pilot.
    "AI-FND-01": (STANFORD_AI, CS50_AI, CS188_TEXTBOOK, MIT_6034),
    # Problem formulation and search representation are textbook definitions,
    # so the two course textbooks first.
    "AI-SRC-01": (CS188_TEXTBOOK, CS50_AI, MIT_6034),
    "AI-SRC-02": (CS188_TEXTBOOK, CS50_AI, MIT_6034),
    # Heuristics are where Red Blob Games explains better than the textbooks.
    "AI-SRC-08": (REDBLOB_PATHFINDING, CS188_TEXTBOOK, MIT_6034),
}

# Retrieved text is third-party material under review, so it defaults to an
# ignored directory: committing a review trail is a decision, not a side
# effect of running the CLI. Point --store elsewhere to keep one in git.
DEFAULT_STORE_PATH = Path("outputs/reference_candidates.json")


def load_pilot_catalogue() -> SkillCatalogue:
    return load_skills(*course_paths("ai"))


def pilot_skills(
    catalogue: SkillCatalogue,
    skill_ids: Sequence[str] = PILOT_SKILL_IDS,
) -> list[SkillDefinition]:
    by_id = {skill.skill_id: skill for skill in catalogue.skills}

    return [by_id[skill_id] for skill_id in skill_ids]


def scopes_for(skill_id: str) -> tuple[SourceScope, ...]:
    return PILOT_SOURCE_SCOPES.get(skill_id, ())


def domains_for(skill_id: str) -> tuple[str, ...]:
    """This skill's scoped domains first, then the rest of the allowlist.

    The scopes name paths and the search index is asked by domain, so what
    comes out of them here is the domains they sit on, in their order. The
    paths still do their work in the relevance gate, where a result is judged
    against the URL it actually came back with.

    The allowlist stays the boundary - this only decides what gets asked
    first, and every scope has to be on it.
    """
    preferred = tuple(dict.fromkeys(scope.domain for scope in scopes_for(skill_id)))

    return preferred + tuple(
        domain for domain in PILOT_ALLOWED_DOMAINS if domain not in preferred
    )


def plan_pilot(
    catalogue: SkillCatalogue,
    skill_ids: Sequence[str] = PILOT_SKILL_IDS,
) -> dict[str, list[str]]:
    """The queries the run would issue, per skill, without issuing them."""
    return {
        skill.skill_id: build_search_queries(skill)
        for skill in pilot_skills(catalogue, skill_ids)
    }


def run_pilot(
    catalogue: SkillCatalogue,
    provider: SearchProvider,
    fetcher: PageFetcher,
    allowed_domains: Sequence[str] | None = None,
    limit: int = SEARCH_LIMIT,
    clock: Callable[[], datetime] = utc_now,
    diagnostics: RetrievalDiagnostics | None = None,
    by_skill: dict[str, RetrievalDiagnostics] | None = None,
    known: Iterable[ReferenceCandidate] = (),
    skill_ids: Sequence[str] = PILOT_SKILL_IDS,
) -> list[ReferenceCandidate]:
    """Collect pending candidates for the pilot skills. Nothing is written.

    Each skill searches its own priority order unless one list is given for
    the whole run, and each keeps its own diagnostics record, which by_skill
    collects in pilot order. The run total is the sum of them: a pilot where
    two skills fill up and one finds nothing reads as an average otherwise,
    and the average is the one number that describes no skill in the run.

    Each skill also gets its own budget, so a skill that spends everything
    looking cannot leave the next one unsearched.

    known is whatever the store already holds - pass the store's contents to
    run against it a second time without paying for the first run's pages
    again. Nothing here reads or writes the store itself.
    """
    diagnostics = diagnostics if diagnostics is not None else RetrievalDiagnostics()
    by_skill = by_skill if by_skill is not None else {}
    known = list(known)

    candidates: list[ReferenceCandidate] = []

    for skill in pilot_skills(catalogue, skill_ids):
        for_skill = RetrievalDiagnostics()
        by_skill[skill.skill_id] = for_skill

        candidates += retrieve_candidates(
            skill,
            provider,
            fetcher,
            allowed_domains or domains_for(skill.skill_id),
            limit=limit,
            clock=clock,
            diagnostics=for_skill,
            known=known_for(skill.skill_id, known),
            scopes=scopes_for(skill.skill_id),
        )

        diagnostics.absorb(for_skill)

    return candidates
