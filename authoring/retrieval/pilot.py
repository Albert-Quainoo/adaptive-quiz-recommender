"""The first retrieval-assisted authoring run: AI-SRC-01, AI-SRC-02, AI-SRC-08.

Three generated, understand-level search skills that carry no reference
material yet, which is what makes them the honest test of the workflow.

This module plans the run and can execute it against an injected provider. It
writes nothing: references.csv stays as it is until the candidates it produces
have been read and approved, and which domains are trustworthy for a course is
a review decision, not one retrieval gets to make for itself.
"""

from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

from authoring.retrieval.diagnostics import RetrievalDiagnostics
from authoring.retrieval.models import ReferenceCandidate, utc_now
from authoring.retrieval.search import (
    SEARCH_LIMIT,
    PageFetcher,
    SearchProvider,
    build_search_queries,
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
    "ai.berkeley.edu",  # Berkeley CS188, Introduction to AI
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
)

# One allowlist cannot be ordered well for every skill. Red Blob Games is the
# best source in the list for how a heuristic guides a search, and one of the
# worst for what a transition model is - it teaches grid pathfinding, not
# problem formulation. The first ordered run showed exactly that: twenty-one
# candidates, two of them usable, all for the heuristics skill.
#
# So the priority order is per skill. These are preferences, not restrictions:
# domains_for appends the rest of the allowlist behind them, so a skill whose
# preferred sites come up thin can still reach the others.
PILOT_SKILL_DOMAINS: dict[str, tuple[str, ...]] = {
    # Problem formulation and search representation are textbook definitions,
    # so course notes first.
    "AI-SRC-01": ("ai.berkeley.edu", "ocw.mit.edu", "cs50.harvard.edu"),
    "AI-SRC-02": ("ai.berkeley.edu", "ocw.mit.edu", "cs50.harvard.edu"),
    # Heuristics are where Red Blob Games explains better than the textbooks.
    "AI-SRC-08": ("redblobgames.com", "ai.berkeley.edu", "ocw.mit.edu"),
}

# Retrieved text is third-party material under review, so it defaults to an
# ignored directory: committing a review trail is a decision, not a side
# effect of running the CLI. Point --store elsewhere to keep one in git.
DEFAULT_STORE_PATH = Path("outputs/reference_candidates.json")


def load_pilot_catalogue() -> SkillCatalogue:
    return load_skills(*course_paths("ai"))


def pilot_skills(catalogue: SkillCatalogue) -> list[SkillDefinition]:
    by_id = {skill.skill_id: skill for skill in catalogue.skills}

    return [by_id[skill_id] for skill_id in PILOT_SKILL_IDS]


def domains_for(skill_id: str) -> tuple[str, ...]:
    """This skill's preferred domains first, then the rest of the allowlist.

    The allowlist stays the boundary - this only decides what gets asked
    first, and every preference has to be on it.
    """
    preferred = PILOT_SKILL_DOMAINS.get(skill_id, ())

    return preferred + tuple(
        domain for domain in PILOT_ALLOWED_DOMAINS if domain not in preferred
    )


def plan_pilot(catalogue: SkillCatalogue) -> dict[str, list[str]]:
    """The queries the run would issue, per skill, without issuing them."""
    return {
        skill.skill_id: build_search_queries(skill)
        for skill in pilot_skills(catalogue)
    }


def run_pilot(
    catalogue: SkillCatalogue,
    provider: SearchProvider,
    fetcher: PageFetcher,
    allowed_domains: Sequence[str] | None = None,
    limit: int = SEARCH_LIMIT,
    clock: Callable[[], datetime] = utc_now,
    diagnostics: RetrievalDiagnostics | None = None,
) -> list[ReferenceCandidate]:
    """Collect pending candidates for the pilot skills. Nothing is written.

    Each skill searches its own priority order unless one list is given for
    the whole run. One diagnostics record covers the run either way, so the
    counts describe the pilot rather than whichever skill was searched last.
    """
    diagnostics = diagnostics if diagnostics is not None else RetrievalDiagnostics()

    return [
        candidate
        for skill in pilot_skills(catalogue)
        for candidate in retrieve_candidates(
            skill,
            provider,
            fetcher,
            allowed_domains or domains_for(skill.skill_id),
            limit=limit,
            clock=clock,
            diagnostics=diagnostics,
        )
    ]
