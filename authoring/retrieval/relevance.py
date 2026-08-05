"""Whether a result is about the skill that went looking for it.

The allowlist says which sites may be read. It cannot say whether a page is on
topic, and the first HTML-first live pilot showed exactly that gap: fifteen
candidates off reviewed domains, of which two or three were about AI search at
all. MIT OpenCourseWare on its own supplied electromagnetic field theory,
computational mechanics of materials, probabilistic systems analysis, feedback
control and functional analysis - every one a real page on an approved domain,
and none of them able to ground "what is a transition model".

So relevance is a second layer, deliberately separate from the allowlist. The
allowlist stays the network boundary and decides what may be fetched; this
decides what is worth a reviewer's attention once it has been read. Nothing
here approves anything: a page that scores well is still pending, and the
score exists to remove the obvious false positives rather than to judge the
rest.

Scoring is term overlap and nothing cleverer, and it is written down rather
than inferred. A candidate carries the terms it matched and the total they
came to, so a reviewer can see why it survived - and so a threshold that turns
out to be wrong can be argued with from the record instead of from memory.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from urllib.parse import urlsplit

from authoring.retrieval.passage import MIN_TERM_LENGTH, STOPWORDS
from authoring.retrieval.safety import domain_is_allowed
from taxonomy.schemas import SkillDefinition

WORD = re.compile(r"[a-z0-9]+")

# The anchor every query carries. "Problem formulation", "state" and "problem
# solving" are the taxonomy's own words and they are not AI's: searched alone
# they returned a problem-solving approach to electromagnetic field theory and
# a state-space model of an aircraft, both correctly. The anchor is what makes
# the question the one the taxonomy is actually asking.
AI_CONTEXT_ANCHOR = "introduction to artificial intelligence"

# Vocabulary that marks a page as Introduction-to-AI search material. These
# are phrases rather than single words on purpose: "search", "state", "path"
# and "problem" appear on every technical page ever written, and it was
# precisely those words that let five unrelated MIT courses through. "state
# space" and "goal test" belong to this subject.
CONTEXT_TERMS = frozenset(
    {
        "artificial intelligence",
        "intelligent agent",
        "search problem",
        "search algorithm",
        "search strategy",
        "search tree",
        "search space",
        "state space",
        "graph search",
        "tree search",
        "informed search",
        "uninformed search",
        "heuristic",
        "admissible",
        "pathfinding",
        "path cost",
        "goal state",
        "goal test",
        "initial state",
        "transition model",
        "successor function",
        "breadth first",
        "depth first",
        "best first",
        "shortest path",
        "frontier",
    }
)

# The verb a learning objective opens with says what the learner does, not
# what the page has to contain. Every objective in the taxonomy starts with
# one, so scoring them would give every page the same free point.
BLOOM_VERBS = frozenset(
    {
        "classify",
        "compare",
        "construct",
        "define",
        "distinguish",
        "evaluate",
        "explain",
        "identify",
        "recommend",
        "trace",
    }
)

# What the skill is named after weighs most: it is the one phrase chosen to
# describe this skill and nothing else. The objective adds the detail a
# reference has to be able to support. Context says the page is about this
# subject at all, and a preferred scope is a path Albert has already looked at
# and vouched for.
CONCEPT_WEIGHT = 3
OBJECTIVE_WEIGHT = 2
CONTEXT_WEIGHT = 2
SCOPE_WEIGHT = 3

# Roughly: one context phrase, plus either two of the skill's own words or one
# of them and two from its objective.
#
# Set by scoring the live pilot's own fifteen candidates. The pages worth
# reviewing came in between 10 and 37; the ones a reviewer would throw out
# came in between 2 and 9, and the highest of them - electromagnetic field
# theory at 9 - got there without a single context phrase, which is what the
# second condition in is_relevant is for. Eight sits in the gap.
MIN_RELEVANCE_SCORE = 8


@dataclass(frozen=True)
class SourceScope:
    """One part of one site: the domain, and the path worth reading on it.

    ocw.mit.edu is one domain and two thousand courses; cs50.harvard.edu is
    one domain and a introductory-programming course sitting next to the AI
    one. A domain is the wrong unit for saying where the AI material is, and
    it is the only unit the allowlist has - which is why this is a separate
    layer rather than a longer allowlist.
    """

    domain: str
    path: str = ""

    def covers(self, url: str) -> bool:
        """Is this URL inside the scope? Host by whole labels, path by prefix.

        The host test is the allowlist's own, so www.redblobgames.com is
        inside a redblobgames.com scope and berkeley.edu.attacker.example is
        inside nothing. The trailing slash is trimmed off the prefix so that
        /~cs188/textbook and /~cs188/textbook/ name the same place.
        """
        if not domain_is_allowed(url, (self.domain,)):
            return False

        return urlsplit(url).path.lower().startswith(self.path.lower().rstrip("/"))

    def __str__(self) -> str:
        return f"{self.domain}{self.path}"


def singular(word: str) -> str:
    """Fold a trailing plural s away, so heuristics matches heuristic.

    Crude on purpose, and applied to both the term and the page, so the two
    are folded the same way even where the fold is wrong: "process" becomes
    "proces" on each side and still matches itself.
    """
    return word[:-1] if len(word) > MIN_TERM_LENGTH and word.endswith("s") else word


def tokens(text: str) -> list[str]:
    return [singular(word) for word in WORD.findall(text.casefold())]


def haystack(*parts: str) -> str:
    """The texts to match against, as padded streams of folded words.

    Each word is separated by one space and each part by more, which does two
    things. The padding means a phrase is matched as " goal test " and hits
    whole words only - otherwise "path" would match "pathological". The wider
    gap between parts means a phrase cannot be assembled across the seam
    between them: a title ending in "state" beside a snippet opening with
    "space" is not a page about state spaces.
    """
    return "".join(f"  {' '.join(tokens(part))}  " for part in parts)


def matches(term: str, hay: str) -> bool:
    return f" {' '.join(tokens(term))} " in hay


def terms_of(text: str) -> set[str]:
    """The words in a taxonomy field that could tell one page from another."""
    return {
        word
        for word in tokens(text)
        if len(word) >= MIN_TERM_LENGTH
        and word not in STOPWORDS
        and word not in BLOOM_VERBS
    }


def concept_terms(skill: SkillDefinition) -> set[str]:
    return terms_of(f"{skill.name} {skill.subtopic}")


def objective_terms(skill: SkillDefinition) -> set[str]:
    """The objective's own words, less the ones the concept already scores.

    Subtracted so that a word carried by both fields earns its points once:
    "search" is in AI-SRC-02's name and in its objective, and counting it
    twice would let a page about any kind of search halfway to the threshold
    on one word.
    """
    return terms_of(skill.learning_objective) - concept_terms(skill)


@dataclass(frozen=True)
class RelevanceScore:
    """A score, and the terms that earned it.

    The terms are the point. A number on its own tells a reviewer that
    something was filtered but not what the filter saw, and a threshold
    nobody can audit is a threshold nobody can correct.
    """

    score: int
    concept: tuple[str, ...] = ()
    objective: tuple[str, ...] = ()
    context: tuple[str, ...] = ()
    scope: str = ""

    @property
    def matched_terms(self) -> list[str]:
        """Every term that earned a point, labelled with what earned it."""
        labelled = [
            *(f"concept:{term}" for term in self.concept),
            *(f"objective:{term}" for term in self.objective),
            *(f"context:{term}" for term in self.context),
        ]

        return labelled + ([f"scope:{self.scope}"] if self.scope else [])

    def is_relevant(self, minimum: int = MIN_RELEVANCE_SCORE) -> bool:
        """Two conditions, because one of them cannot be traded away.

        The score alone is not enough. A page can climb on the taxonomy's
        generic words - "problem", "search", "state", "test" - and MIT's
        course pages did: electromagnetic field theory scored 9 for AI-SRC-01
        on "component", "problem" and "search" without containing one
        sentence about AI. So a result also has to show that it is about this
        subject at all, and no amount of overlap substitutes for that.
        """
        return bool(self.context) and self.score >= minimum


def score_relevance(
    skill: SkillDefinition,
    url: str,
    title: str,
    snippet: str,
    passage: str,
    scopes: Sequence[SourceScope] = (),
) -> RelevanceScore:
    """Score one result against the skill that searched for it.

    All three texts are read together: the title says what the page calls
    itself, the provider's snippet says what the index thought matched, and
    the passage is what a reviewer would actually be shown. A page that only
    matches in its title is a page whose body went somewhere else.
    """
    hay = haystack(title, snippet, passage)

    concept = tuple(sorted(term for term in concept_terms(skill) if matches(term, hay)))
    objective = tuple(
        sorted(term for term in objective_terms(skill) if matches(term, hay))
    )
    context = tuple(sorted(term for term in CONTEXT_TERMS if matches(term, hay)))
    scope = next((str(item) for item in scopes if item.covers(url)), "")

    score = (
        CONCEPT_WEIGHT * len(concept)
        + OBJECTIVE_WEIGHT * len(objective)
        + CONTEXT_WEIGHT * len(context)
        + SCOPE_WEIGHT * bool(scope)
    )

    return RelevanceScore(score, concept, objective, context, scope)
