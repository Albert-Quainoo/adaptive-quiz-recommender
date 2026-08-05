"""Choosing which part of a page to quote.

Taking the first 1200 characters of a page quotes its navigation, its imports
or its chapter list - which is what the first live pilot run collected. The
relevant sentences are usually further down, so the passage is cut around the
sentence that best answers the query that found the page.

Selection is by term overlap and nothing cleverer. It decides what a reviewer
is shown, not whether the page is any good: the passage still arrives pending,
and a well-chosen quote from a useless page is still a useless reference.
"""

import re
from urllib.parse import urlsplit

from authoring.retrieval.models import (
    MAX_PASSAGE_CHARS,
    MIN_PASSAGE_CHARS,
    flatten,
)

WORD = re.compile(r"[a-z0-9]+")
SENTENCE = re.compile(r"(?<=[.!?])\s+")

MIN_TERM_LENGTH = 4

# A URL is full of words that look like query terms - l1-path-finder scores
# for "path" and "finder" - so a footer of links outranks the prose above it.
# The first Red Blob run quoted two such footers. Links are not reading text
# and are removed before anything is scored.
LINK = re.compile(r"(https?://|www\.)\S+|\S+@\S+\.\S+|\[\d+\]:?")

# Source files served as text/plain pass every check the fetcher makes, so
# the shape of the URL is the cheapest place to catch them - and it catches
# them before the fetch rather than after.
SOURCE_SUFFIXES = (
    ".py",
    ".org",
    ".lisp",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".js",
    ".ts",
    ".rs",
    ".go",
    ".rb",
)

# Words that hold an English sentence together. Prose cannot be written
# without them; a navigation menu, a table of contents and a course sidebar
# are lists of nouns and contain almost none.
#
# This is what tells "1.2 State Spaces and Search Problems 1.3 Uninformed
# Search" apart from "The computation performed by such a function is
# specific to the search problem being solved." Both are about search, both
# match the same query terms, and only one of them is writing.
FUNCTION_WORDS = frozenset(
    """
    a an the this that these those of in on at to for from by with without
    into over under is are was were be been being do does did has have had
    can could will would may might must it its they them their we our you
    your he she his her not no nor and or but if then than as so such which
    who whom whose what when where while because although however therefore
    there here about between among during before after above below up down
    out off again
    """.split()
)

# Measured on the live run's own passages: the seven that turned out to be
# navigation scored between 0.031 and 0.122, and the eight worth reading
# between 0.317 and 0.525. The threshold sits in the middle of that gap.
PROSE_RATIO = 0.20

# Below this a window is too short to tell prose from a menu item: "of the
# search" is two thirds function words and says nothing.
MIN_PROSE_WORDS = 8

# Punctuation that carries meaning in code and is rare in prose. Measured
# against the pages of the second pilot run once their links were stripped:
# prose came in between 0.003 and 0.019, source files between 0.036 and 0.076.
# The threshold sits in that gap, with room on both sides.
CODE_PUNCTUATION = set("{}[]<>;=|&*/\\_#")
CODE_DENSITY = 0.03

# Words a taxonomy query carries that say nothing about which page matches.
#
# The last three are the words build_search_queries anchors every query with
# (relevance.AI_CONTEXT_ANCHOR). They steer the index towards the right
# subject, which is their whole job, but every query carries them equally - so
# scoring them here would only pull the quote towards whichever sentence
# introduces the course. test_the_anchor_cannot_steer_which_sentence_is_quoted
# holds the two lists together.
STOPWORDS = frozenset(
    {
        "artificial",
        "intelligence",
        "introduction",
        "and",
        "between",
        "each",
        "from",
        "into",
        "such",
        "than",
        "that",
        "them",
        "then",
        "they",
        "this",
        "using",
        "what",
        "when",
        "which",
        "with",
    }
)


def looks_like_source(url: str) -> bool:
    """A file of code, whatever content type it was served as."""
    return urlsplit(url).path.lower().endswith(SOURCE_SUFFIXES)


def is_code_dense(text: str) -> bool:
    """Text carrying more punctuation than prose does.

    A page of C++ or org-mode markup reads as noise in a reference passage
    even when its words match the query. Links are stripped before measuring,
    because a footer of URLs is punctuation-heavy without being code - that
    alone put two prose pages above the threshold.
    """
    sample = strip_links(flatten(text))[:2000]

    if not sample:
        return False

    return (
        sum(character in CODE_PUNCTUATION for character in sample) / len(sample)
        > CODE_DENSITY
    )


def strip_links(text: str) -> str:
    return " ".join(LINK.sub(" ", text).split())


def prose_ratio(text: str) -> float:
    """How much of this text is the words that hold a sentence together."""
    words = WORD.findall(text.casefold())

    if not words:
        return 0.0

    return sum(word in FUNCTION_WORDS for word in words) / len(words)


def reads_as_prose(text: str) -> bool:
    """Is this writing, or is it a list of links someone can click?

    Asked of one window rather than of a whole page, because the two are not
    the same question. Every course page carries a menu, a breadcrumb trail
    and a footer, so a page that is 20% navigation still holds the paragraph
    worth quoting - and MIT's instructor-insights page, which holds nothing
    worth quoting, scores higher across the whole document than a page that
    does. What can be judged is the window about to be quoted.
    """
    return (
        len(text.split()) >= MIN_PROSE_WORDS and prose_ratio(text) >= PROSE_RATIO
    )


def query_terms(query: str) -> set[str]:
    return {
        word
        for word in WORD.findall(query.casefold())
        if len(word) >= MIN_TERM_LENGTH and word not in STOPWORDS
    }


def select_passage(
    text: str,
    query: str,
    max_chars: int = MAX_PASSAGE_CHARS,
    min_chars: int = MIN_PASSAGE_CHARS,
) -> str:
    """The best-matching window of prose on the page, up to max_chars.

    Only windows that read as prose can be quoted. The old rule - highest
    term overlap, earliest on the page to break a tie - picked navigation
    twice over, and for two different reasons. On MIT's lecture pages the
    whole sidebar is one unpunctuated 196-word run, so it collected more
    distinct query terms than any real sentence could and won outright. On
    the CS188 textbook ten windows tied on two terms each and the tie went to
    the earliest, which is the table of contents, because a table of contents
    is always at the top. Neither is a near miss that a weight would fix:
    both windows are lists of headings, and no list of headings should be
    quoted to a reviewer as though it explained something.

    Returns "" when the page holds no prose at all. That is a page with
    nothing to quote rather than a page quoted badly, and the caller drops it
    on the passage-length check.
    """
    flat = strip_links(flatten(text))
    terms = query_terms(query)

    # A page short enough to quote whole is quoted whole - but only if it
    # reads as writing. cs50's week-0 index is 1400 characters of menu and
    # nothing else, and returning it entire is how it reached a reviewer.
    if len(flat) <= max_chars:
        return flat if reads_as_prose(flat) else ""

    sentences = SENTENCE.split(flat)
    readable = [
        index
        for index, sentence in enumerate(sentences)
        if reads_as_prose(sentence)
    ]

    if not readable:
        return ""

    if not terms:
        return grow_window(sentences, [0] * len(sentences), readable[0], max_chars, min_chars)

    scores = [
        len(terms & set(WORD.findall(sentence.casefold()))) for sentence in sentences
    ]

    # Highest overlap, earliest on the page when two windows tie, so the same
    # page and query always yield the same quote.
    best = max(readable, key=lambda index: (scores[index], -index))

    # Nothing matched, so quote where the page starts explaining rather than
    # where it starts - the reviewer still sees a page with no overlap, and
    # sees it as prose rather than as a menu.
    if not scores[best]:
        best = readable[0]

    return grow_window(sentences, scores, best, max_chars, min_chars)


def grow_window(
    sentences: list[str],
    scores: list[int],
    best: int,
    max_chars: int,
    min_chars: int,
) -> str:
    """Widen from the best sentence for as long as the context is relevant.

    A lone sentence rarely reads as an explanation, so its neighbours come
    with it - but only while they still match the query. Filling the whole
    budget regardless is how a passage ends up being mostly the navigation
    that happened to surround the one useful line.

    The exception is a window still too short to survive the minimum, which
    keeps growing: a reviewer can judge a padded quote, but a quote thrown
    away for being short never reaches them at all.

    Growth stops at anything that is not prose either way. Padding a short
    quote up to the minimum is what would otherwise reach past the end of a
    section and pull the next menu in with it, which is the same fault the
    selection above exists to avoid.
    """
    start = end = best
    passage = sentences[best]

    while True:
        widened = False
        forward, backward = end + 1, start - 1

        if (
            forward < len(sentences)
            and len(passage) + len(sentences[forward]) < max_chars
            and reads_as_prose(sentences[forward])
            and (scores[forward] or len(passage) < min_chars)
        ):
            end = forward
            passage = f"{passage} {sentences[end]}"
            widened = True

        if (
            backward >= 0
            and len(sentences[backward]) + len(passage) < max_chars
            and reads_as_prose(sentences[backward])
            and (scores[backward] or len(passage) < min_chars)
        ):
            start = backward
            passage = f"{sentences[start]} {passage}"
            widened = True

        if not widened:
            return passage[:max_chars]
