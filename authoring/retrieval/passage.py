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

# Punctuation that carries meaning in code and is rare in prose. Measured
# against the pages of the second pilot run once their links were stripped:
# prose came in between 0.003 and 0.019, source files between 0.036 and 0.076.
# The threshold sits in that gap, with room on both sides.
CODE_PUNCTUATION = set("{}[]<>;=|&*/\\_#")
CODE_DENSITY = 0.03

# Words a taxonomy query carries that say nothing about which page matches.
STOPWORDS = frozenset(
    {
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
    """The window of page text that best matches the query, up to max_chars.

    Falls back to the head of the page when the query matches nothing, so a
    page with no overlap is still shown to the reviewer rather than silently
    reshaped into something that looks relevant.
    """
    flat = strip_links(flatten(text))
    terms = query_terms(query)

    if not terms or len(flat) <= max_chars:
        return flat[:max_chars]

    sentences = SENTENCE.split(flat)
    scores = [
        len(terms & set(WORD.findall(sentence.casefold()))) for sentence in sentences
    ]

    # Highest overlap, earliest on the page when two sentences tie, so the
    # same page and query always yield the same quote.
    best = max(range(len(sentences)), key=lambda index: (scores[index], -index))

    if not scores[best]:
        return flat[:max_chars]

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
    """
    start = end = best
    passage = sentences[best]

    while True:
        widened = False
        forward, backward = end + 1, start - 1

        if (
            forward < len(sentences)
            and len(passage) + len(sentences[forward]) < max_chars
            and (scores[forward] or len(passage) < min_chars)
        ):
            end = forward
            passage = f"{passage} {sentences[end]}"
            widened = True

        if (
            backward >= 0
            and len(sentences[backward]) + len(passage) < max_chars
            and (scores[backward] or len(passage) < min_chars)
        ):
            start = backward
            passage = f"{sentences[start]} {passage}"
            widened = True

        if not widened:
            return passage[:max_chars]
