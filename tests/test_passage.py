"""Which part of a page gets quoted.

The first live pilot run quoted the top of every page and collected a table of
contents and two copies of a Python file. These tests are written against that
shape of page: the answer is present, but not in the first 1200 characters.
"""

import pytest

from authoring.retrieval.models import MAX_PASSAGE_CHARS
from authoring.retrieval.passage import (
    is_code_dense,
    looks_like_source,
    query_terms,
    select_passage,
    strip_links,
)

QUERY = "Heuristic function Explain how a heuristic estimates the remaining cost"

FILLER = (
    "Home Editions Errata Exercises Figures Instructors Pseudocode Reviews "
    "Chapter 1 Introduction 1 Chapter 2 Intelligent Agents 36 Chapter 3 "
    "Solving Problems by Searching 63 Chapter 4 Search in Complex "
    "Environments 110 Chapter 5 Adversarial Search and Games 146. "
)

ANSWER = (
    "A heuristic function estimates the cost of the cheapest path from the "
    "state at a node to a goal state."
)


def page_with_answer_below_the_fold() -> str:
    return FILLER * 12 + ANSWER + " " + FILLER * 12


# Term extraction


def test_terms_come_from_the_query_without_its_filler_words():
    assert query_terms(QUERY) == {
        "heuristic",
        "function",
        "explain",
        "estimates",
        "remaining",
        "cost",
    }


def test_punctuation_and_case_do_not_make_new_terms():
    assert query_terms("Goal test, PATH COST.") == {"goal", "test", "path", "cost"}


# Selection


def test_the_answer_is_quoted_even_when_it_is_far_down_the_page():
    passage = select_passage(page_with_answer_below_the_fold(), QUERY)

    assert ANSWER in passage
    assert len(passage) <= MAX_PASSAGE_CHARS


def test_the_head_of_the_page_is_not_what_gets_quoted():
    page = page_with_answer_below_the_fold()

    assert select_passage(page, QUERY) != page[:MAX_PASSAGE_CHARS]


def test_relevant_neighbours_come_along_with_the_answer():
    elaboration = (
        "A heuristic that never overestimates the remaining cost is called "
        "admissible."
    )
    page = FILLER * 12 + ANSWER + " " + elaboration + " " + FILLER * 12

    assert elaboration in select_passage(page, QUERY)


def test_the_quote_is_not_padded_out_with_navigation():
    """Filling the budget regardless is how the first live run quoted a menu."""
    passage = select_passage(page_with_answer_below_the_fold(), QUERY)

    assert passage.startswith(ANSWER[:40])
    assert passage.count("Instructors Pseudocode") <= 1


def test_a_short_page_is_quoted_whole():
    assert select_passage(f"  {ANSWER}\n\n", QUERY) == ANSWER


def test_a_page_with_no_overlap_falls_back_to_its_head():
    page = FILLER * 30

    assert select_passage(page, "photosynthesis chlorophyll") == page[:MAX_PASSAGE_CHARS]


def test_selection_is_deterministic():
    page = page_with_answer_below_the_fold()

    assert select_passage(page, QUERY) == select_passage(page, QUERY)


def test_a_tie_is_broken_towards_the_start_of_the_page():
    sentence = "A heuristic estimates the remaining cost to the goal."
    page = f"{sentence} {FILLER * 20} {sentence} {FILLER * 20}"

    assert select_passage(page, QUERY, max_chars=200).startswith(sentence)


def test_the_quote_never_exceeds_the_limit():
    page = page_with_answer_below_the_fold()

    for limit in (120, 400, MAX_PASSAGE_CHARS):
        assert len(select_passage(page, QUERY, max_chars=limit)) <= limit


def test_one_long_sentence_is_cut_to_the_limit():
    page = "A heuristic " + "estimates remaining cost " * 200

    assert len(select_passage(page, QUERY, max_chars=300)) == 300


# Links are not reading text

FOOTER = (
    "Email me redblobgames@gmail.com , or comment here: Links [1]: "
    "https://mikolalysenko.github.io/l1-path-finder/www/ [2]: "
    "http://users.cecs.anu.edu.au/~dharabor/data/papers/harabor-icaps14.pdf"
)


def test_a_link_footer_does_not_outrank_the_prose_above_it():
    """Two of the first Red Blob candidates were exactly this footer."""
    page = FILLER * 6 + ANSWER + " " + FILLER * 6 + FOOTER

    passage = select_passage(page, QUERY)

    assert ANSWER in passage
    assert "mikolalysenko" not in passage


def test_urls_and_addresses_are_stripped_from_the_quote():
    assert "http" not in strip_links(FOOTER)
    assert "redblobgames@gmail.com" not in strip_links(FOOTER)
    assert "Email me" in strip_links(FOOTER)


# Pages that are not prose


@pytest.mark.parametrize(
    "url",
    [
        "https://www.redblobgames.com/pathfinding/a-star/implementation.py",
        "https://www.redblobgames.com/pathfinding/a-star/implementation.org",
        "http://aima.cs.berkeley.edu/python/search.py",
        "https://example.edu/notes/Graph.java",
    ],
)
def test_source_files_are_recognised_by_their_url(url):
    assert looks_like_source(url)


@pytest.mark.parametrize(
    "url",
    [
        "https://www.redblobgames.com/pathfinding/a-star/introduction.html",
        "https://ai.berkeley.edu/search.html",
        "https://ocw.mit.edu/courses/6-034/",
    ],
)
def test_pages_are_not_mistaken_for_source_files(url):
    assert not looks_like_source(url)


def test_code_is_recognised_by_its_punctuation():
    cpp = (
        "inline double heuristic(GridLocation a, GridLocation b) { return "
        "std::abs(a.x - b.x) + std::abs(a.y - b.y); } template void "
        "a_star_search (Graph graph, Location start, std::unordered_map & "
        "came_from, std::unordered_map & cost_so_far) { PriorityQueue frontier; }"
    )

    assert is_code_dense(cpp)


def test_prose_is_not_mistaken_for_code():
    assert not is_code_dense(ANSWER + " " + FILLER)
    assert not is_code_dense("")


def test_an_org_mode_header_is_recognised_as_markup():
    assert is_code_dense(
        "#+title: Implementation of A* #+DATE: #+options: toc:2 #+property: "
        "header-args :exports both :results output :wrap example :eval never-export"
    )


def test_a_page_of_links_is_not_mistaken_for_code():
    """URL punctuation put two prose pages over the threshold before."""
    assert not is_code_dense(ANSWER + " " + FOOTER)
