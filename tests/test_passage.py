"""Which part of a page gets quoted.

The first live pilot run quoted the top of every page and collected a table of
contents and two copies of a Python file. These tests are written against that
shape of page: the answer is present, but not in the first 1200 characters.
"""

import pytest

from authoring.retrieval.models import MAX_PASSAGE_CHARS
from authoring.retrieval.passage import (
    PROSE_RATIO,
    is_code_dense,
    looks_like_source,
    prose_ratio,
    query_terms,
    reads_as_prose,
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


def test_a_page_with_no_overlap_falls_back_to_its_first_prose():
    """Not to its head, which on every course page is the menu."""
    page = FILLER * 12 + ANSWER + " " + FILLER * 12

    passage = select_passage(page, "photosynthesis chlorophyll")

    assert passage.startswith(ANSWER[:40])


def test_a_page_of_nothing_but_navigation_has_nothing_to_quote():
    """FILLER is a table of contents, and a table of contents explains nothing.

    Returning it was how a reviewer ended up looking at MIT's course sidebar
    under the heading of a passage about heuristics. An empty answer here is
    what the caller turns into a rejection.
    """
    assert select_passage(FILLER * 30, QUERY) == ""
    assert select_passage(FILLER * 30, "photosynthesis chlorophyll") == ""


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
    page = "A heuristic estimates the remaining cost to the goal, and " * 40

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


# Navigation is not writing: the two shapes the second live run quoted


# One window, because nothing in it ends a sentence until the very end -
# which is how MIT's course sidebar reaches select_passage. It names three of
# the query's terms in passing, so under the old rule it outscored every real
# sentence on the page simply by being ten times longer than one.
SIDEBAR = (
    "Browse Course Material Syllabus Calendar Instructor Insights Teaching "
    "Heuristic Search Experiencing the Large Lecture as Theater Assessment "
    "Informed by a Student-Centered Ethic Managing an Online Forum Challenges "
    "Teaching Assistants Face Readings Lecture Videos Mega-Recitation Videos "
    "Tutorials Assignments Exams Demonstrations Course Info Estimates of Cost "
    "Evaluation Function Departments Electrical Engineering and Computer "
    "Science Learning Resource Types Lecture Notes Problem Sets Exams "
    "Download Course menu search Give Now About OCW Help Faqs Contact Us."
)

CONTENTS = (
    "1. Search 1.1 Agents 1.2 State Spaces and Search Problems 1.3 Uninformed "
    "Search 1.4 Informed Search 1.5 Local Search 1.6 Summary 2."
)


def test_a_long_unpunctuated_sidebar_does_not_win_on_term_count():
    """MIT's lecture pages run their whole sidebar together as one window.

    It collected more distinct query terms than any real sentence, because it
    is two hundred words long and a sentence is twenty. Length is not
    relevance.
    """
    page = f"{SIDEBAR} {ANSWER} {SIDEBAR}"

    passage = select_passage(page, QUERY, max_chars=300)

    assert passage.startswith(ANSWER[:40])
    assert "Browse Course Material" not in passage


def test_a_table_of_contents_does_not_win_a_tie_by_being_first():
    """Ten CS188 windows tied on two terms and the earliest took it.

    A table of contents is always the earliest thing on a page, so breaking
    ties by position handed it every tie it entered.
    """
    page = f"{CONTENTS} {FILLER * 4} {ANSWER}"

    passage = select_passage(page, "Search problem components Problem formulation")

    assert passage.startswith(ANSWER[:40])


def test_prose_is_told_from_navigation_by_the_words_holding_it_together():
    assert reads_as_prose(ANSWER)
    assert not reads_as_prose(CONTENTS)
    assert not reads_as_prose(SIDEBAR)
    assert not reads_as_prose(FILLER)


def test_a_fragment_too_short_to_judge_is_not_prose():
    """"of the search" is two thirds function words and says nothing."""
    assert not reads_as_prose("of the search")
    assert not reads_as_prose("")


def test_the_measured_gap_is_where_the_threshold_sits():
    """The live run's own passages, either side of it."""
    assert prose_ratio(SIDEBAR) < PROSE_RATIO < prose_ratio(ANSWER)


def test_growth_stops_at_the_edge_of_the_prose():
    """A short quote is padded to the minimum, but never with a menu."""
    page = f"{FILLER * 6} {ANSWER} {FILLER * 6}"

    passage = select_passage(page, QUERY, min_chars=600)

    assert "Instructors Pseudocode" not in passage
