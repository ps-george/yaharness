"""Tests for the extended GAIA normalisation / matching.

Mirrors the official GAIA grading spec: lowercase, articles, punctuation,
number-words → digits, set-comparison for comma-separated lists.
"""

from __future__ import annotations

import pytest

from yaharness.benchmarks.gaia import gaia_answers_match, gaia_normalise

# 10+ canonical pairs covering each rule + several combinations.
EQUIVALENT_PAIRS: list[tuple[str, str, str]] = [
    ("article", "The Eiffel Tower", "Eiffel Tower"),
    ("article-an", "An apple", "apple"),
    ("punct", "Paris.", "Paris"),
    ("punct-multi", "Paris!!", "paris"),
    ("case", "PARIS", "paris"),
    ("whitespace", "  hello   world  ", "hello world"),
    ("number-word-unit", "twelve", "12"),
    ("number-word-compound", "twenty-one", "21"),
    ("number-word-large", "one hundred", "100"),
    ("number-word-thousands", "two thousand and three", "2003"),
    ("article+number", "The twelve apostles", "12 apostles"),
    ("article+punct", "The cat.", "cat"),
    ("zero", "zero", "0"),
    ("scale-mixed", "three hundred and twenty-one", "321"),
]


@pytest.mark.parametrize(("name", "lhs", "rhs"), EQUIVALENT_PAIRS)
def test_gaia_normalise_equivalence(name: str, lhs: str, rhs: str) -> None:
    assert gaia_normalise(lhs) == gaia_normalise(rhs), name


def test_gaia_normalise_distinguishes_different() -> None:
    assert gaia_normalise("Paris") != gaia_normalise("London")
    assert gaia_normalise("twelve") != gaia_normalise("13")
    assert gaia_normalise("twenty one") == "21"


def test_gaia_normalise_does_not_mangle_non_number_words() -> None:
    assert gaia_normalise("Alice") == "alice"
    # "and" alone is not a number and must not become "0".
    assert gaia_normalise("salt and pepper") == "salt and pepper"


# ---- set-comparison via gaia_answers_match -------------------------------


def test_gaia_match_set_comma_separated_order_independent() -> None:
    assert gaia_answers_match("Paris, France", "France, Paris")
    assert gaia_answers_match("apple, banana, cherry", "cherry, apple, banana")


def test_gaia_match_set_with_normalisation_per_element() -> None:
    # Articles and number-words still normalised per element.
    assert gaia_answers_match("The Eiffel Tower, twelve", "12, Eiffel Tower")


def test_gaia_match_scalar_path() -> None:
    assert gaia_answers_match("twelve", "12")
    assert gaia_answers_match("The cat.", "cat")
    assert not gaia_answers_match("Paris", "London")


def test_gaia_match_set_inequality() -> None:
    assert not gaia_answers_match("Paris, France", "London, France")
    assert not gaia_answers_match("a, b", "a, b, c")


def test_gaia_match_scalar_vs_list_no_false_positive() -> None:
    # If only one side is comma-separated, fall back to scalar exact-match
    # (which will likely mismatch — that's the intended conservative path).
    assert not gaia_answers_match("Paris, France", "Paris")
