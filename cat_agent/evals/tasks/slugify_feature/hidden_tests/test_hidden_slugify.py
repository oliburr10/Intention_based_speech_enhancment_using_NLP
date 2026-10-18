import pytest

from text_utils import slugify, word_count


@pytest.mark.parametrize("title, slug", [
    ("Hello World", "hello-world"),
    ("Café au lait", "cafe-au-lait"),
    ("  Spaces   everywhere  ", "spaces-everywhere"),
    ("C++ & Python: a love story!", "c-python-a-love-story"),
    ("Version 2.0 released", "version-2-0-released"),
    ("---already---hyphenated---", "already-hyphenated"),
    ("Naïve résumé", "naive-resume"),
])
def test_basic_slugs(title, slug):
    assert slugify(title) == slug


@pytest.mark.parametrize("title", ["", "   ", "!!!", "---"])
def test_empty_becomes_untitled(title):
    assert slugify(title) == "untitled"


def test_fits_exactly():
    assert slugify("hello world", max_length=11) == "hello-world"


def test_cuts_at_last_hyphen_that_fits():
    assert slugify("hello wonderful world", max_length=12) == "hello"
    assert slugify("hello world again", max_length=11) == "hello-world"
    assert slugify("hello world again", max_length=14) == "hello-world"


def test_cuts_mid_word_only_when_it_must():
    assert slugify("supercalifragilistic", max_length=5) == "super"


def test_default_max_length():
    slug = slugify("word " * 30)
    assert len(slug) <= 50 and not slug.endswith("-")


def test_word_count_still_works():
    assert word_count("a b c") == 3
