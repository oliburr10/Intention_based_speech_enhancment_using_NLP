import pytest

from pager import page_count, paginate


def test_first_page():
    assert paginate(list(range(25)), 1) == list(range(10))


def test_page_count_with_partial_last_page():
    assert page_count(25) == 3


def test_rejects_page_zero():
    with pytest.raises(ValueError):
        paginate([1, 2, 3], 0)
