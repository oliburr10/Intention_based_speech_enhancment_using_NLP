import pytest

from pager import page_count, paginate

ITEMS = list(range(25))


@pytest.mark.parametrize("page, expected", [
    (1, list(range(0, 10))),
    (2, list(range(10, 20))),
    (3, list(range(20, 25))),
    (4, []),
])
def test_pages(page, expected):
    assert paginate(ITEMS, page) == expected


def test_custom_page_size():
    assert paginate(ITEMS, 2, per_page=4) == [4, 5, 6, 7]


@pytest.mark.parametrize("n, per_page, expected", [(0, 10, 0), (1, 10, 1), (10, 10, 1), (20, 10, 2), (21, 10, 3), (7, 3, 3)])
def test_page_count(n, per_page, expected):
    assert page_count(n, per_page) == expected


@pytest.mark.parametrize("page, per_page", [(0, 10), (-1, 10), (1, 0)])
def test_invalid_arguments(page, per_page):
    with pytest.raises(ValueError):
        paginate(ITEMS, page, per_page)
