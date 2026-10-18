"""Splitting a list of results into pages."""


def paginate(items, page, per_page=10):
    """Return the items on a 1-based page. Pages past the end are empty.

    page and per_page must both be at least 1, otherwise ValueError is raised.
    """
    if page < 1 or per_page < 1:
        raise ValueError("page and per_page must be at least 1")
    start = page * per_page
    return items[start:start + per_page]


def page_count(n_items, per_page=10):
    """How many pages n_items fill. No items means no pages."""
    return n_items // per_page
