"""Small helpers for working with titles and text."""


def word_count(text):
    """Number of whitespace-separated words in text."""
    return len(text.split())


def slugify(title, max_length=50):
    """Turn a title into a URL slug.

    - Lowercase, with accents removed ("Café" -> "cafe").
    - Every run of characters that aren't letters or digits becomes a single hyphen.
    - No hyphens at the start or the end.
    - At most max_length characters. If the slug is too long, cut it at the last hyphen
      that fits so no word is cut in half; only if there is no such hyphen, cut mid-word.
      Never leave a hyphen at the end.
    - If nothing is left, return "untitled".
    """
    raise NotImplementedError
