from text_utils import word_count


def test_word_count():
    assert word_count("the quick  brown fox") == 4
