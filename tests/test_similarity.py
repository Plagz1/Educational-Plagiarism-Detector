from src.similarity import lcs_similarity, ngram_similarity


def test_lcs():
    assert lcs_similarity("abc", "abc") == 1.0


def test_ngram():
    assert ngram_similarity("hello", "hello") == 1.0
