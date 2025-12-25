from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from difflib import SequenceMatcher


def cosine_similarity_score(texts: list[str]) -> list[list[float]]:
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)
    return cosine_similarity(tfidf)


def lcs_similarity(a: str, b: str) -> float:
    matcher = SequenceMatcher(None, a, b)
    return matcher.ratio()


def ngram_similarity(a: str, b: str, n: int = 3) -> float:
    def ngrams(text):
        return set(
            text[i:i+n]
            for i in range(len(text) - n + 1)
        )

    ngrams_a = ngrams(a)
    ngrams_b = ngrams(b)

    intersection = ngrams_a & ngrams_b
    union = ngrams_a | ngrams_b

    return len(intersection) / len(union) if union else 0.0
