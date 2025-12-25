import re
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

stemmer = PorterStemmer()
STOP_WORDS = set(stopwords.words("english"))


def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = nltk.word_tokenize(text)
    tokens = [
        stemmer.stem(token)
        for token in tokens
        if token not in STOP_WORDS
    ]

    return " ".join(tokens)
