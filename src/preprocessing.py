import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

REQUIRED_RESOURCES = [
    "punkt",
    "punkt_tab",
    "wordnet",
    "stopwords",
]

for resource in REQUIRED_RESOURCES:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource)

lemmatizer = WordNetLemmatizer()
STOP_WORDS = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)

    tokens = nltk.word_tokenize(text)
    tokens = [
        lemmatizer.lemmatize(token)
        for token in tokens
        if token not in STOP_WORDS
    ]

    return " ".join(tokens)
