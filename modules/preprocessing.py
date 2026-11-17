import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize


def ensure_nltk_data():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
    ]
    for path, package in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(package, quiet=True)


def preprocess_text(text):
    ensure_nltk_data()
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words("english"))
    return " ".join(token for token in tokens if token.isalpha() and token not in stop_words)


def preprocessing_comparison(text):
    ensure_nltk_data()
    tokens = [t for t in word_tokenize(text) if t.isalpha()]
    stop_words = set(stopwords.words("english"))
    filtered = [t.lower() for t in tokens if t.lower() not in stop_words]
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    return [
        {"Original": token, "Lowercase": token.lower(),
         "After stop-word removal": token.lower() if token.lower() not in stop_words else "—",
         "Stem": stemmer.stem(token.lower()),
         "Lemma": lemmatizer.lemmatize(token.lower())}
        for token in tokens[:100]
    ]


def sentence_tokens(text):
    ensure_nltk_data()
    return sent_tokenize(text)
