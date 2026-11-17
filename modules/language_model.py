import re
from collections import Counter


def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9+#.]+\b", text.lower())


def build_ngram_report(text):
    words = tokenize(text)
    unigrams = Counter(words)
    bigrams = Counter(zip(words, words[1:]))
    trigrams = Counter(zip(words, words[1:], words[2:]))
    return {
        "unigram_count": len(words),
        "bigram_count": max(0, len(words) - 1),
        "trigram_count": max(0, len(words) - 2),
        "top_bigrams": [(" ".join(k), v) for k, v in bigrams.most_common(15)],
        "top_trigrams": [(" ".join(k), v) for k, v in trigrams.most_common(15)]
    }
