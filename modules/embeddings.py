import re
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.decomposition import PCA


def train_word2vec(texts):
    sentences = []
    for text in texts:
        words = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        if words:
            sentences.append(words)
    if not sentences or sum(map(len, sentences)) < 5:
        return None
    return Word2Vec(
        sentences=sentences,
        vector_size=50,
        window=3,
        min_count=1,
        workers=1,
        sg=1,
        epochs=100,
        seed=42
    )


def get_similar_words(model, word, topn=8):
    if model is None or word not in model.wv:
        return []
    return [(w, round(float(score), 4)) for w, score in model.wv.most_similar(word, topn=min(topn, len(model.wv) - 1))]


def plot_embeddings(model):
    if model is None or len(model.wv) < 3:
        return None
    words = list(model.wv.index_to_key)[:30]
    vectors = [model.wv[word] for word in words]
    points = PCA(n_components=2).fit_transform(vectors)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(points[:, 0], points[:, 1])
    for i, word in enumerate(words):
        ax.annotate(word, (points[i, 0], points[i, 1]), fontsize=8)
    ax.set_title("Word2Vec Embeddings — PCA projection")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()
    return fig
