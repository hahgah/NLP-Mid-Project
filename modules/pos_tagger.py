import nltk
from nltk import pos_tag, word_tokenize


def nltk_pos_tag(text):
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
    nltk.download('punkt_tab', quiet=True)

    tokens = nltk.word_tokenize(text)
    tagged_words = nltk.pos_tag(tokens)

    return tagged_words


def pos_frequencies(tagged_tokens):
    counts = {}
    for _, tag in tagged_tokens:
        counts[tag] = counts.get(tag, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[1], reverse=True))


def tag_with_spacy(text):
    import spacy
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = {token.i: ent.label_ for ent in doc.ents for token in ent}
    return [
        {
            "Token": token.text,
            "POS": token.pos_,
            "Fine-grained tag": token.tag_,
            "Named entity": entities.get(token.i, "")
        }
        for token in doc
    ]
