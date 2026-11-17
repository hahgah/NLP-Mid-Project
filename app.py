import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from modules.cv_extractor import extract_cv_text
from modules.preprocessing import preprocess_text, preprocessing_comparison
from modules.matching import compare_cv_to_job
from modules.language_model import build_ngram_report
from modules.embeddings import train_word2vec, get_similar_words, plot_embeddings
from modules.pos_tagger import tag_with_nltk, tag_with_spacy, pos_frequencies

st.set_page_config(page_title="AI CV Analyzer & Job Matcher", page_icon="📄", layout="wide")

st.title("📄 AI-Powered CV Analyzer & Job Matcher")
st.write("Upload a CV, paste a job description, and explore an NLP-based match analysis.")
st.caption("Academic NLP project: preprocessing, BoW, TF-IDF, N-grams, Word2Vec and POS tagging.")

with st.sidebar:
    st.header("How to use")
    st.write("1. Upload a text-based PDF or DOCX CV.")
    st.write("2. Paste the job description.")
    st.write("3. Click Analyze CV.")
    st.info("Scanned/image-only PDFs need OCR and are not supported in this version.")

uploaded_file = st.file_uploader("Upload CV (PDF or DOCX)", type=["pdf", "docx"])
job_description = st.text_area(
    "Paste Job Description",
    height=220,
    placeholder="Paste the full job description here..."
)

if "cv_text" not in st.session_state:
    st.session_state.cv_text = ""

if uploaded_file is not None:
    try:
        st.session_state.cv_text = extract_cv_text(uploaded_file)
        if st.session_state.cv_text.strip():
            st.success(f"CV text extracted: approximately {len(st.session_state.cv_text.split())} words.")
        else:
            st.warning("No readable text was found in this file.")
    except Exception as exc:
        st.error(f"Could not read this file: {exc}")

if st.session_state.cv_text:
    with st.expander("View extracted CV text"):
        st.text_area("Extracted text", st.session_state.cv_text, height=220, disabled=True)

analyze = st.button("🔍 Analyze CV", type="primary", disabled=not (st.session_state.cv_text.strip() and job_description.strip()))

if analyze:
    st.session_state.analysis = compare_cv_to_job(st.session_state.cv_text, job_description)

if "analysis" in st.session_state and st.session_state.cv_text and job_description.strip():
    result = st.session_state.analysis
    st.divider()
    st.header("1. CV and Job Match")
    c1, c2, c3 = st.columns(3)
    c1.metric("Text similarity", f"{result['similarity']:.1f}%")
    c2.metric("Matched skills", len(result["matched_skills"]))
    c3.metric("Missing listed skills", len(result["missing_skills"]))
    st.caption("The similarity percentage measures textual similarity using TF-IDF and cosine similarity. It is not a hiring probability or a guarantee of job suitability.")

    left, right = st.columns(2)
    with left:
        st.subheader("Skills found in CV")
        st.write(", ".join(result["cv_skills"]) if result["cv_skills"] else "No recognized skills found.")
        st.subheader("Matching skills")
        st.success(", ".join(result["matched_skills"]) if result["matched_skills"] else "No exact skill matches found.")
    with right:
        st.subheader("Skills mentioned in job description")
        st.write(", ".join(result["job_skills"]) if result["job_skills"] else "No recognized skills found.")
        st.subheader("Potential skills to improve")
        st.warning(", ".join(result["missing_skills"]) if result["missing_skills"] else "All detected job skills also appear in the CV.")

    st.subheader("TF-IDF comparison")
    st.write("The system converts the CV and job description into TF-IDF vectors and calculates cosine similarity.")
    st.dataframe(pd.DataFrame(result["top_terms"], columns=["Important term", "TF-IDF weight"]), use_container_width=True)

    st.divider()
    st.header("2.Text Preprocessing")
    st.write("Lowercasing, punctuation removal, tokenization, stop-word removal, stemming and lemmatization.")
    comparison = preprocessing_comparison(st.session_state.cv_text[:2500])
    st.dataframe(pd.DataFrame(comparison), use_container_width=True)
    with st.expander("Cleaned CV text"):
        st.write(preprocess_text(st.session_state.cv_text))

    st.divider()
    st.header("3.Bag of Words and TF-IDF")
    st.write("The comparison above uses TF-IDF. The matching engine also uses a Bag of Words vectorizer as a frequency-based representation.")
    st.write(f"Bag-of-Words vocabulary size: **{result['bow_vocabulary_size']}**")
    st.write(f"TF-IDF vocabulary size: **{result['tfidf_vocabulary_size']}**")

    st.divider()
    st.header("4.N-gram Language Model")
    ngram = build_ngram_report(st.session_state.cv_text)
    st.write(f"Unigrams: **{ngram['unigram_count']}** · Bigrams: **{ngram['bigram_count']}** · Trigrams: **{ngram['trigram_count']}**")
    st.write("Most frequent bigrams")
    st.dataframe(pd.DataFrame(ngram["top_bigrams"], columns=["Bigram", "Frequency"]), use_container_width=True)
    st.write("Most frequent trigrams")
    st.dataframe(pd.DataFrame(ngram["top_trigrams"], columns=["Trigram", "Frequency"]), use_container_width=True)
    st.caption("This is an exploratory n-gram frequency demonstration, not a trained predictive language model or a calibrated perplexity evaluation.")

    st.divider()
    st.header("5.Word2Vec Word Embeddings")
    corpus = [st.session_state.cv_text, job_description]
    model = train_word2vec(corpus)
    if model is not None:
        candidates = [w for w in ["python", "data", "developer", "machine", "learning", "database", "security"] if w in model.wv]
        selected = st.selectbox("Explore a word in the trained embedding model", candidates) if candidates else None
        if selected:
            similar = get_similar_words(model, selected)
            st.write(f"Words most similar to **{selected}**")
            st.dataframe(pd.DataFrame(similar, columns=["Word", "Similarity"]), use_container_width=True)
        fig = plot_embeddings(model)
        if fig is not None:
            st.pyplot(fig)
        st.caption("Word2Vec is trained on the uploaded CV and job description. With a small corpus, embeddings and similarity results are only illustrative.")
    else:
        st.info("Not enough text to train Word2Vec. Add more CV/job-description text.")

    st.divider()
    st.header("6.POS Tagging")
    pos_text = st.text_area("Text for POS tagging", value=st.session_state.cv_text[:1200], height=120, key="pos_input")
    if st.button("Run POS tagging"):
        try:
            tagged = tag_with_nltk(pos_text)
            st.subheader("NLTK POS tags")
            st.dataframe(pd.DataFrame(tagged, columns=["Token", "POS tag"]), use_container_width=True)
            frequencies = pos_frequencies(tagged)
            st.subheader("POS tag frequencies")
            st.bar_chart(pd.DataFrame({"Count": frequencies}))
        except LookupError as exc:
            st.error("NLTK data is missing. Run: python -c \"import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')\"")
            st.code(str(exc))
        try:
            spacy_rows = tag_with_spacy(pos_text)
            st.subheader("spaCy POS tags and named entities")
            st.dataframe(pd.DataFrame(spacy_rows), use_container_width=True)
        except Exception as exc:
            st.info(f"spaCy model is not available. Install it using the README instructions. Details: {exc}")

st.divider()
st.caption("NLP Mid Project | AI-Powered CV Analyzer & Job Matcher | Educational prototype")
