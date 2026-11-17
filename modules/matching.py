import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SKILL_LIST = [
    "python", "java", "javascript", "c++", "sql", "mysql", "mongodb",
    "html", "css", "react", "node.js", "php", "django", "flask",
    "machine learning", "deep learning", "artificial intelligence",
    "natural language processing", "nlp", "data analysis", "pandas",
    "numpy", "scikit-learn", "tensorflow", "pytorch", "excel",
    "power bi", "tableau", "git", "github", "docker", "linux",
    "cybersecurity", "networking", "communication", "teamwork",
    "problem solving", "project management", "fastapi", "streamlit"
]


def extract_skills(text):
    text_lower = text.lower()
    found = []
    for skill in SKILL_LIST:
        pattern = r"(?<![a-z0-9])" + re.escape(skill.lower()) + r"(?![a-z0-9])"
        if re.search(pattern, text_lower):
            found.append(skill)
    return sorted(set(found))


def compare_cv_to_job(cv_text, job_text):
    documents = [cv_text, job_text]
    bow = CountVectorizer(stop_words="english")
    bow_matrix = bow.fit_transform(documents)

    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=3000)
    tfidf_matrix = tfidf.fit_transform(documents)
    similarity = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] * 100)

    cv_skills = extract_skills(cv_text)
    job_skills = extract_skills(job_text)
    matched = sorted(set(cv_skills).intersection(job_skills))
    missing = sorted(set(job_skills).difference(cv_skills))

    job_vector = tfidf_matrix[1].toarray().ravel()
    terms = tfidf.get_feature_names_out()
    top_indices = job_vector.argsort()[::-1][:15]
    top_terms = [(terms[i], round(float(job_vector[i]), 4)) for i in top_indices if job_vector[i] > 0]

    return {
        "similarity": similarity,
        "cv_skills": cv_skills,
        "job_skills": job_skills,
        "matched_skills": matched,
        "missing_skills": missing,
        "bow_vocabulary_size": len(bow.get_feature_names_out()),
        "tfidf_vocabulary_size": len(tfidf.get_feature_names_out()),
        "top_terms": top_terms
    }
