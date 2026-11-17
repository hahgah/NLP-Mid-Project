from modules.matching import extract_skills, compare_cv_to_job


def test_extract_skills():
    skills = extract_skills("I use Python, SQL and machine learning.")
    assert "python" in skills
    assert "sql" in skills
    assert "machine learning" in skills


def test_compare():
    result = compare_cv_to_job(
        "Python SQL data analysis",
        "Python SQL machine learning"
    )
    assert 0 <= result["similarity"] <= 100
    assert "python" in result["matched_skills"]
    assert "machine learning" in result["missing_skills"]
