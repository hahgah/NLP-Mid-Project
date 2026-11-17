# AI-Powered CV Analyzer & Job Matcher

A Streamlit-based academic NLP project. It extracts text from PDF/DOCX CVs, compares a CV with a job description, identifies listed skills, and demonstrates the NLP topics covered in the lab sessions.

## Features

- Upload PDF and DOCX CV files and extract their text.
- Text preprocessing: lowercasing, punctuation removal, tokenization, stop-word removal, stemming and lemmatization.
- Bag of Words and TF-IDF vectorization.
- Cosine similarity between CV and job description.
- Rule-based skill identification and matched/missing skill lists.
- Unigram, bigram and trigram frequency analysis.
- Word2Vec Skip-Gram training, similar-word exploration and PCA visualization.
- NLTK POS tagging, POS frequency chart, and spaCy POS/NER output.

## Requirements

Recommended: **Python 3.11**. Use a fresh virtual environment.

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

If PowerShell blocks activation, use Command Prompt:

```cmd
.venv\Scripts\activate
```

### Run

```bash
streamlit run app.py
```

The first preprocessing/POS operation may download small NLTK datasets. An internet connection may be needed the first time.

## Project structure

```text
CV_Analyzer_Project/
├── app.py
├── requirements.txt
├── README.md
├── modules/
│   ├── __init__.py
│   ├── cv_extractor.py
│   ├── preprocessing.py
│   ├── matching.py
│   ├── language_model.py
│   ├── embeddings.py
│   └── pos_tagger.py
├── data/
│   ├── job_descriptions/
│   │   └── sample_data_analyst_job.txt
│   └── sample_cvs/
│       └── sample_cv.txt
└── tests/
    └── test_matching.py
```

## How to test

1. Run the application.
2. Upload `data/sample_cvs/sample_cv.txt` only as a reference for creating your own DOCX/PDF test CV; the uploader intentionally accepts PDF/DOCX only.
3. Copy the sample job description into the job-description box.
4. Upload a text-based PDF or DOCX CV and click **Analyze CV**.
5. Explore the Lab 2–6 outputs.

## Important limitations

- Scanned PDFs and image-only documents are not OCR processed.
- Skill detection uses a small editable keyword list in `modules/matching.py`; it is not a full semantic skill extractor.
- The match percentage is TF-IDF cosine similarity between two texts. It is **not** a hiring probability, objective employability score, or guarantee of suitability.
- Word2Vec is trained on the submitted CV and job description only. Small corpora produce weak/illustrative embeddings; use a larger text corpus for meaningful semantic relationships.
- N-gram output is frequency analysis, not a fully trained next-word prediction/perplexity system.
- The application is an educational prototype, not a validated recruitment tool. Do not use it as the sole basis for hiring decisions.

## Lab mapping

| Lab topic | Project implementation |
|---|---|
| Lab 1: NLP libraries and introduction | Python NLP modules and Streamlit application |
| Lab 2: Preprocessing | `modules/preprocessing.py` |
| Lab 3: Bag of Words and TF-IDF | `modules/matching.py` |
| Lab 4: N-gram language modeling | `modules/language_model.py` |
| Lab 5: Word embeddings / Word2Vec | `modules/embeddings.py` |
| Lab 6: POS tagging | `modules/pos_tagger.py` |
| Lab 7: Open-ended project | Full CV Analyzer application |

## Team division suggestion

- Member 1: CV extraction and preprocessing.
- Member 2: BoW, TF-IDF, cosine similarity, N-grams and Word2Vec.
- Member 3: Streamlit UI, POS tagging, integration and testing.

All team members should understand the overall workflow and the limitations of the similarity score.
