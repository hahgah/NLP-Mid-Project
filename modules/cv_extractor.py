from pypdf import PdfReader
from docx import Document


def extract_pdf(file):
    reader = PdfReader(file)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def extract_docx(file):
    document = Document(file)
    parts = [paragraph.text for paragraph in document.paragraphs]
    for table in document.tables:
        for row in table.rows:
            parts.append(" ".join(cell.text for cell in row.cells))
    return "\n".join(parts)


def extract_cv_text(file):
    name = getattr(file, "name", "").lower()
    if name.endswith(".pdf"):
        return extract_pdf(file)
    if name.endswith(".docx"):
        return extract_docx(file)
    raise ValueError("Only PDF and DOCX files are supported.")
