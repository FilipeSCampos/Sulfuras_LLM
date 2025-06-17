# utils/file_processing.py

import fitz
import pandas as pd
from docx import Document


def process_document(uploaded_file):
    if uploaded_file.type == "application/pdf":
        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = "".join(page.get_text() for page in doc)
        doc.close()
    elif (
        uploaded_file.type == "application/vnd.openxmlformats-"
        "officedocument.wordprocessingml.document"
    ):
        doc = Document(uploaded_file)
        text = "\n".join(p.text for p in doc.paragraphs)
    elif uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        # Opcional: exibir um preview no Streamlit
        text = df.to_string(index=False)
    else:
        text = ""
    return text
