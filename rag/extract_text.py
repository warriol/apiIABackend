# pip install pymupdf
import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text") + "\n"
    return text

# Ejemplo de uso
pdf_file = "pdfs/codigo_penal.pdf"  # Reemplázalo con tu archivo
text = extract_text_from_pdf(pdf_file)
print(text)
