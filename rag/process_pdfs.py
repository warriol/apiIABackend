# pip install pymupdf
# pip install faiss-cpu
# pip install sentence-transformers
# python rag/process_pdfs.py
import faiss
import numpy as np
import os
import fitz  # PyMuPDF
import json
from sentence_transformers import SentenceTransformer

# Ruta donde tienes los PDFs
pdf_folder = "../pdfs"

# Cargar el modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Lista para guardar los textos extraídos de los PDFs
all_texts = []

# Guardar los textos en un archivo JSON
def save_texts(texts):
    with open('texts.json', 'w') as f:
        json.dump(texts, f)

# Cargar los textos guardados
def load_texts():
    with open('texts.json', 'r') as f:
        return json.load(f)

def extract_text_from_pdfs(pdf_folder):
    texts = []

    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            doc = fitz.open(pdf_path)
            text = " ".join([page.get_text("text") for page in doc])
            texts.append(text)
            all_texts.append(text)  # Guardar en all_texts para futuras consultas

    save_texts(all_texts)
    return texts


# Extraer texto de los PDFs
texts = extract_text_from_pdfs(pdf_folder)

# Convertir los textos en embeddings
embeddings = np.array([model.encode(text) for text in texts])

# Crear un índice FAISS
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

# Guardar el índice en un archivo
faiss.write_index(index, "embeddings.index")

print("Embeddings generados y guardados en FAISS correctamente.")
