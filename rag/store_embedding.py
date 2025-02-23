# pip install faiss-cpu
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Cargar el modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Lista de textos (Ejemplo: reemplázalo con el texto de tus PDFs)
texts = [
    "Este es un documento importante sobre derecho penal.",
    "Aquí se explica el Código de Faltas en Uruguay.",
    "Las sanciones están detalladas en el artículo 5."
]

# Convertir los textos en embeddings
embeddings = np.array([model.encode(text) for text in texts])

# Crear un índice FAISS para búsqueda rápida
dimension = embeddings.shape[1]  # Tamaño del vector
index = faiss.IndexFlatL2(dimension)  # Tipo de índice L2 (distancia euclidiana)
index.add(embeddings)  # Agregar los embeddings al índice

# Guardar el índice en un archivo
faiss.write_index(index, "embeddings.index")

print("Embeddings guardados en FAISS correctamente.")
