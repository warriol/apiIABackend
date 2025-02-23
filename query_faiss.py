import faiss
import numpy as np
import json
from sentence_transformers import SentenceTransformer

# Cargar el modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Cargar los textos guardados
def load_texts():
    with open('rag/texts.json', 'r') as f:
        return json.load(f)

# Cargar el índice FAISS
index = faiss.read_index("embeddings.index")

# Lista de textos originales (debes asegurarte de que coincidan con los embeddings almacenados)
texts = load_texts()

def search_faiss(query, top_k=2):
    """Busca en FAISS la información más relevante para una consulta."""
    query_embedding = np.array([model.encode(query)])
    distances, indices = index.search(query_embedding, top_k)

    results = [texts[i] for i in indices[0]]  # Obtener los textos más relevantes
    return results

# Consulta de prueba
query = "¿Qué dice el documento sobre la ley penal?"
results = search_faiss(query)

print("\n🔍 Resultados encontrados:")
for i, res in enumerate(results):
    print(f"{i+1}. {res[:500]}...")  # Mostrar solo los primeros 500 caracteres
