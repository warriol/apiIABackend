# ngrok http 5000
# ngrok http --url=deciding-rested-badger.ngrok-free.app 127.0.0.1:5000
from flask import Flask, request, jsonify
from gpt4all import GPT4All
from flask_cors import CORS
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
# CORS(app, resources={r"/api/*": {"origins": "warriols.site"}}) # Para deploy en producción
CORS(app) # Para desarrollo en local

# Cargar el modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Cargar los textos guardados
def load_texts():
    with open('rag/texts.json', 'r') as f:
        return json.load(f)

# Cargar el índice FAISS ya guardado
index = faiss.read_index("embeddings.index")
texts = load_texts()  # Cargar los textos al iniciar el servidor

# Crear el objeto GPT4All
gpt = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf", model_path="./models")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get("question", "")

    # Generar el embedding de la consulta
    query_embedding = np.array([model.encode(question)])

    # Realizar la búsqueda en FAISS para obtener los textos más relevantes
    distances, indices = index.search(query_embedding, 2)

    # Obtener los textos más relevantes
    relevant_texts = [texts[i] for i in indices[0]]

    # Limitar el número de textos o fragmentarlos en partes más pequeñas
    relevant_texts = relevant_texts[0]  # Limitar a 2 textos, por ejemplo.

    # Combinar los textos relevantes en un solo string para pasarlo al modelo GPT
    combined_text = " ".join(relevant_texts)

    # Hacer la consulta al modelo GPT, usando los textos relevantes como contexto
    with gpt.chat_session():
        response = gpt.generate(f"Pregunta: {question}\nContexto: {combined_text}\nRespuesta:")

    return jsonify({"response": response})

@app.route('/')
def hello_world():  # put application's code here
    return 'Servidor de Wilson Arriola on line!'


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
