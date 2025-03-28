# ngrok http --url=deciding-rested-badger.ngrok-free.app 127.0.0.1:5000
from flask import Flask, request, jsonify, Response
from gpt4all import GPT4All
from flask_cors import CORS
import json
# import numpy as np
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
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"response": "Por favor, ingresa una pregunta."}), 400

    """
    # Respuesta: ERROR: The prompt size exceeds the context window size and cannot be processed.
    # Generar el embedding de la consulta
    query_embedding = np.array([model.encode(question)])

    # Realizar la búsqueda en FAISS para obtener los textos más relevantes
    distances, indices = index.search(query_embedding, 2)

    # Obtener los textos más relevantes
    relevant_texts = [texts[i] for i in indices[0]]

    # Combinar los textos relevantes en un solo string para pasarlo al modelo GPT
    combined_text = " ".join(relevant_texts)

    # Hacer la consulta al modelo GPT, usando los textos relevantes como contexto
    with gpt.chat_session():
        response = gpt.generate(f"Pregunta: {question}\nContexto: {combined_text}\nRespuesta:")
    """

    # Hacer la consulta al modelo GPT, sin contexto
    try:
        response_text = ""

        def generate_response():
            with gpt.chat_session() as chat:
                for chunk in chat.generate(
                    f"Pregunta: {question}\nRespuesta:",
                    max_tokens=512,
                    temp=0.7,
                    top_p=0.9,
                    repeat_penalty=1.1,
                    n_predict=512,
                    streaming=True  # Activa el modo streaming
                ):  # response_text += chunk  # Construye la respuesta en tiempo real

                    yield chunk  # Enviar solo texto sin JSON

        return Response(generate_response(), content_type='text/plain')

        # return jsonify({"response": response_text}), 200

    except Exception as e:
        print(f"Error en la generación: {str(e)}")
        return jsonify({"response": "Error en el procesamiento"}), 500

@app.route('/')
def hello_world():  # put application's code here
    return 'Servidor de Wilson Arriola on line!'


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
