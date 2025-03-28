# ngrok http --url=deciding-rested-badger.ngrok-free.app 127.0.0.1:5000
import faiss
import kagglehub
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import torch
from sentence_transformers import SentenceTransformer
from transformers import Gemma3ForConditionalGeneration, Gemma3Processor

app = Flask(__name__)
# CORS(app, resources={r"/api/*": {"origins": "warriols.site"}}) # Para deploy en producción
CORS(app)  # Para desarrollo en local

# Cargar el modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")

# Cargar el modelo Gemma3
gemma3_path = kagglehub.model_download("google/gemma-3/transformers/gemma-3-4b-pt")
processor = Gemma3Processor.from_pretrained(gemma3_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Gemma3ForConditionalGeneration.from_pretrained(gemma3_path, torch_dtype=torch.float16).to(device)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"response": "Por favor, ingresa una pregunta."}), 400

    print(f"Llegó una pregunta: {question}")

    # Hacer la consulta al modelo Gemma3, sin contexto
    try:
        input_ids = processor(text=question, return_tensors="pt").to(device)
        outputs = model.generate(**input_ids, max_new_tokens=512)
        response_text = processor.batch_decode(
            outputs,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False
        )[0]

        return jsonify({"response": response_text}), 200

    except Exception as e:
        print(f"Error en la generación: {str(e)}")
        return jsonify({"response": "Error en el procesamiento"}), 500

@app.route('/')
def hello_world():  # put application's code here
    return 'Servidor de Wilson Arriola on line!'

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)