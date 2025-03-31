import requests
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

OLLAMA_API_URL = "http://localhost:11434/api/generate"

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"response": "Por favor, ingresa una pregunta."}), 400

    print(f"Llegó una pregunta: {question}")

    try:
        payload = {
            "model": "gemma3",  # Nombre del modelo tras importarlo a Ollama
            "prompt": question,
            "stream": False,
            "max_tokens": 512
        }

        response = requests.post(OLLAMA_API_URL, json=payload)
        response_data = response.json()

        if 'response' in response_data:
            respuesta = response_data['response'].strip()
            print(f"Response Text: {respuesta}")
            return jsonify({"response": respuesta}), 200
        else:
            print(f"Error en la respuesta de Ollama: {response_data}")
            return jsonify({"response": "Error en la generación del texto"}), 500

    except Exception as e:
        print(f"Error en la generación: {str(e)}")
        return jsonify({"response": "Error en el procesamiento"}), 500

@app.route('/')
def saludo():
    return 'Servidor de Wilson Arriola on line!'

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
