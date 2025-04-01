import requests
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

OLLAMA_API_URL = "http://localhost:11434/api/generate"
conversation_history_leyes = []
conversation_history_ortografia = []

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get("question", "").strip()
    role = data.get("role", "").strip() or "leyes"

    if not question:
        return jsonify({"response": "Por favor, ingresa una pregunta."}), 400

    print(f"Llegó una pregunta: {question}")

    if role == "leyes":
        conversation_history = conversation_history_leyes
        prompt = (
            "Eres un abogado experto en derecho penal uruguayo. "
            "Responde siempre en español, utilizando un lenguaje técnico pero claro. "
            "Debes fundamentar tus respuestas con base en el Código Penal y Código de Faltas de Uruguay. "
            "Si la pregunta no está relacionada con derecho penal, indica que no puedes responder. "
            f"Historial de la conversación:\n{'\n'.join(conversation_history)}\n"
            f"Pregunta del usuario: {question}"
        )
    else:
        conversation_history = conversation_history_ortografia
        prompt = (
            "Eres un profesor de lengua española con excelente ortografía y gramática. "
            "Revisarás el texto que se te envíe y harás las correcciones necesarias. "
            "Al comienzo de cada párrafo, verifica que hayan 5 espacios. "
            "Entre un párrafo y el siguiente, debe haber un interlineado. "
            "Si se encuentran pasajes textuales, se espera que se indique con comillas el comienzo y el final, y la cita textual irá escrita en mayúsculas. "
            "Presentarás como respuesta el texto corregido y al final una lista de faltas ortográficas encontradas y su corrección; además de otras correcciones que hayas hecho. "
            f"Historial de la conversación:\n{'\n'.join(conversation_history)}\n"
            f"Texto del usuario: {question}"
        )

    # Añadir la nueva pregunta al historial de la conversación
    conversation_history.append(f"Usuario: {question}")

    # Enviar pregunta a la API de Ollama
    try:
        payload = {
            "model": "gemma3:4b",
            "prompt": prompt,
            "stream": True,
            "max_tokens": 512
        }
        with requests.post(OLLAMA_API_URL, json=payload, stream=True) as response:
            response.raise_for_status()
            respuesta = ""
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    respuesta += decoded_line
                    print(f"Streamed Response Line: {decoded_line}")

            # Añadir la respuesta al historial de la conversación
            conversation_history.append(f"IA: {respuesta.strip()}")

            return Response(respuesta, content_type='text/plain')

    except Exception as e:
        print(f"Error en la generación: {str(e)}")
        return jsonify({"response": "Error en el procesamiento"}), 500

@app.route('/')
def saludo():
    return 'Servidor de Wilson Arriola on line!'

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)