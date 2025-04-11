import requests
import pandas as pd
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

OLLAMA_API_URL = "http://localhost:11434/api/generate"

train_data = pd.read_csv("train.csv", sep=",", encoding="utf-8")
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get("question", "").strip()
    role = data.get("role", "").strip() or "leyes"

    if not question:
        return jsonify({"response": "Por favor, ingresa una pregunta."}), 400

    print(f"Llegó una pregunta: {question}")

    if role == "leyes":
        prompt = (
            "Eres un abogado experto en derecho penal uruguayo. "
            "Responde siempre en español, utilizando un lenguaje técnico pero claro. "
            "Debes fundamentar tus respuestas con base en el Código Penal y Código de Faltas de Uruguay. "
            "Si la pregunta no está relacionada con derecho penal, indica que no puedes responder. "
            f"Pregunta del usuario: {question}"
        )
    elif role == "ortografia":
        prompt = f"""Eres un profesor de lengua española con excelente ortografía y gramática, tienes como tareas:
            - Revisarás el texto que se te envíe y harás las correcciones necesarias.
            - Verifica que al comienzo de cada párrafo haya una sangría de 5 espacios.
            - Entre un párrafo y el siguiente, debe haber un interlineado.
            - Si hay texto entre comillas indica que es un pasaje textual, se espera que no hagas correcciones orográficas o gramaticales de los pasajes textuales, pero si el mismo esta escrito en minúsculas debes cambiar el texto del pasaje textual a mayúsculas.
            - Presentarás como respuesta el texto corregido y al final una lista con las correcciones que hayas realizado.
            Texto del usuario: {question}"""
    elif role == "sgsp":
        context = "\n".join(
            f"{row['Contexto']}: {row['Response']}" for  _, row in train_data.iterrows()
        )
        prompt = (
            "Eres un experto en el Sistema de Gestión de Seguridad Pública (SGSP). "
            "Responde siempre en español, utilizando un lenguaje técnico pero claro. "
            "Utiliza el siguiente contexto para responder las preguntas del usuario:\n"
            f"{context}\n"
            f"Pregunta del usuario: {question}"
        )

    else:
        return jsonify({"response": "Rol no reconocido."}), 400

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
            return Response(respuesta, content_type='text/plain')

    except Exception as e:
        print(f"Error en la generación: {str(e)}")
        return jsonify({"response": "Error en el procesamiento"}), 500

@app.route('/')
def saludo():
    return 'Servidor de Wilson Arriola on line!'

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
