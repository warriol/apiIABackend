# ngrok http --url=deciding-rested-badger.ngrok-free.app 127.0.0.1:5000
from flask import Flask, request, jsonify, Response
from gpt4all import GPT4All
from flask_cors import CORS

app = Flask(__name__)
# CORS(app, resources={r"/api/*": {"origins": "warriols.site"}}) # Para deploy en producción
CORS(app) # Para desarrollo en local

# Crear el objeto GPT4All
gpt = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf", model_path="./models")

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"response": "Por favor, ingresa una pregunta."}), 400

    # Hacer la consulta al modelo GPT, sin contexto
    try:
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
                ):
                    yield chunk  # Enviar solo texto sin JSON

        return Response(generate_response(), content_type='text/plain')

    except Exception as e:
        print(f"Error en la generación: {str(e)}")
        return jsonify({"response": "Error en el procesamiento"}), 500

@app.route('/')
def hello_world():  # put application's code here
    return 'Servidor de Wilson Arriola on line!'


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
