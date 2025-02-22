from flask import Flask, request, jsonify
from gpt4all import GPT4All
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Nombre del modelo
model_name = "mistral-7b-instruct-v0.1.Q4_0.gguf"

gpt = GPT4All(model_name, model_path="./models")

#CORS(app, resources={r"/chat/*": {"origins": "*"}})
@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get("question", "")

    with gpt.chat_session():
        response = gpt.generate(question)

    return jsonify({"response": response})

@app.route('/')
def hello_world():  # put application's code here
    return 'Servidor de Wilson Arriola on line!'


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
