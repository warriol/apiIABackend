# ngrok http --url=deciding-rested-badger.ngrok-free.app 127.0.0.1:5000
import kagglehub
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration

app = Flask(__name__)
CORS(app)

# Cargar el modelo Gemma3
gemma3_path = kagglehub.model_download("google/gemma-3/transformers/gemma-3-4b-pt")
device = "cuda" if torch.cuda.is_available() else "cpu"

model = Gemma3ForConditionalGeneration.from_pretrained(
    gemma3_path,
    device_map="auto",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).eval()

tokenizer = AutoTokenizer.from_pretrained(gemma3_path)

print(f"Device: {device}")

prompt_style = """
### Question:
{}

### Response:
"""

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"response": "Por favor, ingresa una pregunta."}), 400

    print(f"Llegó una pregunta: {question}")

    # Generar texto usando el modelo Gemma3
    try:
        input_text = prompt_style.format(question)
        inputs = tokenizer(input_text, return_tensors="pt").to(device)

        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=512,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )

        response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Response Text: {response_text}")

        return jsonify({"response": response_text.strip()}), 200

    except Exception as e:
        print(f"Error en la generación: {str(e)}")
        return jsonify({"response": "Error en el procesamiento"}), 500

@app.route('/')
def saludo():
    return 'Servidor de Wilson Arriola on line!'

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)