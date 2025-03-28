# ngrok http --url=deciding-rested-badger.ngrok-free.app 127.0.0.1:5000
import kagglehub
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, Gemma3ForConditionalGeneration
#from transformers.models.gemma3 import Gemma3ForConditionalGeneration, Gemma3Processor

app = Flask(__name__)
CORS(app)  # Para desarrollo en local

# Cargar el modelo Gemma3
#gemma3_path = "D:/servidores/Python/apiIABackend/models/gemma3"  # Update this path if necessary
gemma3_path = kagglehub.model_download("google/gemma-3/transformers/gemma-3-4b-pt")
#processor = Gemma3Processor.from_pretrained(gemma3_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = Gemma3ForConditionalGeneration.from_pretrained(gemma3_path, device_map="auto").eval()
#model = Gemma3ForConditionalGeneration.from_pretrained(gemma3_path, torch_dtype=torch.float16).to(device)

tokenizers = AutoTokenizer.from_pretrained(gemma3_path)

print(f"Device: {device}")
print(model)
prompt_style = """Below is an instruction that describes a task, paired with an input that provides further context. 
Write a response that appropriately completes the request. 
Before answering, think carefully about the question and create a step-by-step chain of thoughts to ensure a logical and accurate response.

### Question:
{}

### Response:
<think>
{}
"""

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get("question", "").strip()

    if not question:
        return jsonify({"response": "Por favor, ingresa una pregunta."}), 400

    print(f"Llegó una pregunta: {question}")

    # Hacer la consulta al modelo Gemma3, sin contexto
    try:
        input_text = f"<bos>{question}<eos>"
        #input_ids = processor(text=question, return_tensors="pt").to(device)
        inputs = tokenizers([prompt_style.format(question, "") + tokenizers.oes_token], return_tensors="pt").to(device)
        print(f"Input IDs: {inputs}")

        outputs = model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1200,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True
        )
        print(f"Outputs: {outputs}")

        response_text = tokenizers.batch_decode(
            outputs,
            skip_special_tokens=False
        )[0]
        print(f"Response Text: {response_text}")

        return jsonify({"response": response_text}), 200

    except Exception as e:
        print(f"Error en la generación: {str(e)}")
        return jsonify({"response": "Error en el procesamiento"}), 500

@app.route('/')
def saludo():  # put application's code here
    return 'Servidor de Wilson Arriola on line!'

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)