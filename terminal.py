import kagglehub
import torch
from transformers.models.gemma3 import Gemma3ForConditionalGeneration, Gemma3Processor

GEMMA_PATH = kagglehub.model_download("google/gemma-3/transformers/gemma-3-4b-pt")
processor = Gemma3Processor.from_pretrained(GEMMA_PATH)

# Determine if CUDA (GPU) is available
device = "cuda" #if torch.cuda.is_available() else "cpu"

print(f"Device: {device}")

model = Gemma3ForConditionalGeneration.from_pretrained(GEMMA_PATH, torch_dtype=torch.float16).to(device)
print(model)

prompt = """It was a dark and stormy night. """
input_ids = processor(text=prompt, return_tensors="pt").to(device)
outputs = model.generate(**input_ids, max_new_tokens=512)
text = processor.batch_decode(
    outputs,
    skip_special_tokens=False,
    clean_up_tokenization_spaces=False
)
print(text[0])