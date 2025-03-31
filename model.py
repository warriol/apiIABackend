import kagglehub
#GEMMA_PATH = kagglehub.model_download("google/gemma-3/transformers/gemma-3-4b-pt")
GEMMA_PATH = kagglehub.model_download("google/gemma-3/transformers/gemma-3-4b-pt", path="./models")

print(f"El modelo se descargó en: {GEMMA_PATH}")