## Iniciar el servicio

- Por defecto el servicio ya esta corriendo en mi pc
- Si no lo has hecho aun, instala ollama

## 🚀 Instalación y Configuración

```sh
# Iniciar el servicio de backend
python serverOllama.py
```

## 📝 Uso de ollama

- Para ver una lista de modelos instalados
```sh
ollama list
```
- Para instalar un modelo nuevo
```sh
ollama pull gemma3:4b
```
- Para usar un modelo
```sh
# actaulzia el codigo de serverOllama.py
# ...
"model": "gemma3:4b"
```
