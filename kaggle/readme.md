## 1. Descargar el modelo desde Hugging Face

Si no lo has hecho aún, puedes descargar el modelo utilizando huggingface-cli:

- Instala huggingface-cli si no lo tienes:

```shell
  pip install huggingface_hub
```
Inicia sesion en Hugging Face:
```shell
    huggingface-cli login
```
Usa el siguiente comando para descargar el modelo:
```shell
    huggingface-cli download google/gemma-3-4b-pt --local-dir ./gemma3
```
Esto descargará el modelo en la carpeta ./gemma3.


## 2. Convertir el modelo a .gguf (si es necesario para Ollama)

Si necesitas convertirlo a un formato compatible con Ollama ( .gguf), puedes usar la herramienta transformers-to-gguf:

- Instala la herramienta de conversión:
```shell
    pip install transformers llama-cpp-python
```
Convierte el modelo descargado:
```shell
    transformers-to-gguf ./gemma3/gemma3 --outtype f16
```
Este comando generará un archivo .gguf del modelo Gemma3. Asegúrate de tener suficiente espacio de almacenamiento y recursos disponibles, ya que los modelos de este tamaño pueden ser bastante pesados.

## 3. Cargar el modelo en Ollama

Si has convertido el modelo a .gguf con éxito, puedes cargarlo en Ollama de la siguiente manera:
```shell
    ollama create gemma3 -f ./gemma3.gguf
```
Este comando cargará el modelo a Ollama para que puedas usarlo.

## 4. Verificar el modelo en Ollama

Una vez que hayas creado el modelo en Ollama, asegúrate de que puedes ver el modelo correctamente con:
```shell
ollama list
```
Este comando debería listar todos los modelos disponibles, incluido el gemma3 que acabas de cargar.