## Instalar CUDA
https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=11&target_type=exe_local

## Instalar torch para GPU
pip uninstall torch
pip install torch --index-url https://download.pytorch.org/whl/cu121


pip install 'accelerate>=0.26.0'

## Descargar el modelo de Kaggle

- Descargar el modelo Gemma 3 de Kaggle:
- https://www.kaggle.com/models/google/gemma-3/transformers/gemma-3-4b-pt
- Haz clic en "Dowload" para descargar el modelo.
- Previameente debes estar logueado en Kaggle.

## crear Kaggle API Token

- Descargar el Kaggle API Token:  
-- Ve a tu cuenta de Kaggle.
-- En la sección "API", haz clic en "Create New API Token".
-- Esto descargará un archivo llamado kaggle.json.
- Colocar el kaggle.json en el Directorio Correcto:  
-- Para sistemas Unix (Linux, macOS), coloca el archivo en ~/.kaggle/.
-- Para Windows, coloca el archivo en C:\Users\<TuNombreDeUsuario>\.kaggle\.
- Asegurarse de que el Directorio .kaggle Existe

## El error 403 Client Error indica que no tienes permiso para acceder al modelo porque no has aceptado los términos de uso de Gemma 3 en Kaggle.
- Solución Paso a Paso

    1. Inicia sesión en Kaggle

        Ve a Kaggle e inicia sesión en tu cuenta.

    2. Aceptar los términos de uso del modelo

        Dirígete a la página del modelo Gemma 3 en Kaggle:
        🔗 Gemma 3 en Kaggle

        Acepta los términos de uso que se te presenten.

    3. Verifica tu autenticación en Kaggle

        Descarga tu archivo kaggle.json desde tu cuenta en Kaggle.

        Coloca este archivo en la ruta:

```C:\Users\<TU_USUARIO>\.kaggle\kaggle.json```

## Verifica que kaggle esté instalado

1. Ejecuta este comando en tu terminal para confirmar que el paquete está instalado:

pip show kaggle

2. Si no está instalado, instálalo con:

pip install kaggle

