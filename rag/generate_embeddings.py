# pip install sentence-transformers
from sentence_transformers import SentenceTransformer

# Cargar el modelo de embeddings
model = SentenceTransformer("all-MiniLM-L6-v2")  # Modelo liviano y eficiente

def generate_embedding(text):
    return model.encode(text)

# Ejemplo de uso

text = "División Estudios Legislativos    Cámara de Senadores    República Oriental del Uruguay    CODIGO PENAL   (Actualizado julio 2020)    LIBRO I    TITULO I    PARTE GENERAL    CAPITULO I    Principios generales    Artículo 1. (Concepto del delito)  Es delito toda acción u omisión expresamente prevista por la ley penal.    Para que ésta se considere tal, debe contener una norma y una sanción.    Artículo 2. (División de los delitos) Los delitos, atendida su gravedad, se dividen en crímenes, delitos y    faltas. Los crímenes son los ilícitos de competencia de la Corte Penal Internacional de acuerdo a lo    dispuesto en el artículo 5 del Estatuto de Roma y además todos los que por su extrema gravedad se rijan    por leyes especiales, por este Código y las normas de derecho internacional en cuanto le sean aplicables.    Los delitos son todos los demás que no revistan la gravedad indicada en el párrafo anterior.    Las faltas se rigen por lo dispuesto en el libro III del presente Código.    Redacción dada por el artículo 1 de la Ley N° 18026 de 25/09/2006.    Artículo 3. (Relación de causalidad) Nadie puede ser castigado por un hecho previsto por la ley como    delito, si el daño o el peligro del cual depende la existencia del delito, no resulta ser la consecuencia de su    acción o de su omisión. No impedir un resultado que se tiene la obligación de evitar, equivale a producirlo.    Artículo 4. (De la concausa) No se responde de la concausa preexistente, superviniente o simultánea,    independiente del hecho, que no se ha podido prever. La que se ha podido prever y no se ha prevista,    será tenida en cuenta por el Juez para rebajar la pena, según su criterio, de acuerdo con las    circunstancias del caso, y lo dispuesto en el artículo 18."

embedding = generate_embedding(text)
print(embedding)  # Imprimirá un vector numérico
