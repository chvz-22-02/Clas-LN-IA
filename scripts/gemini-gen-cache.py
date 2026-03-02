import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import datetime
from vertexai.preview import caching
import pickle

# 1. Ruta a tu archivo JSON de credenciales
PATH_TO_JSON = "../poc-enei-b8ddad81f742.json"

# 2. Cargar las credenciales
credentials = service_account.Credentials.from_service_account_file(PATH_TO_JSON)

# 3. Inicializar Vertex AI con tu proyecto y región
vertexai.init(
    project="poc-enei", 
    location="us-central1", # O la región que prefieras
    credentials=credentials
)

# 4. Crear el contenido del Cache
# Aquí "congelamos" tus archivos de taxonomía
system_instruction = """
Contexto: Eres un experto en arquitectura de datos y procesamiento de lenguaje natural (NLP) especializado en taxonomías peruanas. Tu tarea es clasificar frases del archivo frases.txt en dos estructuras jerárquicas cerradas extraídas de sistema.txt y conceptos.txt.Entradas:
1. Jerarquía A (Sistema): Tema > Subtema > Categoría > Tabla. 

2. Jerarquía B (Conceptos): Concepto > Codelist > Code. 

3. Ejemplos: Guíate por frases_clasificadas.txt. Reglas de Clasificación:
* Proximidad Semántica: Clasifica basándote en el significado y contexto, no solo en palabras clave. 
- Múltiples Resultados: Si una frase aplica a más de una Tabla o Code, genera una entrada independiente para cada combinación.
- Umbral de Confianza: Asigna un valor de 0 a 1 para tu nivel de confianza en la asignación.
- Casos sin coincidencia: Si no existe una categoría lógica, devuelve el campo como "N/A".

Formato de Salida Obligatorio (JSON):Devuelve exclusivamente un array de objetos con esta estructura:
[
  {
    "frase_original": "texto",
    "jerarquia_sistema": {
      "tema": "...", "subtema": "...", "categoria": "...", "tabla": "...", "confianza": 0.0
    },
    "jerarquia_conceptos": {
      "concepto": "...", "codelist": "...", "code": "...", "confianza": 0.0
    }
  }
]
"""

# Leemos tus archivos TXT
with open("../data/inputs/sistema.txt", "r", encoding="utf-8") as f:
    sistema_data = f.read()
with open("../data/inputs/conceptos.txt", "r", encoding="utf-8") as f:
    conceptos_data = f.read()

# 5. Generar el Cache (esto tiene costo por hora de almacenamiento)
cached_content = caching.CachedContent.create(
    model_name="gemini-2.5-flash-lite",
    system_instruction=system_instruction,
    contents=[sistema_data, conceptos_data],
    ttl=datetime.timedelta(hours=1)
)

with open("../cached_name.pkl", 'wb') as file:
    pickle.dump(cached_content.name, file)
# # 6. Crear el modelo usando el contenido cacheado
# model = GenerativeModel.from_cached_content(cached_content=cached_content)

# # 7. Ejemplo de uso con un lote de frases
# # response = model.generate_content("Clasifica las siguientes frases: [frase 1, frase 2...]")
# response = model.generate_content("Agricultura en Piura y Junin")
# print(response.text)