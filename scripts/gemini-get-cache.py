import vertexai
from vertexai.generative_models import GenerativeModel, Part
from google.oauth2 import service_account
import datetime
from vertexai.preview import caching
import pickle

file_path = '../cached_name.pkl'
with open(file_path, 'rb') as file:
    loaded_data = pickle.load(file)

# Recuperamos el cache existente por su nombre
CACHED_CONTENT_ID = f"projects/poc-enei/locations/us-central1/cachedContents/{loaded_data}" 
cache = caching.CachedContent.get(CACHED_CONTENT_ID)

# Creamos el modelo usando el cache recuperado
model = GenerativeModel.from_cached_content(cached_content=cache)

# Ahora puedes iterar sobre tus 200k frases en lotes
# Este código ya no vuelve a leer ni a pagar por sistema.txt ni conceptos.txt
response = model.generate_content("""Precio de la palta en Piura
Tasa de feminicidios en Lambayeque""")
print(response.text)