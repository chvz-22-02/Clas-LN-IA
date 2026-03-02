"""
Pipeline de clasificación semántica con la Batch API de Anthropic
=================================================================
Estrategia híbrida en 3 etapas:
  Etapa 1 — Haiku clasifica todo el volumen (barato y rápido)
  Etapa 2 — Sonnet reclasifica los registros con confianza < umbral
  Etapa 3 — Reporte de los casos que siguen siendo ambiguos

Uso:
  1. Instala dependencias:      pip install anthropic pandas
  2. Configura tu API key:      export ANTHROPIC_API_KEY="sk-ant-..."
  3. Ajusta las constantes de configuración en la sección CONFIG
  4. Ejecuta:                   python clasificador_pipeline.py
"""

import os
import time
import json
import pandas as pd
import anthropic
from pathlib import Path


# ─────────────────────────────────────────────
# CONFIG — ajusta estos valores a tu proyecto
# ─────────────────────────────────────────────
ANTHROPIC_API_KEY   = "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"#os.environ.get("ANTHROPIC_API_KEY", "TU_API_KEY_AQUI")

FRASES_CSV          = "../data/inputs/frases.txt"          # Archivo con las frases a clasificar (una por línea o CSV)
SISTEMA_CSV         = "../data/inputs/sistema.txt"         # Archivo con la jerarquía Tema→Subtema→Categoría→Tabla
CONCEPTOS_CSV       = "../data/inputs/conceptos.txt"       # Archivo con la jerarquía Concepto→Codelist→Code
EJEMPLOS_CSV        = "../data/inputs/frases_clasificadas.txt"  # Ejemplos de clasificación correcta

OUTPUT_ETAPA1       = "../data/outputs/resultados_haiku.csv"
OUTPUT_ETAPA2       = "../data/outputs/resultados_sonnet.csv"
OUTPUT_FINAL        = "../data/outputs/resultados_finales.csv"
ESTADO_BATCHES      = "../data/outputs/estado_batches.json"  # Persiste IDs de batches para recuperación ante fallos

MODELO_ETAPA1       = "claude-haiku-4-5-20251001"
MODELO_ETAPA2       = "claude-sonnet-4-6"

TAMANO_LOTE         = 50       # Frases por llamada a la API
UMBRAL_CONFIANZA    = 70       # % mínimo para dar por buena la clasificación de Etapa 1
MAX_TOKENS_SALIDA   = 800      # Tokens máximos de respuesta por frase clasificada
POLL_INTERVALO_SEG  = 60       # Segundos entre consultas de estado del batch


# ─────────────────────────────────────────────
# UTILIDADES
# ─────────────────────────────────────────────
def cargar_archivo(path: str) -> str:
    """Lee un archivo y devuelve su contenido como texto."""
    return Path(path).read_text(encoding="utf-8")


def cargar_frases(path: str) -> list[str]:
    """Carga las frases desde un archivo de texto (una por línea)."""
    texto = cargar_archivo(path)
    frases = [
        line.strip().strip('"').strip("'")
        for line in texto.splitlines()
        if line.strip()
    ]
    return frases


def guardar_estado(estado: dict):
    """Persiste los IDs de batch para poder recuperarse ante fallos."""
    with open(ESTADO_BATCHES, "w", encoding="utf-8") as f:
        json.dump(estado, f, indent=2, ensure_ascii=False)


def cargar_estado() -> dict:
    """Recupera los IDs de batch de una ejecución anterior."""
    if Path(ESTADO_BATCHES).exists():
        with open(ESTADO_BATCHES, encoding="utf-8") as f:
            return json.load(f)
    return {}


def extraer_confianza(texto_respuesta: str) -> int:
    """Extrae el porcentaje de confianza de la respuesta del modelo."""
    import re
    match = re.search(r'(\d{1,3})\s*%', texto_respuesta)
    if match:
        return int(match.group(1))
    return 0


# ─────────────────────────────────────────────
# CONSTRUCCIÓN DEL PROMPT
# ─────────────────────────────────────────────
def construir_sistema_prompt(sistema: str, conceptos: str, ejemplos: str) -> str:
    return f"""Eres un clasificador semántico experto en estadísticas del Perú.
Tu tarea es clasificar cada frase en dos jerarquías de categorías independientes,
asignando la categoría más próxima semánticamente.

Las frases pueden contener errores ortográficos, abreviaciones o redacción informal.
Interpreta siempre la intención semántica más probable antes de clasificar.

## JERARQUÍA 1: SISTEMA (Tema → Subtema → Categoría → Tabla)
Los campos "desc_" son orientativos para entender el alcance de cada nivel.
{sistema}

## JERARQUÍA 2: CONCEPTOS (Concepto → Codelist → Code)
{conceptos}

## REGLAS DE CLASIFICACIÓN
1. Clasifica por proximidad semántica, no por coincidencia literal de palabras.
2. Si una frase puede pertenecer a múltiples clasificaciones válidas en la misma
   jerarquía, inclúyelas todas como filas CSV separadas con el mismo valor de "frase".
3. Si no existe ninguna categoría adecuada en algún nivel, deja ese campo vacío.
4. Asigna un porcentaje de confianza (0-100%) a cada fila de clasificación.
5. Razona en 1 línea antes del CSV de cada frase (prefija con "# ").

## EJEMPLOS DE CLASIFICACIÓN CORRECTA
{ejemplos}

## FORMATO DE SALIDA OBLIGATORIO
Devuelve exclusivamente filas CSV con separador ";" y estas columnas exactas
(sin encabezado, sin texto adicional fuera de los comentarios "#"):

frase;nombre_tema;nombre_subtema;nombre_categoria;nombre_tabla;nombre_concepto;nombre_codelist;nombre_code;confianza_pct
"""


def construir_user_prompt(frases_lote: list[str]) -> str:
    frases_formateadas = "\n".join(f"{i+1}. {f}" for i, f in enumerate(frases_lote))
    return f"""Clasifica las siguientes frases según las instrucciones del sistema:

{frases_formateadas}

Responde con el razonamiento breve (prefijado con "# ") y las filas CSV para cada frase."""


# ─────────────────────────────────────────────
# PROCESAMIENTO BATCH
# ─────────────────────────────────────────────
def crear_requests_batch(
    frases: list[str],
    sistema_prompt: str,
    modelo: str,
    tamano_lote: int
) -> list[dict]:
    """Genera la lista de requests en el formato que espera la Batch API."""
    requests = []
    for i in range(0, len(frases), tamano_lote):
        lote = frases[i:i + tamano_lote]
        custom_id = f"lote_{i // tamano_lote:05d}"
        requests.append({
            "custom_id": custom_id,
            "params": {
                "model": modelo,
                "max_tokens": MAX_TOKENS_SALIDA * len(lote),
                "system": sistema_prompt,
                "messages": [
                    {"role": "user", "content": construir_user_prompt(lote)}
                ]
            }
        })
    return requests


def enviar_batch(client: anthropic.Anthropic, requests: list[dict]) -> str:
    """Envía un batch a la API y devuelve su ID."""
    # La Batch API acepta hasta 10,000 requests por llamada
    # Para volúmenes muy grandes, dividimos en múltiples batches
    LIMITE_BATCH = 10_000
    if len(requests) > LIMITE_BATCH:
        raise ValueError(
            f"El batch tiene {len(requests)} requests, máximo permitido: {LIMITE_BATCH}. "
            "Divide el procesamiento en múltiples llamadas a esta función."
        )

    batch = client.messages.batches.create(requests=requests)
    print(f"  ✓ Batch enviado — ID: {batch.id} ({len(requests)} lotes de frases)")
    return batch.id


def esperar_batch(client: anthropic.Anthropic, batch_id: str) -> object:
    """Espera activamente a que un batch termine de procesarse."""
    print(f"  ⏳ Esperando resultados del batch {batch_id}...")
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        estado = batch.processing_status
        conteos = batch.request_counts

        print(
            f"     Estado: {estado} | "
            f"Completados: {conteos.succeeded} | "
            f"Procesando: {conteos.processing} | "
            f"Errores: {conteos.errored}"
        )

        if estado == "ended":
            print(f"  ✓ Batch {batch_id} completado.")
            return batch

        time.sleep(POLL_INTERVALO_SEG)


def recuperar_resultados(client: anthropic.Anthropic, batch_id: str) -> dict[str, str]:
    """Descarga y parsea los resultados de un batch. Devuelve {custom_id: texto_respuesta}."""
    resultados = {}
    for result in client.messages.batches.results(batch_id):
        if result.result.type == "succeeded":
            contenido = result.result.message.content
            texto = " ".join(
                bloque.text for bloque in contenido if hasattr(bloque, "text")
            )
            resultados[result.custom_id] = texto
        elif result.result.type == "errored":
            print(f"  ⚠ Error en {result.custom_id}: {result.result.error}")
            resultados[result.custom_id] = ""
    return resultados


# ─────────────────────────────────────────────
# PARSEO DE RESULTADOS
# ─────────────────────────────────────────────
def parsear_csv_respuesta(texto: str) -> list[dict]:
    """Extrae las filas CSV de clasificación de la respuesta del modelo."""
    columnas = [
        "frase", "nombre_tema", "nombre_subtema", "nombre_categoria",
        "nombre_tabla", "nombre_concepto", "nombre_codelist", "nombre_code",
        "confianza_pct"
    ]
    filas = []
    for linea in texto.splitlines():
        linea = linea.strip()
        if not linea or linea.startswith("#"):
            continue
        partes = linea.split(";")
        if len(partes) >= 2:
            # Rellenamos con vacíos si faltan columnas
            while len(partes) < len(columnas):
                partes.append("")
            fila = dict(zip(columnas, partes[:len(columnas)]))
            # Convertir confianza a entero
            try:
                fila["confianza_pct"] = int(fila["confianza_pct"].replace("%", "").strip())
            except (ValueError, AttributeError):
                fila["confianza_pct"] = 0
            filas.append(fila)
    return filas


def reconstruir_dataframe(
    resultados: dict[str, str],
    frases: list[str],
    tamano_lote: int
) -> pd.DataFrame:
    """Convierte los resultados del batch en un DataFrame estructurado."""
    todas_filas = []
    for i in range(0, len(frases), tamano_lote):
        custom_id = f"lote_{i // tamano_lote:05d}"
        texto = resultados.get(custom_id, "")
        filas = parsear_csv_respuesta(texto)
        todas_filas.extend(filas)
    return pd.DataFrame(todas_filas)


# ─────────────────────────────────────────────
# PIPELINE PRINCIPAL
# ─────────────────────────────────────────────
def ejecutar_pipeline():
    print("\n" + "="*60)
    print("  PIPELINE DE CLASIFICACIÓN SEMÁNTICA — BATCH API ANTHROPIC")
    print("="*60 + "\n")

    # Inicializar cliente
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    estado = cargar_estado()

    # Cargar archivos
    print("📂 Cargando archivos de entrada...")
    frases_todas  = cargar_frases(FRASES_CSV)
    sistema       = cargar_archivo(SISTEMA_CSV)
    conceptos     = cargar_archivo(CONCEPTOS_CSV)
    ejemplos      = cargar_archivo(EJEMPLOS_CSV)
    print(f"   {len(frases_todas)} frases cargadas desde {FRASES_CSV}")

    # Construir sistema prompt (compartido entre etapas)
    sistema_prompt = construir_sistema_prompt(sistema, conceptos, ejemplos)

    # ── ETAPA 1: Haiku clasifica todo ──────────────────────────────
    print(f"\n{'─'*50}")
    print(f"  ETAPA 1 — {MODELO_ETAPA1} (volumen completo)")
    print(f"{'─'*50}")

    if "batch_id_etapa1" not in estado:
        requests_e1 = crear_requests_batch(
            frases_todas, sistema_prompt, MODELO_ETAPA1, TAMANO_LOTE
        )
        print(f"  Enviando {len(requests_e1)} lotes ({len(frases_todas)} frases)...")
        batch_id_e1 = enviar_batch(client, requests_e1)
        estado["batch_id_etapa1"] = batch_id_e1
        guardar_estado(estado)
    else:
        batch_id_e1 = estado["batch_id_etapa1"]
        print(f"  Recuperando batch existente: {batch_id_e1}")

    esperar_batch(client, batch_id_e1)
    resultados_e1 = recuperar_resultados(client, batch_id_e1)
    df_e1 = reconstruir_dataframe(resultados_e1, frases_todas, TAMANO_LOTE)
    df_e1["modelo_usado"] = MODELO_ETAPA1

    # Guardar resultados etapa 1
    df_e1.to_csv(OUTPUT_ETAPA1, index=False, sep=";", encoding="utf-8-sig")
    print(f"  ✓ Resultados Etapa 1 guardados en {OUTPUT_ETAPA1}")
    print(f"  Distribución de confianza:")
    print(f"    Alta (≥{UMBRAL_CONFIANZA}%): {len(df_e1[df_e1.confianza_pct >= UMBRAL_CONFIANZA])} registros")
    print(f"    Baja (<{UMBRAL_CONFIANZA}%):  {len(df_e1[df_e1.confianza_pct < UMBRAL_CONFIANZA])} registros")

    # ── ETAPA 2: Sonnet reclasifica los de baja confianza ──────────
    frases_baja_confianza = df_e1[df_e1.confianza_pct < UMBRAL_CONFIANZA]["frase"].unique().tolist()

    if frases_baja_confianza:
        print(f"\n{'─'*50}")
        print(f"  ETAPA 2 — {MODELO_ETAPA2} ({len(frases_baja_confianza)} frases de baja confianza)")
        print(f"{'─'*50}")

        if "batch_id_etapa2" not in estado:
            requests_e2 = crear_requests_batch(
                frases_baja_confianza, sistema_prompt, MODELO_ETAPA2, TAMANO_LOTE
            )
            print(f"  Enviando {len(requests_e2)} lotes...")
            batch_id_e2 = enviar_batch(client, requests_e2)
            estado["batch_id_etapa2"] = batch_id_e2
            guardar_estado(estado)
        else:
            batch_id_e2 = estado["batch_id_etapa2"]
            print(f"  Recuperando batch existente: {batch_id_e2}")

        esperar_batch(client, batch_id_e2)
        resultados_e2 = recuperar_resultados(client, batch_id_e2)
        df_e2 = reconstruir_dataframe(resultados_e2, frases_baja_confianza, TAMANO_LOTE)
        df_e2["modelo_usado"] = MODELO_ETAPA2

        df_e2.to_csv(OUTPUT_ETAPA2, index=False, sep=";", encoding="utf-8-sig")
        print(f"  ✓ Resultados Etapa 2 guardados en {OUTPUT_ETAPA2}")
    else:
        df_e2 = pd.DataFrame()
        print("\n  ✓ Todos los registros superaron el umbral de confianza. Etapa 2 omitida.")

    # ── CONSOLIDACIÓN FINAL ────────────────────────────────────────
    print(f"\n{'─'*50}")
    print("  CONSOLIDACIÓN FINAL")
    print(f"{'─'*50}")

    # Para cada frase, usar el resultado de Etapa 2 si existe, si no el de Etapa 1
    if not df_e2.empty:
        frases_en_e2 = set(df_e2["frase"].unique())
        df_e1_filtrado = df_e1[~df_e1["frase"].isin(frases_en_e2)]
        df_final = pd.concat([df_e1_filtrado, df_e2], ignore_index=True)
    else:
        df_final = df_e1

    df_final = df_final.sort_values("frase").reset_index(drop=True)
    df_final.to_csv(OUTPUT_FINAL, index=False, sep=";", encoding="utf-8-sig")

    # ── REPORTE RESUMEN ────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  RESUMEN FINAL")
    print(f"{'='*60}")
    print(f"  Total frases procesadas:  {len(frases_todas)}")
    print(f"  Total filas en resultado: {len(df_final)} (incluye clasificaciones múltiples)")
    print(f"  Procesadas solo con Haiku:  {len(df_final[df_final.modelo_usado == MODELO_ETAPA1])}")
    if not df_e2.empty:
        print(f"  Reprocesadas con Sonnet:    {len(df_final[df_final.modelo_usado == MODELO_ETAPA2])}")
    print(f"  Confianza promedio final:   {df_final.confianza_pct.mean():.1f}%")
    print(f"\n  📄 Archivo de resultados: {OUTPUT_FINAL}")

    # Casos que siguen siendo ambiguos después de Etapa 2
    aun_ambiguos = df_final[df_final.confianza_pct < UMBRAL_CONFIANZA]["frase"].unique()
    if len(aun_ambiguos) > 0:
        print(f"\n  ⚠ {len(aun_ambiguos)} frases siguen con confianza baja tras ambas etapas.")
        print("    Considera revisarlas manualmente o con Extended Thinking:")
        for f in aun_ambiguos[:10]:  # muestra solo las primeras 10
            print(f"    → {f}")
        if len(aun_ambiguos) > 10:
            print(f"    ... y {len(aun_ambiguos) - 10} más.")

    # Limpiar estado de batches completados
    if Path(ESTADO_BATCHES).exists():
        Path(ESTADO_BATCHES).unlink()

    print("\n  ✅ Pipeline completado exitosamente.\n")


# ─────────────────────────────────────────────
# MODO DE USO ALTERNATIVO: llamada estándar (no batch)
# Útil para pruebas rápidas con pocas frases
# ─────────────────────────────────────────────
def clasificar_frases_directo(
    frases: list[str],
    modelo: str = MODELO_ETAPA2,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Alternativa sin batch para pruebas o volúmenes pequeños (<500 frases).
    Hace llamadas síncronas una a una, útil para depuración.
    """
    client    = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    sistema   = cargar_archivo(SISTEMA_CSV)
    conceptos = cargar_archivo(CONCEPTOS_CSV)
    ejemplos  = cargar_archivo(EJEMPLOS_CSV)
    sys_prompt = construir_sistema_prompt(sistema, conceptos, ejemplos)

    todas_filas = []
    for i in range(0, len(frases), TAMANO_LOTE):
        lote = frases[i:i + TAMANO_LOTE]
        if verbose:
            print(f"  Procesando frases {i+1}–{min(i+len(lote), len(frases))}...")

        respuesta = client.messages.create(
            model=modelo,
            max_tokens=MAX_TOKENS_SALIDA * len(lote),
            system=sys_prompt,
            messages=[{"role": "user", "content": construir_user_prompt(lote)}]
        )
        texto = respuesta.content[0].text
        filas = parsear_csv_respuesta(texto)
        todas_filas.extend(filas)

        # Pequeña pausa para respetar rate limits en modo estándar
        if i + TAMANO_LOTE < len(frases):
            time.sleep(2)

    return pd.DataFrame(todas_filas)


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    ejecutar_pipeline()
    # frases_todas  = cargar_frases(FRASES_CSV)
    # clasificar_frases_directo(frases_todas[:10], modelo=MODELO_ETAPA1, verbose=True)