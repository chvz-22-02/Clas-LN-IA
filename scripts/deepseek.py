"""
Script de Clasificación Masiva con DeepSeek API
Procesa 200,000 frases usando lotes concurrentes con optimización de caché
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
import csv
from datetime import datetime
import time
import os
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
import hashlib
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)
import aiofiles
from tqdm import tqdm

# ==================== CONFIGURACIÓN ====================

@dataclass
class Config:
    """Configuración centralizada del procesamiento"""
    # API Configuration
    API_KEY: str = "sk-aaaaaaaaaaaaaaaaaaaaaaaaaa"  # Reemplazar con tu API key
    API_URL: str = "https://api.deepseek.com/v1/chat/completions"
    MODEL: str = "deepseek-chat"  # o "deepseek-reasoner" para modo pensamiento
    
    # Procesamiento
    BATCH_SIZE: int = 500  # Frases por lote (ajustable: 100-1000)
    MAX_CONCURRENT_BATCHES: int = 5  # Lotes simultáneos
    MAX_RETRIES: int = 3  # Reintentos por lote fallido
    REQUEST_TIMEOUT: int = 120  # Segundos
    
    # Rate limiting
    REQUESTS_PER_SECOND: float = 10  # Máximo 10 requests/segundo
    PAUSE_BETWEEN_BATCHES: int = 2  # Segundos entre lotes secuenciales
    
    # Archivos
    INPUT_FILE: str = "../data/inputs/frases.txt"
    SYSTEM_FILE: str = "../data/inputs/sistema.txt"
    CONCEPTOS_FILE: str = "../data/inputs/conceptos.txt"
    EJEMPLOS_FILE: str = "../data/inputs/frases_clasificadas.txt"
    
    # Directorios
    OUTPUT_DIR: str = "../data/outputs"
    LOGS_DIR: str = "../logs/ds"
    CHECKPOINT_DIR: str = "../checkpoints-ds"
    ERROR_DIR: str = "../data/errors-ds"
    
    # Prompt
    PROMPT_TEMPLATE_FILE: str = "../prompts/deepseek.txt"

# ==================== LOGGING ====================

def setup_logging(config: Config):
    """Configura el sistema de logging"""
    Path(config.LOGS_DIR).mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(config.LOGS_DIR) / f"procesamiento_{timestamp}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)

# ==================== MANEJO DE CACHÉ ====================

class PromptCache:
    """Optimiza el caché para prompts repetitivos"""
    
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
    def get_prompt_hash(self, prompt_base: str, frases: List[str]) -> str:
        """Genera hash único para el prompt completo"""
        content = prompt_base + "".join(frases)
        return hashlib.md5(content.encode('utf-8')).hexdigest()
    
    def save_to_cache(self, prompt_hash: str, response: Dict):
        """Guarda respuesta en caché"""
        cache_file = self.cache_dir / f"{prompt_hash}.json"
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(response, f, ensure_ascii=False, indent=2)
        self.logger.debug(f"Respuesta cacheada: {cache_file}")
    
    def load_from_cache(self, prompt_hash: str) -> Optional[Dict]:
        """Carga respuesta de caché si existe"""
        cache_file = self.cache_dir / f"{prompt_hash}.json"
        if cache_file.exists():
            with open(cache_file, 'r', encoding='utf-8') as f:
                self.logger.info(f"Cache hit para {prompt_hash}")
                return json.load(f)
        return None

# ==================== CHECKPOINT MANAGER ====================

class CheckpointManager:
    """Maneja checkpoints para reanudar procesamiento"""
    
    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_file = self.checkpoint_dir / "progreso.json"
        self.logger = logging.getLogger(__name__)
        
    def load_checkpoint(self) -> Dict[str, Any]:
        """Carga el último checkpoint"""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {
            'lotes_procesados': [],
            'lotes_en_progreso': [],
            'ultimo_lote': 0,
            'frases_procesadas': 0,
            'timestamp': None
        }
    
    def save_checkpoint(self, checkpoint_data: Dict[str, Any]):
        """Guarda checkpoint"""
        checkpoint_data['timestamp'] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(checkpoint_data, f, indent=2)
        self.logger.info(f"Checkpoint guardado: {checkpoint_data['frases_procesadas']} frases")

# ==================== CONSTRUCTOR DE PROMPT ====================

class PromptBuilder:
    """Construye prompts optimizados para caché"""
    
    def __init__(self, config: Config, dt: bool = False):
        self.config = config
        self.dt = dt
        self.base_prompt = self._load_base_prompt()
        self.ejemplos = self._load_ejemplos()
        self.sistema_categorias = self._load_sistema_categorias()
        self.conceptos_categorias = self._load_conceptos_categorias()
        self.logger = logging.getLogger(__name__)
        
    def _load_base_prompt(self) -> str:
        """Carga el template base del prompt"""
        with open(self.config.PROMPT_TEMPLATE_FILE, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _load_ejemplos(self) -> str:
        """Carga ejemplos de clasificación"""
        df = pd.read_csv(self.config.EJEMPLOS_FILE, sep=';', encoding='utf-8')
        ejemplos = []
        for _, row in df.head(5).iterrows():  # Usar 5 ejemplos
            ejemplos.append(f"Frase: {row['Frase']}\nClasificación: {row.to_dict()}")
        return "\n\n".join(ejemplos)
    
    def _load_sistema_categorias(self) -> str:
        """Carga y formatea el sistema de categorías temáticas"""
        try:
            df = pd.read_csv(self.config.SYSTEM_FILE, sep=';', encoding='utf-8')
            
            # Formatear como texto legible para el prompt
            lineas = []
            for _, row in df.iterrows():
                tema = row.get('nombre_tema', '')
                subtema = row.get('nombre_subtema', '')
                categoria = row.get('nombre_categoria', '')
                tabla = row.get('nombre_tabla', '')
                
                # Crear una línea formateada
                if tema and subtema and categoria and tabla:
                    lineas.append(f"{tema} > {subtema} > {categoria} > {tabla}")
                elif tema and subtema and categoria:
                    lineas.append(f"{tema} > {subtema} > {categoria}")
                elif tema and subtema:
                    lineas.append(f"{tema} > {subtema}")
            
            # Limitar a las primeras 200 filas para no exceder tokens
            if len(lineas) > 200:
                self.logger.warning(f"Sistema tiene {len(lineas)} categorías, usando primeras 200")
                lineas = lineas[:200]
            
            return "\n".join(lineas)
        except Exception as e:
            self.logger.error(f"Error cargando sistema de categorías: {e}")
            return "Error cargando sistema de categorías"
    
    def _load_conceptos_categorias(self) -> str:
        """Carga y formatea el sistema de conceptos (dimensiones)"""
        try:
            df = pd.read_csv(self.config.CONCEPTOS_FILE, sep=';', encoding='utf-8')
            
            # Formatear como texto legible para el prompt
            lineas = []
            conceptos_agrupados = {}
            
            # Agrupar por concepto y codelist
            for _, row in df.iterrows():
                concepto = row.get('nombre_concepto', '')
                codelist = row.get('nombre_codelist', '')
                code = row.get('nombre_code', '')
                desc = row.get('desc_concepto', '')
                
                key = f"{concepto} - {codelist}"
                if key not in conceptos_agrupados:
                    conceptos_agrupados[key] = {
                        'descripcion': desc,
                        'codes': []
                    }
                conceptos_agrupados[key]['codes'].append(code)
            
            # Formatear la salida
            for key, data in conceptos_agrupados.items():
                concepto, codelist = key.split(' - ')
                lineas.append(f"\n{concepto} / {codelist}: {data['descripcion'][:50]}...")
                # Limitar a 10 códigos por codelist para no exceder tokens
                codes_str = ", ".join(data['codes'][:10])
                if len(data['codes']) > 10:
                    codes_str += f" ... y {len(data['codes']) - 10} más"
                lineas.append(f"  Códigos: {codes_str}")
            
            return "\n".join(lineas)
        except Exception as e:
            self.logger.error(f"Error cargando conceptos: {e}")
            return "Error cargando conceptos"
        
    def build_prompt(self, frases_lote: List[str], lote_id: int, dt: bool) -> str:
        fixed_part = f"""
# TAREA: CLASIFICACIÓN SEMÁNTICA JERÁRQUICA DE FRASES EN MÚLTIPLES DIMENSIONES

## OBJETIVO
Clasificar cada frase en DOS sistemas jerárquicos:

1. **SISTEMA TEMÁTICO**: Jerarquía de temas estadísticos (Tema → Subtema → Categoría → Tabla)
2. **SISTEMA DE CONCEPTOS**: Variables de desagregación o dimensiones (Concepto → Codelist → Code)

## DEFINICIÓN DE LOS SISTEMAS

### Sistema Temático
Define QUÉ se está midiendo (el indicador principal). Ej: PBI, tasa de analfabetismo, producción, población censada.

**Columnas del sistema temático:**
- `nombre_tema`: Área general (ECONOMÍA, SOCIALES, EDUCACIÓN, DEMOGRAFÍA, etc.)
- `nombre_subtema`: Subárea específica
- `nombre_categoria`: Categoría detallada
- `nombre_tabla`: Tabla estadística concreta

### Sistema de Conceptos
Define CÓMO se desagrega el indicador (las variables de quiebre). Incluye TODAS las dimensiones posibles:
- Geografía (ámbito geográfico: departamentos, ciudades, zonas registrales)
- Tipo de material (vivienda, paredes, pisos)
- Medida (unidades: porcentaje, número absoluto, índice, tasa)
- Tipo de vivienda (casa independiente, departamento, quinta)
- Grupo etario (rangos de edad: 0-4, 5-9, 10-14, etc.)
- Sexo (hombres, mujeres)
- Sector económico (agricultura, minería, manufactura, servicios)
- Nivel educativo (básica, primaria, secundaria, superior)
- Especie animal (ovino, vacuno, alpaca)
- Tipo de bien (televisor, cuero, lana)
- Y cualquier otra variable de desagregación

**Columnas del sistema de conceptos:**
- `nombre_concepto`: La dimensión (ÁMBITO GEOGRÁFICO, TIPO DE MATERIAL, MEDIDA, etc.)
- `nombre_codelist`: Tipo o categoría dentro de la dimensión
- `nombre_code`: Valor específico

## REGLA FUNDAMENTAL: DIFERENCIAR INDICADOR PRINCIPAL VS DIMENSIONES

Para CADA frase, debes separar:
- **El indicador principal** (lo que se está midiendo) → va en el sistema temático
- **Las dimensiones de desagregación** (cómo se segmenta) → van en el sistema de conceptos

**IMPORTANTE**: Una frase puede tener MÚLTIPLES dimensiones. Cada dimensión identificada debe generar UNA FILA SEPARADA en el resultado.

## LOS VALORES EXACTOS ESTÁN EN LAS SECCIONES SIGUIENTES

A continuación encontrarás:
1. **La lista completa de valores del sistema temático** (Temas, Subtemas, Categorías, Tablas)
2. **La lista completa de valores del sistema de conceptos** (Conceptos, Codelists, Códigos)

DEBES usar ÚNICAMENTE estos valores en tus respuestas. No inventes valores nuevos.

## LISTA COMPLETA DE VALORES - SISTEMA TEMÁTICO

{self.sistema_categorias}

## LISTA COMPLETA DE VALORES - SISTEMA DE CONCEPTOS

{self.conceptos_categorias}

## FORMATO DE SALIDA EXACTO

Genera un CSV con las siguientes columnas en este orden:

Frase;nombre_tema;nombre_subtema;nombre_categoria;nombre_tabla;confianza_tematica;nombre_concepto;nombre_codelist;nombre_code;confianza_conceptual

Donde:
- `Frase`: La frase original (se repite en múltiples filas para diferentes dimensiones)
- Las columnas del sistema temático: valores seleccionados de la lista proporcionada (o vacío si no hay coincidencia)
- `confianza_tematica`: Porcentaje (0-100%) de confianza en la asignación temática
- Las columnas del sistema de conceptos: valores seleccionados de la lista proporcionada para UNA DIMENSIÓN ESPECÍFICA
- `confianza_conceptual`: Porcentaje (0-100%) de confianza en la asignación de ESA dimensión

## EJEMPLOS DE CLASIFICACIÓN

{self.ejemplos}

## INSTRUCCIONES DETALLADAS DE CLASIFICACIÓN

### PASO 1: Analizar cada frase
Para CADA frase, identifica:
- **Indicador principal**: ¿Cuál es el fenómeno o variable medida? (busca sustantivos clave: tasa, producción, población, temperatura, etc.)
- **Todas las dimensiones**: Lugares, unidades, características demográficas, materiales, etc.

### PASO 2: Asignar sistema temático (SISTEMA TEMÁTICO)
- Busca en la lista de valores del sistema temático la tabla MÁS ESPECÍFICA que coincida semánticamente
- Asigna a TODOS los niveles (Tema, Subtema, Categoría, Tabla)
- Si no hay coincidencia, dejar campos en blanco

### PASO 3: Asignar dimensiones (SISTEMA DE CONCEPTOS)
Para CADA dimensión identificada:
- Busca en la lista de valores del sistema de conceptos el Concepto → Codelist → Code que corresponda
- Si el valor no existe exactamente, busca el más cercano semánticamente
- Asigna confianza basada en la precisión de la coincidencia

### PASO 4: Generar múltiples filas
- Número de filas = Número de dimensiones identificadas
- Todas las filas comparten la misma asignación temática
- Cada fila tiene UNA dimensión diferente en las columnas de conceptos

## CASOS ESPECIALES

1. **Dimensiones implícitas**: Si una frase menciona "porcentaje", la dimensión MEDIDA/PORCENTAJE está implícita y debe incluirse
2. **Múltiples menciones del mismo concepto**: Si menciona dos lugares (ej: "Lima y Callao"), evaluar si son relevantes ambos o elegir el principal (usar criterio: ¿el indicador se puede desagregar por ambos?)
3. **Sin dimensión identificable**: Si la frase no menciona ninguna dimensión, dejar conceptos en blanco (una sola fila)
4. **Dimensión "TOTAL"**: Usar el código "TOTAL" cuando se hable de la categoría general sin especificar
5. **Rangos aproximados**: Para rangos que no coinciden exactamente, usar el código más cercano y reducir confianza (ej: 26-33 años → 25-29 AÑOS con 80% de confianza)

## RESPUESTA ESPERADA

SOLO debes generar el CSV con las filas correspondientes. Sin explicaciones adicionales, sin texto antes o después.

El CSV debe comenzar directamente con la primera fila de datos (sin encabezado repetido, aunque el ejemplo lo muestra por claridad).
"""
        
        # Parte variable (no cacheable - solo las frases nuevas)
        variable_part = f"""
# LOTE {lote_id} - {len(frases_lote)} FRASES A CLASIFICAR

{chr(10).join([f"{i+1}. {frase}" for i, frase in enumerate(frases_lote)])}

# GENERA EL CSV CON EL SIGUIENTE FORMATO EXACTO:
Frase;nombre_tema;nombre_subtema;nombre_categoria;nombre_tabla;confianza_tematica;nombre_concepto;nombre_codelist;nombre_code;confianza_conceptual

RESPUESTA (SOLO EL CSV, SIN EXPLICACIONES ADICIONALES):
"""
    
        variable_part_dt = f"""
# LOTE {lote_id} - {len(frases_lote)} FRASES A CLASIFICAR

{chr(10).join([f"{i+1}. {frase}" for i, frase in enumerate(frases_lote)])}

# GENERA EL CSV CON EL SIGUIENTE FORMATO EXACTO:
Frase;nombre_tema;nombre_subtema;nombre_categoria;nombre_tabla;confianza_tematica;nombre_concepto;nombre_codelist;nombre_code;confianza_conceptual

Antes de generar la clasificación final para CADA frase, DEBES realizar un razonamiento explícito paso a paso siguiendo esta estructura:

**PASO 1: Limpieza y normalización**
- Eliminar caracteres especiales (comillas sobrantes)
- Corregir errores ortográficos obvios (ej: "bictimas" → "víctimas")
- Identificar palabras clave principales

**PASO 2: Análisis temático (búsqueda en SISTEMA TEMÁTICO)**
- ¿Qué temas principales menciona? (economía, social, género)
- Listar TODAS las posibles coincidencias en Subtema y Categoría
- Comparar con las descripciones de cada tabla
- Evaluar similitud semántica (no solo palabras exactas)

**PASO 3: Razonamiento para la tabla específica**
- ¿Cuál de las tablas en SISTEMA TEMÁTICO es la más cercana?
- Justificar: "Coincide con [tabla X] porque [razón]"
- Si hay múltiples opciones, evaluar cuál es más específica

**PASO 4: Análisis geográfico (búsqueda en SISTEMA DE CONCEPTOS)**
- ¿Menciona algún lugar? ¿Especificar tipo (departamento, ciudad, etc.)?
- Buscar en la lista de códigos disponibles
- Si el lugar no está en la lista, buscar el más cercano (ej: "Lima norte" no está, pero podría ser CIUDADES? o DEPARTAMENTO? - razonar)

**PASO 5: Evaluación de ambigüedad y multi-clasificación**
- ¿La frase podría interpretarse de múltiples maneras?
- ¿Debe generar múltiples filas? ¿Por qué?

**PASO 6: Asignación de confianza**
- Confianza temática: [X]% - Justificación breve
- Confianza geográfica: [Y]% - Justificación breve
"""
        
        if self.dt:  # Usar la versión con razonamiento detallado para mejorar caché
            return fixed_part + variable_part_dt
        else:
            return fixed_part + variable_part

# ==================== API CLIENT ====================

class DeepSeekAPIClient:
    """Cliente asíncrono para DeepSeek API con reintentos y rate limiting"""
    
    def __init__(self, config: Config):
        self.config = config
        self.headers = {
            "Authorization": f"Bearer {config.API_KEY}",
            "Content-Type": "application/json"
        }
        self.semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_BATCHES)
        self.request_times = []
        self.logger = logging.getLogger(__name__)
        
    async def _rate_limit(self):
        """Implementa rate limiting"""
        now = time.time()
        # Limpiar requests viejos (> 1 segundo)
        self.request_times = [t for t in self.request_times if now - t < 1]
        
        if len(self.request_times) >= self.config.REQUESTS_PER_SECOND:
            sleep_time = 1 - (now - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
        
        self.request_times.append(time.time())
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=60),
        retry=retry_if_exception_type((aiohttp.ClientError, asyncio.TimeoutError)),
        before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING)
    )
    async def send_request(self, prompt: str) -> Dict[str, Any]:
        """Envía request a la API con reintentos"""
        await self._rate_limit()
        
        payload = {
            "model": self.config.MODEL,
            "messages": [
                {"role": "system", "content": "Eres un clasificador semántico especializado en datos estadísticos."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,  # Bajo para clasificación consistente
            "max_tokens": 8000,
            "top_p": 0.9
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.config.API_URL,
                headers=self.headers,
                json=payload,
                timeout=self.config.REQUEST_TIMEOUT
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    # Monitorear caché hits
                    if 'usage' in result:
                        cache_hits = result['usage'].get('prompt_cache_hit_tokens', 0)
                        if cache_hits > 0:
                            self.logger.info(f"Cache hit! Ahorrados {cache_hits} tokens")
                    return result
                elif response.status == 429:  # Too Many Requests
                    retry_after = int(response.headers.get('Retry-After', 30))
                    self.logger.warning(f"Rate limit: esperando {retry_after}s")
                    await asyncio.sleep(retry_after)
                    raise aiohttp.ClientError("Rate limit exceeded")
                else:
                    error_text = await response.text()
                    self.logger.error(f"Error API: {response.status} - {error_text}")
                    raise aiohttp.ClientError(f"API error: {response.status}")

# ==================== PROCESADOR DE LOTES ====================

class BatchProcessor:
    """Procesa lotes de frases concurrentemente"""
    
    def __init__(self, config: Config, api_client: DeepSeekAPIClient, dt: bool = False):
        self.config = config
        self.api_client = api_client
        self.dt = dt
        self.prompt_builder = PromptBuilder(config, dt=dt)
        self.cache = PromptCache()
        self.checkpoint = CheckpointManager(config.CHECKPOINT_DIR)
        self.logger = logging.getLogger(__name__)
        
        # Crear directorios
        Path(config.OUTPUT_DIR).mkdir(exist_ok=True)
        Path(config.ERROR_DIR).mkdir(exist_ok=True)
        
    def _parse_response_to_dataframe(self, response_text: str, lote_id: int) -> pd.DataFrame:
        """Convierte la respuesta de la API a DataFrame"""
        try:
            # Intentar parsear CSV de la respuesta
            lines = response_text.strip().split('\n')
            
            # Buscar la línea que contiene el CSV (ignorar explicaciones)
            csv_lines = []
            for line in lines:
                if ';' in line and not line.startswith('#'):
                    csv_lines.append(line)
            
            if not csv_lines:
                self.logger.warning(f"No se encontró CSV en respuesta del lote {lote_id}")
                return pd.DataFrame()
            
            # Crear DataFrame
            import io
            csv_text = '\n'.join(csv_lines)
            df = pd.read_csv(io.StringIO(csv_text), sep=';', encoding='utf-8')
            
            # Añadir metadata del lote
            df['lote_id'] = lote_id
            df['timestamp_procesamiento'] = datetime.now().isoformat()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error parseando respuesta lote {lote_id}: {e}")
            self._guardar_respuesta_error(response_text, lote_id)
            return pd.DataFrame()
    
    def _guardar_respuesta_error(self, response_text: str, lote_id: int):
        """Guarda respuestas que no se pudieron parsear para inspección"""
        error_file = Path(self.config.ERROR_DIR) / f"lote_{lote_id}_error_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(error_file, 'w', encoding='utf-8') as f:
            f.write(response_text)
        self.logger.info(f"Respuesta de error guardada en {error_file}")
    
    async def procesar_lote(self, frases: List[str], lote_id: int, checkpoint_data: Dict, dt: bool = False) -> Optional[pd.DataFrame]:
        """Procesa un lote de frases"""
        
        async with self.api_client.semaphore:  # Control de concurrencia
            self.logger.info(f"Procesando lote {lote_id} con {len(frases)} frases")
            
            # Construir prompt
            prompt = self.prompt_builder.build_prompt(frases, lote_id, dt)
            
            # Verificar caché local
            prompt_hash = self.cache.get_prompt_hash(prompt, frases)
            cached_response = self.cache.load_from_cache(prompt_hash)
            
            if cached_response:
                self.logger.info(f"Lote {lote_id} servido desde caché local")
                response_text = cached_response['choices'][0]['message']['content']
            else:
                try:
                    # Llamar API
                    response = await self.api_client.send_request(prompt)
                    
                    # Guardar en caché local
                    self.cache.save_to_cache(prompt_hash, response)
                    
                    response_text = response['choices'][0]['message']['content']
                    
                except Exception as e:
                    self.logger.error(f"Error procesando lote {lote_id}: {e}")
                    return None
            
            # Parsear respuesta
            df_resultado = self._parse_response_to_dataframe(response_text, lote_id)
            
            if not df_resultado.empty:
                # Actualizar checkpoint
                checkpoint_data['lotes_procesados'].append(lote_id)
                checkpoint_data['frases_procesadas'] += len(frases)
                
                # Guardar resultados parciales
                output_file = Path(self.config.OUTPUT_DIR) / f"resultados_lote_{lote_id}.csv"
                df_resultado.to_csv(output_file, sep=';', index=False, encoding='utf-8')
                self.logger.info(f"Lote {lote_id} guardado en {output_file}")
                
                return df_resultado
            else:
                self.logger.warning(f"Lote {lote_id} no produjo resultados válidos")
                return None

# ==================== PROCESADOR PRINCIPAL ====================

class DeepSeekBatchClassifier:
    """Orquestador principal del procesamiento"""
    
    def __init__(self, config: Config, dt: bool = False):
        self.config = config
        self.dt = dt
        self.logger = setup_logging(config)
        self.api_client = DeepSeekAPIClient(config)
        self.processor = BatchProcessor(config, self.api_client, dt=dt)
        self.checkpoint = CheckpointManager(config.CHECKPOINT_DIR)
        
    def load_frases(self) -> pd.DataFrame:
        """Carga todas las frases a procesar"""
        self.logger.info(f"Cargando frases desde {self.config.INPUT_FILE}")
        df = pd.read_csv("../data/inputs/frases.txt" , encoding='utf-8', header=None, names=['Frase'])
        
        # Asegurar que tenemos una columna de frases
        if len(df.columns)>1:
            raise ValueError("El archivo solo debe tener una columna 'frase' o 'Frase'")
        
        col_frase = 'frase' if 'frase' in df.columns else 'Frase'
        frases = df[col_frase].dropna().tolist()
        
        self.logger.info(f"Total frases cargadas: {len(frases)}")
        return frases
    
    def crear_lotes(self, frases: List[str]) -> List[Tuple[int, List[str]]]:
        """Divide las frases en lotes"""
        lotes = []
        for i in range(0, len(frases), self.config.BATCH_SIZE):
            lote_id = i // self.config.BATCH_SIZE
            lote_frases = frases[i:i + self.config.BATCH_SIZE]
            lotes.append((lote_id, lote_frases))
        self.logger.info(f"Dividido en {len(lotes)} lotes de {self.config.BATCH_SIZE} frases")
        return lotes
    
    async def procesar_todos_los_lotes(self):
        """Procesa todos los lotes concurrentemente"""
        
        # Cargar frases
        frases = self.load_frases()
        lotes = self.crear_lotes(frases)
        
        # Cargar checkpoint
        checkpoint = self.checkpoint.load_checkpoint()
        lotes_procesados = set(checkpoint['lotes_procesados'])
        
        # Filtrar lotes ya procesados
        lotes_pendientes = [
            (lote_id, lote_frases) 
            for lote_id, lote_frases in lotes 
            if lote_id not in lotes_procesados
        ]
        
        self.logger.info(f"Lotes pendientes: {len(lotes_pendientes)} de {len(lotes)} totales")
        
        # Procesar lotes en batches de concurrencia máxima
        resultados_totales = []
        
        with tqdm(total=len(lotes_pendientes), desc="Procesando lotes") as pbar:
            for i in range(0, len(lotes_pendientes), self.config.MAX_CONCURRENT_BATCHES):
                batch_lotes = lotes_pendientes[i:i + self.config.MAX_CONCURRENT_BATCHES]
                
                # Crear tareas concurrentes
                tareas = [
                    self.processor.procesar_lote(frases_lote, lote_id, checkpoint, self.dt)
                    for lote_id, frases_lote in batch_lotes
                ]
                
                # Ejecutar concurrentemente
                resultados_batch = await asyncio.gather(*tareas)
                
                # Procesar resultados
                for resultado in resultados_batch:
                    if resultado is not None and not resultado.empty:
                        resultados_totales.append(resultado)
                
                # Actualizar checkpoint
                self.checkpoint.save_checkpoint(checkpoint)
                
                # Actualizar barra de progreso
                pbar.update(len(batch_lotes))
                
                # Pequeña pausa entre batches para evitar sobrecarga
                await asyncio.sleep(self.config.PAUSE_BETWEEN_BATCHES)
        
        # Consolidar resultados finales
        if resultados_totales:
            df_final = pd.concat(resultados_totales, ignore_index=True)
            output_final = Path(self.config.OUTPUT_DIR) / f"resultados_completos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            df_final.to_csv(output_final, sep=';', index=False, encoding='utf-8')
            self.logger.info(f"Resultados finales guardados en {output_final}")
            self.logger.info(f"Total registros clasificados: {len(df_final)}")
            
            # Estadísticas
            self._generar_estadisticas(df_final)
        else:
            self.logger.warning("No se obtuvieron resultados")
    
    def _generar_estadisticas(self, df: pd.DataFrame):
        """Genera estadísticas de calidad"""
        self.logger.info("=== ESTADÍSTICAS DE CLASIFICACIÓN ===")
        
        # Confianza promedio
        if 'confianza_tematica' in df.columns:
            conf_prom = df['confianza_tematica'].str.rstrip('%').astype(float).mean()
            self.logger.info(f"Confianza temática promedio: {conf_prom:.1f}%")
        
        if 'confianza_conceptual' in df.columns:
            conf_prom = df['confianza_conceptual'].str.rstrip('%').astype(float).mean()
            self.logger.info(f"Confianza conceptual promedio: {conf_prom:.1f}%")
        
        # Distribución por tema
        if 'nombre_tema' in df.columns:
            temas = df['nombre_tema'].value_counts()
            self.logger.info("Distribución por tema:")
            for tema, count in temas.head(10).items():
                self.logger.info(f"  {tema}: {count} registros")

# ==================== MAIN ====================

async def main():
    """Función principal"""
    
    # Configuración
    config = Config(
        API_KEY="sk-aaaaaaaaaaaaaaaaaaaaaaaaaa",  # <--- REEMPLAZAR CON TU API KEY
        BATCH_SIZE=500,  # Ajustable según resultados de pruebas
        MAX_CONCURRENT_BATCHES=5  # Ajustable según capacidad
    )
    
    # Verificar que la API key está configurada
    if not config.API_KEY:
        print("ERROR: No se encontró DEEPSEEK_API_KEY en variables de entorno")
        print("Crea un archivo .env con: DEEPSEEK_API_KEY=tu-api-key-aqui")
        print("O obtén tu API key en: https://platform.deepseek.com/")
        return
    
    # Crear clasificador y ejecutar
    classifier = DeepSeekBatchClassifier(config, dt=True)  # Usar modo pensamiento para mejorar caché
    
    try:
        await classifier.procesar_todos_los_lotes()
    except KeyboardInterrupt:
        print("\nProcesamiento interrumpido por el usuario")
        print("Los checkpoints permitirán reanudar desde donde quedó")
    except Exception as e:
        logging.getLogger(__name__).exception("Error fatal en el procesamiento")

if __name__ == "__main__":
    asyncio.run(main())