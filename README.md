# RAG Currículum UNLaR

Asistente conversacional que responde preguntas sobre los planes de estudios de
**Ingeniería en Sistemas**, **Licenciatura en Sistemas** e **Ingeniería Mecatrónica**
(UNLaR, Plan 2024), usando RAG (Retrieval-Augmented Generation) sobre un LLM local
corriendo en LMStudio.

---

## Cómo funciona el sistema RAG

El sistema se divide en dos grandes fases: **Ingesta** (procesar los documentos una
sola vez) y **Consulta** (responder preguntas en tiempo real). Toda la lógica de
consulta está encapsulada en `scripts/rag_core.py` y es compartida por la consola
(`05_rag_query.py`) y la interfaz web (`app/streamlit_app.py`).

### Fase 1 — Ingesta (offline, una sola vez)

El sistema combina **dos fuentes de datos** procesadas por caminos separados y
unificadas al final en la misma base vectorial (ChromaDB):

| Fuente | Script | Qué hace |
|---|---|---|
| PDFs de ordenanzas (`data/raw/`) | `01b_extract_docling.py` | Extrae texto respetando estructura (títulos, secciones, tablas) con Docling. Opcionalmente aplica Semantic Chunking para cortar por coherencia temática en vez de por tamaño fijo. |
| Tablas de correlatividades (`data/structured/*.json`) | `02_correlativities_to_chunks.py` | Convierte los JSON estructurados en chunks de texto natural (uno por asignatura + un resumen por año). Se procesan aparte del OCR porque las tablas escaneadas pierden columnas y números. |

Todos los chunks resultantes se vectorizan con `paraphrase-multilingual-MiniLM-L12-v2`
y se guardan en ChromaDB (`03_build_vectordb.py`).

### Fase 2 — Consulta (en tiempo real)

Cuando el usuario hace una pregunta, el sistema ejecuta los siguientes pasos en orden:

#### 2.1 Routing y Query Rewriting (siempre activo)

Una llamada corta al LLM que cumple dos funciones simultáneas:

- **Routing:** Decide si la pregunta necesita buscar en la base o se puede responder
  con el historial (saludos, aclaraciones, resúmenes).
- **Rewriting:** Si necesita buscar, reescribe la pregunta para optimizar la búsqueda:
  resuelve pronombres ("esa materia" → nombre real), expande siglas ("BD" → "Bases de
  Datos"), agrega sinónimos del dominio, infiere la carrera del contexto y elimina
  muletillas.

Además, `detect_career_stems()` identifica la carrera mencionada y prioriza
automáticamente los chunks de esa carrera.

#### 2.2 Query Enhancement (opcional, acumulativo)

Tres técnicas que enriquecen la búsqueda antes de ir a ChromaDB. Se pueden activar
individualmente o **todas al mismo tiempo** — el sistema acumula las consultas de
cada técnica en una sola lista y las busca juntas:

| Técnica | Qué hace | Cuándo conviene |
|---|---|---|
| **Multi-Query** | Genera hasta 5 reformulaciones de la pregunta con distintas palabras clave. | La información puede estar expresada con terminología diferente en los documentos. |
| **HyDE** (Hypothetical Document Embeddings) | El LLM genera un párrafo hipotético de respuesta y se busca con el embedding de ese texto en vez del de la pregunta. | Preguntas técnicas sobre ordenanzas y documentos formales. |
| **Query Decomposition** | Descompone preguntas complejas en sub-preguntas simples e independientes. | Preguntas que mezclan múltiples aspectos (año + carrera + tipo de correlativa). |

Cada técnica agrega ~3–10 segundos de latencia (una llamada al LLM local).

#### 2.3 Recuperación (Retrieval)

La pregunta reformulada (o la lista acumulada de queries del paso anterior) se busca
en ChromaDB:

- **Búsqueda semántica** (siempre): convierte la query en vector y busca por similitud coseno.
- **Búsqueda híbrida** (opcional, `--hybrid`): combina la semántica con búsqueda
  léxica exacta BM25 y fusiona ambos rankings con **Reciprocal Rank Fusion (RRF)**.
  Evita perder nombres propios o siglas que la semántica sola podría ignorar.
- Si hay múltiples queries (por Multi-Query, HyDE o Decomposition), cada una recupera
  sus propios chunks y todos se fusionan con RRF ponderado.
- Si la pregunta menciona un año ("primer año", "2do año"), se fuerza la inclusión del
  chunk-resumen de ese año.

#### 2.4 Reranking (opcional)

Los fragmentos recuperados se pasan por un modelo **Cross-Encoder**
(`BAAI/bge-reranker-v2-m3`) que evalúa cada par (pregunta, fragmento) conjuntamente.
A diferencia del embedding bi-encoder (rápido pero aproximado), el Cross-Encoder lee
ambos textos al mismo tiempo y asigna un puntaje de relevancia mucho más preciso,
reordenando los fragmentos de mayor a menor utilidad real.

#### 2.5 Generación

Los mejores fragmentos + el historial conversacional se inyectan en el prompt final y
se envían a LMStudio. El LLM genera la respuesta y el sistema la muestra junto con las
fuentes utilizadas para que el usuario pueda verificar de dónde salió cada dato.

---

## Armar el sistema desde cero

### 1. Entorno virtual y dependencias

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& ".\.venv\Scripts\Activate.ps1")
pip install -r requirements.txt
```

> El OCR (`pytesseract`) necesita el binario de **tesseract** instalado aparte en
> Windows — no es un paquete de pip.

### 2. Extraer texto de los PDFs

Hay 4 opciones de chunking. La **D** es la recomendada:

**A) Chunking clásico** — corta por fronteras naturales de texto (párrafos, oraciones, espacios) con máximo 800 caracteres y 150 de solapamiento:
```powershell
python ./scripts/01_extract_pdfs.py `
    --input_dir ./data/raw `
    --output_json ./data/processed/chunks_pdfs.json `
    --tesseract_path "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

**B) Chunking semántico** — agrupa oraciones por similitud coseno (≥ 0.75), cortando en saltos de tema:
```powershell
python ./scripts/01_extract_pdfs.py `
    --input_dir ./data/raw `
    --output_json ./data/processed/chunks_pdfs.json `
    --semantic --threshold 0.75 --max_words 400
```

**C) Chunking estructural (Docling)** — respeta títulos, secciones y tablas del PDF:
```powershell
python ./scripts/01b_extract_docling.py `
    --input_dir ./data/raw `
    --output_json ./data/processed/chunks_pdfs.json
```

**D) Híbrido: Estructural + Semántico (recomendado)** — Docling para la estructura + Semantic Chunking para las fronteras:
```powershell
python ./scripts/01b_extract_docling.py `
    --input_dir ./data/raw `
    --output_json ./data/processed/chunks_pdfs.json `
    --semantic --threshold 0.75 --max_words 400
```

### 3. Generar chunks de correlatividades

```powershell
python ./scripts/02_correlativities_to_chunks.py `
    --input_json ./data/structured/correlatividades_ing_sistemas_2024.json `
                 ./data/structured/correlatividades_lic_sistemas_2024.json `
                 ./data/structured/correlatividades_ing_mecatronica_2024.json `
    --output_dir ./data/processed
```

Genera un JSON de chunks por carrera en `data/processed/`. Para agregar una carrera
nueva: crear su JSON en `data/structured/` y agregarlo a `--input_json` de este paso
y del siguiente.

### 4. Construir la base vectorial (ChromaDB)

```powershell
python ./scripts/03_build_vectordb.py `
    --input_json ./data/processed/chunks_pdfs.json `
                 ./data/processed/chunks_correlatividades_ing_sistemas_2024.json `
                 ./data/processed/chunks_correlatividades_lic_sistemas_2024.json `
                 ./data/processed/chunks_correlatividades_ing_mecatronica_2024.json `
    --db_path ./chroma_db --collection curriculum --reset
```

> Usá `--reset` siempre que hayas regenerado algún JSON de chunks para evitar
> mezclar versiones viejas con nuevas.

### 5. (Opcional) Probar el retrieval sin LLM

```powershell
python ./scripts/04_query_test.py --query "¿Qué correlativas tiene Cálculo Numérico?"
```

### 6. Levantar LMStudio

Necesitás LMStudio con un modelo cargado y el servidor local corriendo en el puerto
1234 (default).

- **Modelo recomendado:** Llama 3.2 3B Instruct (rápido, preciso, optimizado para este proyecto).
- Alternativas: Qwen3 8B, Mistral 7B.

### 7. Probar el pipeline completo por consola

```powershell
python ./scripts/05_rag_query.py --query "¿Qué correlativas tiene Cálculo Numérico?" --show_prompt
```

Flags opcionales de retrieval avanzado:

| Flag | Efecto |
|---|---|
| `--hybrid` | Activa búsqueda híbrida (semántica + BM25 + RRF). |
| `--reranker BAAI/bge-reranker-v2-m3` | Activa reranking con Cross-Encoder. Primera vez descarga ~1GB. |
| `--multi_query` | Genera 5 variantes de la pregunta, fusiona con RRF. |
| `--hyde` | Genera documento hipotético, busca con su embedding. |
| `--decompose` | Descompone en sub-preguntas, fusiona con RRF. |
| `--show_prompt` | Muestra el prompt enviado al LLM y detalles de debug. |

Todos los flags son combinables entre sí. Ejemplo con todo activado:
```powershell
python ./scripts/05_rag_query.py `
    --query "¿Cuántas materias de 3er año de Sistemas tienen correlativas de 2do año?" `
    --hybrid --reranker BAAI/bge-reranker-v2-m3 `
    --multi_query --hyde --decompose --show_prompt
```

### 8. Lanzar la interfaz web

```powershell
streamlit run app/streamlit_app.py
```

La interfaz web incluye todas las opciones anteriores como checkboxes en la barra
lateral. Detecta automáticamente la carrera mencionada en cada consulta.

---

## Referencia técnica

### Datos estructurados (`data/structured/`)

| Archivo | Carrera | Fuente |
|---|---|---|
| `correlatividades_ing_sistemas_2024.json` | Ingeniería en Sistemas | Ord. CS N° 232/23 |
| `correlatividades_lic_sistemas_2024.json` | Licenciatura en Sistemas | Ord. CS N° 236/23 |
| `correlatividades_ing_mecatronica_2024.json` | Ingeniería Mecatrónica | Ord. CS N° 234/23 |

### Semantic Chunking — Algoritmo

Implementado en `scripts/semantic_chunker.py`:

1. **Segmentación en oraciones** con regex optimizada para español (protege abreviaturas: *Art.*, *Ord.*, *Ing.*, *Lic.*, *N°*, etc.).
2. **Embedding por oración** con `paraphrase-multilingual-MiniLM-L12-v2`.
3. **Detección de fronteras** por similitud coseno entre oraciones consecutivas. Si cae debajo del `--threshold` (default 0.75), se corta.
4. **Post-procesamiento**: chunks < `--min_words` (30) se fusionan con el siguiente; chunks > `--max_words` (400) se dividen en la frontera de oración más cercana al medio.

### Chunking Estructural (Docling) — Algoritmo

Implementado en `scripts/01b_extract_docling.py`:

1. Docling (IBM) analiza el maquetado del PDF reconociendo la jerarquía (Capítulos, Artículos, Secciones, Tablas).
2. `HierarchicalChunker` genera chunks que nunca cruzan límites de sección.
3. Las tablas se convierten a Markdown con delimitadores `|`.
4. Cada chunk incluye metadatos: `heading` (jerarquía completa), `heading_level`, `is_table`.
5. Fallback automático a `pdfplumber` si un PDF no se puede parsear con Docling.

> Los modelos de layout de Docling (~600 MB) se descargan automáticamente la primera vez.

---

## Troubleshooting

- `Warning: You are sending unauthenticated requests to the HF Hub` → Inofensivo, no afecta el funcionamiento.
- `ReadTimeoutError` en `05_rag_query.py` → Subir el timeout con `--timeout 300` antes de asumir que LMStudio está colgado.
