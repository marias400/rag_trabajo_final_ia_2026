# RAG Currículum UNLaR

Asistente conversacional que responde preguntas sobre el plan de estudios de
Ingeniería en Sistemas de Información (UNLaR, Plan 2024) usando RAG
(Retrieval-Augmented Generation) sobre un LLM local corriendo en LMStudio.

## Cómo funciona el sistema RAG

### Vectorización (se hace una sola vez, o cada vez que cambian los datos fuente)

El sistema combina **dos fuentes de datos distintas**, procesadas por caminos
separados y unificadas recién al final en la misma base vectorial:

1. **PDFs escaneados** (ordenanzas de la UNLaR) → extracción de texto con OCR
   (tesseract) cuando no tienen capa de texto → división en chunks.
2. **Tabla de correlatividades**, transcripta a mano en un JSON estructurado
   (`data/structured/correlatividades_ing_sistemas_2024.json`, incluye el año
   de cada asignatura) → conversión a chunks de texto natural por asignatura +
   chunks-resumen por año (uno por cada uno de los 5 años de la carrera).

   *Por qué separado del OCR:* la tabla de correlatividades es lo más
   consultado del sistema y el OCR rompe tablas (mezcla columnas, pierde
   números). Transcribirla a mano garantiza un dato 100% preciso para lo más
   crítico, en vez de depender de la extracción automática.

3. Ambos conjuntos de chunks se unifican, se generan sus **embeddings** con
   `sentence-transformers` (modelo `paraphrase-multilingual-MiniLM-L12-v2`) y
   se guardan en **ChromaDB** (persistente en `chroma_db/`).

### Consulta

Este flujo es el mismo tanto si se usa el script de consola
(`05_rag_query.py`, útil para debug rápido) como la interfaz web
(`app/streamlit_app.py`, la versión final para el usuario):

4. El usuario hace una pregunta (ej. "¿Qué correlativas tiene Cálculo
   Numérico?" o "¿Cuántas materias tiene el segundo año?").
5. La pregunta se convierte en un vector con el mismo modelo de embeddings.
6. ChromaDB busca los chunks más parecidos semánticamente.
   - Si la pregunta menciona explícitamente un año de la carrera (por
     ejemplo "primer año", "2do año", "año 3"), el sistema fuerza además la
     inclusión del chunk-resumen de ese año, porque la búsqueda semántica por
     sí sola no siempre lo prioriza bien en preguntas de tipo "agregado"
     (¿cuántas materias tiene...?).
7. Los chunks recuperados se arman en un prompt y se le envían a LMStudio
   (servidor local, API compatible con OpenAI) como contexto.
8. LMStudio genera la respuesta usando ese contexto.
9. Se muestra la respuesta final junto con los fragmentos/fuentes usados,
   para que se pueda verificar de dónde salió cada dato.

## Pasos para armar el sistema desde cero

**1. Activar el entorno virtual**

```powershell
(Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned) ; (& "z:\Documentos\UNLaR\4to\Inteligencia Artificial\trabajo_final\.venv\Scripts\Activate.ps1")
```

**2. Instalar las dependencias**

```powershell
pip install -r requirements.txt
```

> El OCR (`pytesseract`) necesita además el binario de **tesseract**
> instalado aparte en el sistema — no es un paquete de pip. Ver el detalle
> en `requirements.txt` o en el docstring de `01_extract_pdfs.py`.

**3. Extraer texto de los PDFs (con OCR si hace falta) y dividirlo en chunks**

```powershell
python ./scripts/01_extract_pdfs.py --input_dir ./data/raw --output_json ./data/processed/chunks_pdfs.json --tesseract_path "C:\Program Files\Tesseract-OCR\tesseract.exe"
```

**4. Generar los chunks de correlatividades a partir del JSON estructurado**

```powershell
python ./scripts/02_correlativities_to_chunks.py --input_json ./data/structured/correlatividades_ing_sistemas_2024.json --output_json ./data/processed/chunks_correlatividades.json
```

> Este paso también genera los 5 chunks-resumen por año (uno por cada año de
> la carrera). El JSON de entrada ya tiene el campo `anio` en cada
> asignatura, verificado contra el diagrama "Camino crítico de
> correlatividades" del Anexo II de la Ordenanza 232/23.

**5. Descargar el modelo de embeddings y cargar todos los chunks en ChromaDB**

```powershell
python ./scripts/03_build_vectordb.py --input_json ./data/processed/chunks_pdfs.json ./data/processed/chunks_correlatividades.json --db_path ./chroma_db --collection curriculum --reset
```

> Usá `--reset` siempre que hayas regenerado alguno de los JSON de chunks
> (pasos 3 o 4), para que ChromaDB no se quede con versiones viejas mezcladas
> con las nuevas.

**6. (Opcional) Probar que la recuperación semántica funciona, sin pasar por el LLM todavía**

```powershell
python ./scripts/04_query_test.py --query "¿Qué correlativas tiene Cálculo Numérico?"
```

**7. Levantar LMStudio**

Necesitás LMStudio instalado, con un modelo cargado y el servidor local
corriendo en el puerto 1234 (por defecto).

Modelos probados:
- **Llama 3.2 3B** — funciona, respuestas más simples/rápidas.
- **Qwen3 14B** — mejor calidad y mejor español, pero notablemente más lento
  (puede necesitar timeouts de 300-400s si no hay suficiente VRAM).
- **Qwen3 8B** — punto intermedio recomendado: misma familia que el 14B (buen
  seguimiento de instrucciones en español) pero bastante más rápido al tener
  menos parámetros.

**8. Probar el pipeline completo por consola**

```powershell
python ./scripts/05_rag_query.py --query "¿Qué correlativas tiene Cálculo Numérico?" --show_prompt
```

Este comando recupera los chunks relevantes, arma el prompt, lo manda a
LMStudio y muestra la respuesta junto con los fragmentos usados como fuente.

**9. Lanzar la interfaz web**

```powershell
streamlit run app/streamlit_app.py
```

## Notas / troubleshooting

- El warning `Warning: You are sending unauthenticated requests to the HF Hub`
  al cargar el modelo de embeddings es inofensivo — solo avisa que no hay un
  `HF_TOKEN` configurado, no afecta el funcionamiento.
- Si `05_rag_query.py` tira `ReadTimeoutError`, subí el timeout con
  `--timeout` (ej. `--timeout 300`) antes de asumir que LMStudio está
  colgado; con modelos grandes y poca VRAM, es simplemente lento.
