#!/usr/bin/env python3
"""
rag_core.py
------------
Lógica compartida del pipeline de consulta RAG (recuperación en ChromaDB +
llamadas a LMStudio), extraída para que la reutilicen tanto un script de
consola (05_rag_query.py) como cualquier app (ej. Streamlit) sin duplicar
código ni arreglar bugs en dos lugares distintos.

Qué vive acá:
- Carga cacheada del modelo de embeddings y del cliente de ChromaDB.
- retrieve_chunks(): recuperación semántica + forzado de chunk-resumen de año
  (la lógica que ya tenía 05_rag_query.py, sin cambios de comportamiento).
- call_lmstudio_chat(): llamada genérica al endpoint de chat de LMStudio,
  recibiendo una lista de mensajes ya armada (no un solo string), para poder
  mandar el historial real de la conversación como mensajes de chat.
- route_query(): el router conversacional. Antes de cada pregunta nueva,
  decide (a) si hace falta consultar la base para responder, y (b) si hace
  falta, reformula la pregunta como una consulta autocontenida (sin
  pronombres tipo "esa", "la anterior") para que el embedding tenga algo
  con qué buscar.
- build_rag_user_message(): arma el bloque de contexto + pregunta que se le
  manda al LLM como mensaje de usuario cuando sí hubo retrieval.

Nada de esto imprime en pantalla ni maneja argparse: eso queda en los
scripts que lo usan (05_rag_query.py, streamlit_app.py, etc).

Manejo de errores: call_lmstudio_chat() (y route_query(), que la usa
internamente) NUNCA llaman sys.exit() ni imprimen nada — levantan
LMStudioError con un mensaje ya listo para mostrarle al usuario. Un
sys.exit() acá tumbaría el proceso completo del servidor si esto corre
dentro de Streamlit, no solo la request actual, así que cada caller decide
qué hacer con la excepción (05_rag_query.py la imprime y corta con
sys.exit; streamlit_app.py la muestra con st.error y sigue vivo).
"""

import re
import sys

try:
    import chromadb
except ImportError:
    sys.exit("Falta chromadb. Instalalo con: pip install chromadb --break-system-packages")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Falta sentence-transformers. Instalalo con: pip install sentence-transformers --break-system-packages")

try:
    import requests
except ImportError:
    sys.exit("Falta requests. Instalalo con: pip install requests --break-system-packages")


SYSTEM_PROMPT = (
    "Sos un asistente experto sobre los planes de estudios de las carreras de la UNLaR "
    "(Ingeniería en Sistemas, Licenciatura en Sistemas e Ingeniería Mecatrónica).\n"
    "Respondé las preguntas utilizando la información de los fragmentos de contexto "
    "provistos y el historial de la conversación.\n\n"
    "Ámbito de tus capacidades:\n"
    "- Podés responder sobre asignaturas de las 3 carreras, sus regímenes de cursado (anual, 1C, 2C) y el año de cursada.\n"
    "- Podés detallar las correlativas necesarias para cursar o rendir examen final.\n"
    "- Podés describir perfiles de egreso, incumbencias profesionales y estructuras curriculares según las ordenanzas.\n\n"
    "Instrucciones estrictas:\n"
    "1. Basate ÚNICAMENTE en la información de los fragmentos provistos para responder datos del plan de estudios.\n"
    "2. Si la respuesta está en los fragmentos, brindá la información completa y precisa (requisitos de cursado, examen final, año, etc.).\n"
    "3. Indicá la fuente de la información citando ÚNICAMENTE los fragmentos que aporten datos útiles (ej. 'Según el Fragmento 1...').\n"
    "4. NO menciones fragmentos que no contengan información relevante para la pregunta (ej. NO digas 'El Fragmento X no menciona esto' ni hagas un listado de fragmentos inútiles). Ignorá los fragmentos no relevantes en completo silencio.\n"
    "5. Si absolutamente ningún fragmento provisto contiene datos útiles para responder la pregunta, indicá brevemente que no disponés de esa información en los documentos del contexto.\n"
    "6. Respondé siempre en español, con un tono claro, directo y profesional."
)


ROUTER_SYSTEM_PROMPT = (
    "Sos el enrutador de un sistema RAG sobre los planes de estudios de las carreras de la UNLaR.\n"
    "Analizá el historial de conversación y la PREGUNTA NUEVA del usuario.\n\n"
    "Reglas de decisión:\n"
    "- RETRIEVAL: SI → Si la pregunta solicita información sobre asignaturas, correlativas, régimen, años, requisitos, perfil de egreso, incumbencias o cualquier dato del plan de estudios de la UNLaR.\n"
    "- RETRIEVAL: NO → Si es un saludo, agradecimiento, despedida, pide aclarar/resumir algo dicho previamente, o si es una PREGUNTA TOTALMENTE AJENA AL DOMINIO (ej. preguntas de cultura general, ocio, seguridad, instrucciones ilegales, o cosas que no pertenezcan al plan de estudios de la universidad).\n\n"
    "Si RETRIEVAL es SI, reformulá la pregunta como una consulta AUTOCONTENIDA y OPTIMIZADA para búsqueda semántica:\n"
    "1. Resolvé pronombres y referencias ambiguas usando el historial (ej. 'esa materia' → nombre real de la asignatura).\n"
    "2. Expandí siglas y abreviaturas del dominio (ej. 'BD' → 'Bases de Datos', 'SO' → 'Sistemas Operativos', 'POO' → 'Programación Orientada a Objetos', 'AM' → 'Análisis Matemático', 'FQ' → 'Física y Química').\n"
    "3. Si el contexto del historial permite inferir la carrera, mencionala explícitamente (ej. 'Ingeniería en Sistemas', 'Ingeniería Mecatrónica', 'Licenciatura en Sistemas').\n"
    "4. Eliminá muletillas y frases vacías que contaminen el embedding (ej. '¿Podés decirme...?' → pregunta directa).\n\n"
    "Elegí la ESTRATEGIA de búsqueda según estas reglas PRECISAS (en orden de prioridad):\n\n"
    "1. ESTRATEGIA: decompose\n"
    "   CUÁNDO: La pregunta menciona DOS O MÁS materias distintas, o pide datos de DISTINTOS tipos a la vez (ej. correlativas + régimen + año en una sola pregunta).\n"
    "   SEÑALES: conjunciones entre nombres de materias ('y', 'también', 'además'), listados, preguntas con múltiples signos de interrogación.\n"
    "   NO usar si la pregunta solo menciona una materia aunque sea larga.\n\n"
    "2. ESTRATEGIA: hyde\n"
    "   CUÁNDO: La pregunta es CONCEPTUAL, COMPARATIVA o NARRATIVA — no pregunta por datos puntuales sino por descripciones, objetivos, diferencias o características generales de carreras o del sistema educativo.\n"
    "   SEÑALES: palabras como 'perfil', 'egresado', 'incumbencias', 'enfoque', 'objetivo', 'diferencia entre', 'qué hace', 'qué estudia', 'cómo es', 'por qué', 'cuál es el propósito'.\n"
    "   TAMBIÉN usar si la pregunta habla de normativas generales, reglamentos o estructura curricular sin nombrar una materia específica.\n\n"
    "3. ESTRATEGIA: multi_query\n"
    "   CUÁNDO (cualquiera de estas condiciones):\n"
    "   a) La pregunta tiene MENOS DE 6 PALABRAS SIGNIFICATIVAS (sin artículos ni preposiciones).\n"
    "   b) Usa jerga informal, abreviaturas no estándar o términos ambiguos que pueden referirse a varias cosas (ej. 'proba', 'redes', 'sistemas', 'calculo', 'fisica').\n"
    "   c) La pregunta no nombra una materia exacta sino una TEMÁTICA AMPLIA (ej. 'materias de programación', 'asignaturas relacionadas con matemática', 'todo lo de segundo año').\n"
    "   d) La pregunta podría matchear con varios documentos distintos por ser genérica.\n\n"
    "4. ESTRATEGIA: direct\n"
    "   CUÁNDO: La pregunta nombra UNA materia específica con su nombre completo o casi completo Y pregunta por UN solo dato concreto (correlativas, año, régimen, código). SOLO si no aplica ninguna de las estrategias anteriores.\n\n"
    "REGLA DE DESEMPATE: ante la duda entre direct y multi_query, elegí multi_query. Es mejor generar variantes de más que de menos.\n\n"
    "Formato de respuesta OBLIGATORIO (no agregues introducciones ni explicaciones):\n"
    "RETRIEVAL: SI\n"
    "PREGUNTA: <pregunta reformulada>\n"
    "ESTRATEGIA: <direct|decompose|multi_query|hyde>\n"
    "RAZON: <una frase corta explicando por qué elegiste esa estrategia>\n\n"
    "O bien:\n"
    "RETRIEVAL: NO\n"
    "PREGUNTA: (no aplica)\n"
    "ESTRATEGIA: direct\n"
    "RAZON: no necesita buscar\n\n"
    "EJEMPLOS:\n\n"
    "PREGUNTA NUEVA: ¿Cómo robar un banco?\n"
    "RETRIEVAL: NO\n"
    "PREGUNTA: (no aplica)\n"
    "ESTRATEGIA: direct\n"
    "RAZON: pregunta totalmente ajena al dominio y planes de estudios de la universidad\n\n"
    "PREGUNTA NUEVA: ¿Qué correlativas tiene Cálculo Numérico?\n"
    "RETRIEVAL: SI\n"
    "PREGUNTA: correlativas de Cálculo Numérico Ingeniería en Sistemas UNLaR\n"
    "ESTRATEGIA: direct\n"
    "RAZON: pregunta sobre una sola materia y un solo dato concreto\n\n"
    "PREGUNTA NUEVA: bd\n"
    "RETRIEVAL: SI\n"
    "PREGUNTA: Bases de Datos correlativas régimen año plan de estudios\n"
    "ESTRATEGIA: multi_query\n"
    "RAZON: consulta de una sola palabra ambigua, menos de 6 palabras significativas\n\n"
    "PREGUNTA NUEVA: materias de programación sistemas\n"
    "RETRIEVAL: SI\n"
    "PREGUNTA: asignaturas de programación en Ingeniería en Sistemas UNLaR\n"
    "ESTRATEGIA: multi_query\n"
    "RAZON: temática amplia sin materia específica, podría matchear con muchos documentos\n\n"
    "PREGUNTA NUEVA: proba\n"
    "RETRIEVAL: SI\n"
    "PREGUNTA: Probabilidad y Estadística correlativas régimen cursado plan estudios\n"
    "ESTRATEGIA: multi_query\n"
    "RAZON: jerga informal extremadamente corta, alta ambigüedad\n\n"
    "PREGUNTA NUEVA: ¿Qué correlativas tienen Análisis Matemático II y Cálculo Numérico?\n"
    "RETRIEVAL: SI\n"
    "PREGUNTA: correlativas de Análisis Matemático II y Cálculo Numérico Ingeniería en Sistemas\n"
    "ESTRATEGIA: decompose\n"
    "RAZON: pregunta sobre dos materias distintas a la vez\n\n"
    "PREGUNTA NUEVA: dame las correlativas, el año y el régimen de Física I\n"
    "RETRIEVAL: SI\n"
    "PREGUNTA: correlativas año régimen de cursado de Física I Ingeniería en Sistemas\n"
    "ESTRATEGIA: decompose\n"
    "RAZON: pide tres tipos de datos distintos sobre la misma materia\n\n"
    "PREGUNTA NUEVA: ¿Cuál es el perfil de egreso de Ingeniería Mecatrónica?\n"
    "RETRIEVAL: SI\n"
    "PREGUNTA: perfil de egreso e incumbencias profesionales de Ingeniería Mecatrónica UNLaR\n"
    "ESTRATEGIA: hyde\n"
    "RAZON: pregunta conceptual/narrativa sobre características generales de la carrera\n\n"
    "PREGUNTA NUEVA: ¿Qué diferencia hay entre Ingeniería en Sistemas y Licenciatura en Sistemas?\n"
    "RETRIEVAL: SI\n"
    "PREGUNTA: diferencias entre Ingeniería en Sistemas y Licenciatura en Sistemas UNLaR perfil egreso incumbencias estructura\n"
    "ESTRATEGIA: hyde\n"
    "RAZON: pregunta comparativa y conceptual entre dos carreras\n\n"
    "PREGUNTA NUEVA: ¿Cómo es el plan de estudios de mecatrónica?\n"
    "RETRIEVAL: SI\n"
    "PREGUNTA: estructura curricular plan de estudios Ingeniería Mecatrónica UNLaR\n"
    "ESTRATEGIA: hyde\n"
    "RAZON: pregunta narrativa sobre la estructura general de la carrera, sin materia específica"
)



class LMStudioError(Exception):
    """Error al hablar con LMStudio (conexión, timeout, o HTTP).
    El mensaje ya viene formateado y listo para mostrarle al usuario tal
    cual (por consola o en la UI de Streamlit)."""
    pass


ORDINAL_ANIO = {
    "primer": 1, "1er": 1, "1ro": 1,
    "segundo": 2, "2do": 2,
    "tercer": 3, "tercero": 3, "3er": 3,
    "cuarto": 4, "4to": 4,
    "quinto": 5, "5to": 5,
}

# Caches simples en memoria de proceso: evitan recargar el modelo de
# embeddings o reabrir el cliente de Chroma en cada pregunta de una sesión
# de chat (antes cada llamada a retrieve_chunks() hacía SentenceTransformer(...)
# desde cero, lo cual es carísimo si se repite muchas veces por conversación).
_model_cache: dict[str, "SentenceTransformer"] = {}
_client_cache: dict[str, "chromadb.PersistentClient"] = {}
_bm25_cache: dict[str, tuple] = {}
_reranker_cache: dict[str, "CrossEncoder"] = {}


def get_embedding_model(model_name: str) -> "SentenceTransformer":
    if model_name not in _model_cache:
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


def get_chroma_collection(db_path: str, collection_name: str):
    if db_path not in _client_cache:
        _client_cache[db_path] = chromadb.PersistentClient(path=db_path)
    client = _client_cache[db_path]
    return client.get_collection(collection_name)


def get_bm25_index(db_path: str, collection_name: str):
    key = f"{db_path}_{collection_name}"
    if key not in _bm25_cache:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            sys.exit("Falta rank-bm25. Instalalo con: pip install rank-bm25")
        
        collection = get_chroma_collection(db_path, collection_name)
        data = collection.get()
        docs = data.get("documents", [])
        metas = data.get("metadatas", [])
        
        # Tokenización simple para BM25
        tokenized_corpus = [re.findall(r"\w+", str(doc).lower()) for doc in docs]
        bm25 = BM25Okapi(tokenized_corpus)
        _bm25_cache[key] = (bm25, docs, metas)
    return _bm25_cache[key]


def get_reranker(model_name: str):
    if model_name not in _reranker_cache:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            sys.exit("Falta sentence-transformers. Instalalo con: pip install sentence-transformers")
        print(f"Cargando modelo reranker '{model_name}'...")
        _reranker_cache[model_name] = CrossEncoder(model_name)
    return _reranker_cache[model_name]


def detect_anio_query(query: str) -> int | None:
    """
    Si la pregunta menciona explícitamente un año de la carrera (ej. "primer año",
    "2do año", "año 3"), devuelve el número de año (1-5). Si no, devuelve None.
    """
    q = query.lower()
    if "año" not in q and "anio" not in q and "ano " not in q:
        return None
    for palabra, anio in ORDINAL_ANIO.items():
        if palabra in q:
            return anio
    m = re.search(r"a[nñ]o\s*(?:n[uú]mero\s*)?(\d)\b", q)
    if m:
        return int(m.group(1))
    return None


_ASIGNATURAS_CACHE = None


def _normalize_text(text: str) -> str:
    import unicodedata
    text = text.lower()
    return "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )


def _load_asignaturas_map():
    global _ASIGNATURAS_CACHE
    if _ASIGNATURAS_CACHE is not None:
        return _ASIGNATURAS_CACHE
    
    import json
    from pathlib import Path
    dir_path = Path(__file__).parent.parent / "data" / "structured"
    _ASIGNATURAS_CACHE = []
    if dir_path.is_dir():
        for path in sorted(dir_path.glob("correlatividades_*.json")):
            try:
                stem = path.stem
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                    for asig in data.get("asignaturas", []):
                        _ASIGNATURAS_CACHE.append({
                            "n": asig["n"],
                            "nombre": asig["nombre"],
                            "norm": _normalize_text(asig["nombre"]),
                            "stem": stem,
                        })
            except Exception:
                pass
    _ASIGNATURAS_CACHE.sort(key=lambda x: len(x["norm"]), reverse=True)
    return _ASIGNATURAS_CACHE


ROMAN_TO_INT = {
    "i": 1, "1": 1,
    "ii": 2, "2": 2,
    "iii": 3, "3": 3,
    "iv": 4, "4": 4,
    "v": 5, "5": 5,
    "vi": 6, "6": 6,
}


def _extract_base_name_and_roman(norm_name: str) -> tuple[str, int | None]:
    m = re.search(r"\s+(iii|ii|i|iv|v|vi|\d)$", norm_name)
    if m:
        roman_str = m.group(1)
        base = norm_name[:m.start()].strip()
        num = ROMAN_TO_INT.get(roman_str)
        return base, num
    return norm_name.strip(), None


def detect_asignaturas_query(query: str) -> list[tuple[str, int]]:
    """
    Detecta todas las asignaturas del plan mencionadas en la consulta (ej. 'bases de datos' -> [('correlatividades_ing_sistemas_2024', 25), ...]).
    Maneja plurales y números romanos/arábigos.
    Devuelve lista de tuplas (stem, n) de asignaturas.
    """
    q_norm = _normalize_text(query)
    q_norm = re.sub(r"\bbases\b", "base", q_norm)
    q_norm = re.sub(r"\balgoritmos\b", "algoritmo", q_norm)
    q_norm = re.sub(r"\bredes\b", "red", q_norm)
    q_norm = re.sub(r"\bsistemas\b", "sistema", q_norm)

    asignaturas = _load_asignaturas_map()
    
    base_map = {}
    for asig in asignaturas:
        norm_name = asig["norm"]
        norm_name = re.sub(r"\bbases\b", "base", norm_name)
        norm_name = re.sub(r"\balgoritmos\b", "algoritmo", norm_name)
        norm_name = re.sub(r"\bredes\b", "red", norm_name)
        norm_name = re.sub(r"\bsistemas\b", "sistema", norm_name)
        
        base, num = _extract_base_name_and_roman(norm_name)
        base_map.setdefault(base, []).append((num, asig["stem"], asig["n"]))

    matched = []
    
    for base, num_list in base_map.items():
        if len(base) < 4:
            pattern = r"\b" + re.escape(base) + r"\b"
            found = bool(re.search(pattern, q_norm))
        else:
            found = base in q_norm
            
        if found:
            m_num = re.search(re.escape(base) + r"\s+(iii|ii|i|iv|v|vi|\d)\b", q_norm)
            if m_num:
                req_num = ROMAN_TO_INT.get(m_num.group(1))
                for num, stem, asig_n in num_list:
                    if num == req_num:
                        matched.append((stem, asig_n))
            else:
                for num, stem, asig_n in num_list:
                    matched.append((stem, asig_n))

    res = []
    for item in matched:
        if item not in res:
            res.append(item)
    return res


def detect_career_stems(query: str) -> list[str]:
    """Detects which career(s) the query refers to and returns matching stem prefixes.
    Returns a list of stem strings (e.g., "correlatividades_ing_sistemas_2024")
    or empty list if none detected.
    """
    q = _normalize_text(query)
    stems = []
    # Map keywords to possible stems (filenames without .json)
    career_map = {
        "sistemas": ["correlatividades_ing_sistemas_2024", "correlatividades_lic_sistemas_2024"],
        "licenciatura": ["correlatividades_lic_sistemas_2024"],
        "mecatrónica": ["correlatividades_ing_mecatronica_2024"],
        "mecatronica": ["correlatividades_ing_mecatronica_2024"],
    }
    for key, possible in career_map.items():
        if key in q:
            stems.extend(possible)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for s in stems:
        if s not in seen:
            seen.add(s)
            unique.append(s)
    return unique

def retrieve_chunks(
    query: str,
    db_path: str,
    collection_name: str,
    model_name: str,
    n_results: int,
    use_hybrid: bool = False,
    reranker_model: str = None,
    debug_data: dict = None,
):
    """Recupera los n_results chunks más relevantes para query, forzando
    además el chunk estructurado de asignatura o de año si la pregunta los menciona.
    Soporta búsqueda híbrida (BM25 + Semántica + RRF) y reranking con CrossEncoder.

    Si se pasa `debug_data` (dict vacío), lo rellena con telemetría de cada paso.
    """
    collection = get_chroma_collection(db_path, collection_name)

    # --- Filtro de Metadatos de Carrera ---
    where_filter = None
    q_lower = query.lower()
    
    source_patterns = []
    # 1. Mecatrónica
    if "mecatronica" in q_lower or "mecatrónica" in q_lower:
        source_patterns.append("MECATRNICA")
        
    # 2. Licenciatura en Sistemas
    # Si dice explícitamente licenciatura o lic, o si NO dice ingenieria pero sí sistemas
    if "licenciatura" in q_lower or "lic." in q_lower:
        source_patterns.append("LIC_EN_SISTEMAS")
    # 3. Ingeniería en Sistemas
    elif "ingenieria" in q_lower or "ingeniería" in q_lower or "ing." in q_lower:
        if "sistemas" in q_lower:
            source_patterns.append("ING_EN_SISTEMAS")
            source_patterns.append("Ing_en_Sistemas")
    # 4. Búsqueda genérica de Sistemas (si solo dice sistemas sin especificar)
    elif "sistemas" in q_lower:
        source_patterns.append("SISTEMAS")
        source_patterns.append("Sistemas")

    if source_patterns:
        # Si hay un único patrón
        if len(source_patterns) == 1:
            where_filter = {"source": {"$contains": source_patterns[0]}}
        else:
            # Si hay múltiples patrones compatibles
            where_filter = {"$or": [{"source": {"$contains": p}} for p in source_patterns]}

    if not use_hybrid:
        model = get_embedding_model(model_name)
        query_embedding = model.encode([query]).tolist()
        results = collection.query(query_embeddings=query_embedding, n_results=n_results, where=where_filter)

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]
        retrieved = list(zip(docs, metas, distances))

        if debug_data is not None:
            debug_data["mode"] = "solo_semantica"
            debug_data["semantica"] = {
                "n_candidatos": len(docs),
                "top_distancias": [round(d, 4) for d in distances[:5]],
            }
    else:
        # 1. Búsqueda Semántica
        k_candidates = max(n_results * 3, 30)
        model = get_embedding_model(model_name)
        query_embedding = model.encode([query]).tolist()
        sem_results = collection.query(query_embeddings=query_embedding, n_results=k_candidates, where=where_filter)

        sem_docs = sem_results["documents"][0]
        sem_metas = sem_results["metadatas"][0]
        sem_distances = sem_results["distances"][0]

        if debug_data is not None:
            debug_data["mode"] = "hibrida"
            debug_data["semantica"] = {
                "n_candidatos": len(sem_docs),
                "top_distancias": [round(d, 4) for d in sem_distances[:5]],
                "top_fuentes": [
                    sem_metas[i].get("source", "?") for i in range(min(5, len(sem_metas)))
                ],
            }

        # 2. Búsqueda Léxica (BM25)
        bm25, bm25_docs, bm25_metas = get_bm25_index(db_path, collection_name)
        tokenized_query = re.findall(r"\w+", query.lower())
        bm25_scores = bm25.get_scores(tokenized_query)

        # Aplicar el mismo filtro de carrera sobre los scores de BM25
        filtered_bm25_scores = []
        for idx, (doc, meta) in enumerate(zip(bm25_docs, bm25_metas)):
            score = bm25_scores[idx]
            # Si hay un filtro activo, verificar si el documento coincide con alguna de las palabras clave de carrera
            if source_patterns:
                source_name = meta.get("source", "")
                if not any(pattern in source_name for pattern in source_patterns):
                    score = -9999.0  # Penalización extrema
            filtered_bm25_scores.append(score)

        # Obtener los top k_candidates índices de BM25 filtrados
        top_bm25_idx = sorted(range(len(filtered_bm25_scores)), key=lambda i: filtered_bm25_scores[i], reverse=True)[:k_candidates]
        # Quitar los penalizados
        top_bm25_idx = [i for i in top_bm25_idx if filtered_bm25_scores[i] > -9000.0]

        if debug_data is not None:
            debug_data["bm25"] = {
                "tokens_query": tokenized_query,
                "top_scores": [
                    {"score": round(bm25_scores[i], 4), "fuente": bm25_metas[i].get("source", "?")}
                    for i in top_bm25_idx[:5]
                ],
            }

        # 3. Fusión RRF (Reciprocal Rank Fusion)
        rrf_scores = {}

        for rank, (doc, meta) in enumerate(zip(sem_docs, sem_metas)):
            if doc not in rrf_scores:
                rrf_scores[doc] = {"meta": meta, "score": 0.0}
            rrf_scores[doc]["score"] += 1.0 / (60 + rank + 1)

        for rank, idx in enumerate(top_bm25_idx):
            doc = bm25_docs[idx]
            meta = bm25_metas[idx]
            if doc not in rrf_scores:
                rrf_scores[doc] = {"meta": meta, "score": 0.0}
            rrf_scores[doc]["score"] += 1.0 / (60 + rank + 1)

        sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1]["score"], reverse=True)
        retrieved = [(doc, data["meta"], data["score"]) for doc, data in sorted_rrf[:n_results]]

        if debug_data is not None:
            debug_data["rrf"] = {
                "n_docs_fusionados": len(sorted_rrf),
                "top_resultados": [
                    {"score_rrf": round(data["score"], 6), "fuente": data["meta"].get("source", "?")}
                    for _doc, data in sorted_rrf[:5]
                ],
            }

    forced_ids = []
    # Detect asignaturas and add their specific correlativa IDs
    asig_ids = detect_asignaturas_query(query)
    for stem, asig_n in asig_ids:
        forced_ids.append(f"{stem}_correlativa_{asig_n:02d}")

    # Detect year and add year summary IDs for all stems (or limited to career stems)
    anio = detect_anio_query(query)
    if anio is not None:
        career_stems = detect_career_stems(query)
        if career_stems:
            stems_to_use = career_stems
        else:
            stems_to_use = set(asig["stem"] for asig in _load_asignaturas_map())
        for stem in stems_to_use:
            forced_ids.append(f"{stem}_anio_{anio:02d}")

    if debug_data is not None:
        debug_data["forced_ids"] = forced_ids if forced_ids else []

    # Retrieve forced chunks, prioritize them
    for chunk_id in forced_ids:
        try:
            forced = collection.get(ids=[chunk_id])
        except Exception:
            forced = None
        if forced and forced.get("ids"):
            forced_doc = forced["documents"][0]
            forced_meta = forced["metadatas"][0]
            already_present = any(doc == forced_doc for doc, _meta, _dist in retrieved)
            if already_present:
                retrieved = [(forced_doc, forced_meta, None)] + [
                    r for r in retrieved if r[0] != forced_doc
                ]
            else:
                retrieved = [(forced_doc, forced_meta, None)] + retrieved[: max(len(retrieved) - 1, 0)]

    # 4. Reranking (opcional)
    if reranker_model:
        reranker = get_reranker(reranker_model)
        pairs = [[query, doc] for doc, meta, dist in retrieved]
        scores = reranker.predict(pairs)

        scored_retrieved = list(zip(retrieved, scores))
        scored_retrieved.sort(key=lambda x: x[1], reverse=True)

        if debug_data is not None:
            debug_data["reranking"] = [
                {
                    "rank": i + 1,
                    "score_reranker": round(float(s), 4),
                    "fuente": item[0][1].get("source", "?"),
                    "texto_preview": item[0][0][:80].replace("\n", " "),
                }
                for i, (item, s) in enumerate(sorted(
                    zip(scored_retrieved, [x[1] for x in scored_retrieved]),
                    key=lambda x: x[1], reverse=True
                )[:5])
            ]

        retrieved = [(item[0][0], item[0][1], float(item[1])) for item in scored_retrieved]

    return retrieved


def retrieve_with_multi_query(
    queries: list,
    db_path: str,
    collection_name: str,
    model_name: str,
    n_results: int,
    use_hybrid: bool = False,
    reranker_model: str = None,
    debug_data: dict = None,
) -> list:
    """Fusiona los resultados de múltiples queries usando Reciprocal Rank Fusion (RRF).

    La primera query de la lista recibe mayor peso (1.5x), asumiendo que es la
    pregunta original o la principal. Las demás son variantes o sub-preguntas.
    Si se pasa un reranker, se aplica al final usando la primera query.
    Si se pasa `debug_data` (dict vacío), lo rellena con telemetría de cada paso.
    """
    all_docs: dict = {}  # doc_text -> {meta, score}
    per_query_counts = []

    for q_idx, query in enumerate(queries):
        weight = 1.5 if q_idx == 0 else 1.0
        results = retrieve_chunks(
            query, db_path, collection_name, model_name, n_results,
            use_hybrid=use_hybrid, reranker_model=None
        )
        per_query_counts.append({"query": query[:80], "n_resultados": len(results), "peso": weight})
        for rank, (doc, meta, _dist) in enumerate(results):
            if doc not in all_docs:
                all_docs[doc] = {"meta": meta, "score": 0.0}
            all_docs[doc]["score"] += weight / (60 + rank + 1)

    sorted_docs = sorted(all_docs.items(), key=lambda x: x[1]["score"], reverse=True)
    retrieved = [(doc, data["meta"], data["score"]) for doc, data in sorted_docs[:n_results]]

    if debug_data is not None:
        debug_data["multi_query_rrf"] = {
            "n_queries": len(queries),
            "per_query": per_query_counts,
            "n_docs_fusionados": len(sorted_docs),
            "top_resultados": [
                {"score_rrf": round(data["score"], 6), "fuente": data["meta"].get("source", "?")}
                for _doc, data in sorted_docs[:5]
            ],
        }

    # Reranking unificado al final con la primera query (la original)
    if reranker_model and retrieved:
        reranker = get_reranker(reranker_model)
        pairs = [[queries[0], doc] for doc, _meta, _dist in retrieved]
        scores = reranker.predict(pairs)
        scored = list(zip(retrieved, scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        if debug_data is not None:
            debug_data["reranking"] = [
                {
                    "rank": i + 1,
                    "score_reranker": round(float(s), 4),
                    "fuente": item[1].get("source", "?"),
                    "texto_preview": item[0][:80].replace("\n", " "),
                }
                for i, (item, s) in enumerate(scored[:5])
            ]

        retrieved = [(item[0][0], item[0][1], float(item[1])) for item in scored]

    return retrieved


def generate_multi_queries(
    question: str,
    base_url: str,
    model: str,
    timeout: int,
    n_queries: int = 5,
    api_key: str = None,
) -> list:
    """Genera hasta `n_queries` preguntas alternativas para cubrir distintos
    ángulos de búsqueda sobre el mismo tema."""
    prompt = (
        f"Generá {n_queries} formas diferentes de preguntar lo mismo sobre planes de estudios de la UNLaR.\n"
        "Cada variación debe buscar EXACTAMENTE la misma información pero con distintas palabras.\n"
        "NO cambies el tema. NO agregues preguntas sobre cosas que no se preguntan en la original.\n"
        "Devolvé SOLO las preguntas, una por línea, sin numeración ni explicaciones.\n\n"
        "EJEMPLO:\n"
        "PREGUNTA ORIGINAL: ¿Qué correlativas tiene Cálculo Numérico en Ing. en Sistemas?\n"
        "VARIACIONES:\n"
        "¿Cuáles son las materias previas de Cálculo Numérico en Ingeniería en Sistemas?\n"
        "¿Qué materias necesito tener aprobadas o regulares para cursar Cálculo Numérico?\n"
        "requisitos y correlatividades de Cálculo Numérico plan 2024 sistemas\n"
        "materias correlativas para rendir Cálculo Numérico ingeniería sistemas UNLaR\n"
        "¿Qué asignaturas son prerequisito de Cálculo Numérico en el plan de estudios?\n\n"
        f"PREGUNTA ORIGINAL: {question}\n\nVARIACIONES:"
    )
    messages = [
        {"role": "system", "content": "Generás variaciones de preguntas para mejorar la búsqueda en documentos académicos universitarios."},
        {"role": "user", "content": prompt},
    ]
    raw = call_lmstudio_chat(
        messages, base_url=base_url, model=model,
        temperature=0.6, timeout=timeout, max_tokens=350, api_key=api_key
    )
    queries = []
    for line in raw.strip().splitlines():
        line = re.sub(r"^[\d]+[.)\-]\s*|^[-•*]\s*", "", line.strip()).strip()
        if line and len(line) > 10:
            queries.append(line)
    return queries[:n_queries]



def generate_hyde_document(
    question: str,
    base_url: str,
    model: str,
    timeout: int,
    api_key: str = None,
) -> str:
    """Genera un documento hipotético de respuesta (HyDE)."""
    prompt = (
        "Escribí un párrafo breve (2-3 oraciones) que podría ser la respuesta a la siguiente "
        "pregunta sobre los planes de estudios de la UNLaR.\n"
        "Usá terminología académica: correlativas, régimen de cursado, asignatura, aprobadas, "
        "regularizadas, anual, cuatrimestral, etc.\n"
        "Inventá una respuesta plausible y técnica. NO digas que no sabés.\n\n"
        "EJEMPLO:\n"
        "PREGUNTA: ¿Qué correlativas tiene Análisis Matemático II en Sistemas?\n"
        "RESPUESTA HIPOTÉTICA: Para cursar Análisis Matemático II (asignatura N° 6) de Ingeniería "
        "en Sistemas de Información, Plan 2024, se requiere tener regular Análisis Matemático I (N° 2). "
        "Para rendir el examen final se necesita tener aprobada Análisis Matemático I. "
        "El régimen de cursado es cuatrimestral (2C).\n\n"
        f"PREGUNTA: {question}\n\nRESPUESTA HIPOTÉTICA:"
    )
    messages = [
        {"role": "system", "content": "Generás fragmentos de texto estilo ordenanza universitaria sobre planes de estudios."},
        {"role": "user", "content": prompt},
    ]
    return call_lmstudio_chat(
        messages, base_url=base_url, model=model,
        temperature=0.3, timeout=timeout, max_tokens=250, api_key=api_key
    ).strip()



def decompose_query(
    question: str,
    base_url: str,
    model: str,
    timeout: int,
    api_key: str = None,
) -> list:
    """Descompone una pregunta compleja en sub-preguntas simples e independientes."""
    prompt = (
        "Analizá la siguiente pregunta sobre planes de estudios de la UNLaR.\n"
        "Si pregunta por UNA SOLA materia o un solo dato, devolvé la pregunta original tal cual.\n"
        "Si pregunta por VARIAS materias o varios datos distintos, separá en una sub-pregunta por cada materia o dato pedido (máximo 4).\n"
        "Las sub-preguntas deben ser sobre correlativas, materias, régimen de cursado, etc. NO inventes preguntas sobre temas que no se mencionan.\n"
        "Devolvé SOLO las preguntas, una por línea, sin numeración ni explicaciones.\n\n"
        "EJEMPLO 1 (simple, devolver sin cambios):\n"
        "PREGUNTA: ¿Qué correlativas tiene Cálculo Numérico?\n"
        "SUB-PREGUNTAS:\n"
        "¿Qué correlativas tiene Cálculo Numérico?\n\n"
        "EJEMPLO 2 (compleja, separar por materia):\n"
        "PREGUNTA: ¿Qué correlativas tienen Análisis Matemático II y Cálculo Numérico de Ing. en Sistemas?\n"
        "SUB-PREGUNTAS:\n"
        "¿Qué correlativas tiene Análisis Matemático II de Ingeniería en Sistemas?\n"
        "¿Qué correlativas tiene Cálculo Numérico de Ingeniería en Sistemas?\n\n"
        f"PREGUNTA: {question}\n\nSUB-PREGUNTAS:"
    )
    messages = [
        {"role": "system", "content": "Desponés preguntas complejas sobre planes de estudios universitarios en sub-preguntas simples."},
        {"role": "user", "content": prompt},
    ]
    raw = call_lmstudio_chat(
        messages, base_url=base_url, model=model,
        temperature=0.1, timeout=timeout, max_tokens=300, api_key=api_key
    )
    sub_queries = []
    for line in raw.strip().splitlines():
        line = re.sub(r"^[\d]+[.)\-]\s*|^[-•*]\s*", "", line.strip()).strip()
        if line and len(line) > 10:
            sub_queries.append(line)
    return sub_queries[:4] if sub_queries else [question]



def build_rag_user_message(query: str, retrieved: list) -> str:
    """Arma el mensaje de usuario con el contexto recuperado + la pregunta,
    para mandarlo como el último mensaje 'user' del chat."""
    context_blocks = []
    for i, (doc, meta, _dist) in enumerate(retrieved, start=1):
        fuente = meta.get("source", "desconocido")
        context_blocks.append(f"--- FRAGMENTO {i} (Fuente: {fuente}) ---\n{doc}")
    context = "\n\n".join(context_blocks)
    return (
        f"CONTEXTO RECUPERADO DE LA BASE DE DATOS:\n\n{context}\n\n"
        f"PREGUNTA DEL USUARIO: {query}\n\n"
        f"Respondé a la PREGUNTA DEL USUARIO utilizando únicamente la información relevante del CONTEXTO RECUPERADO anterior. Cita el número de fragmento."
    )


def call_lmstudio_chat(
    messages: list,
    base_url: str,
    model: str,
    temperature: float,
    timeout: int,
    max_tokens: int | None = None,
    api_key: str | None = None,
) -> str:
    """Llamada genérica al endpoint /v1/chat/completions (OpenAI-compatible).

    Funciona con LMStudio (local) y con cualquier proveedor que implemente la
    API de OpenAI: Groq, OpenRouter, Together AI, OpenAI, etc.
    Si `api_key` es None o string vacío, no se manda header de autorización
    (compatible con LMStudio local que no requiere auth).
    """
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise LMStudioError(
            f"No se pudo conectar al proveedor LLM en {url}.\n"
            f"Verificá la URL, tu conexión a internet, y que el servidor esté activo."
        )
    except requests.exceptions.ReadTimeout:
        raise LMStudioError(
            f"El proveedor LLM no respondió en {timeout}s.\n"
            f"Probá aumentar el timeout o usar un modelo más rápido."
        )
    except requests.exceptions.HTTPError as e:
        status = getattr(response, 'status_code', '?')
        body = getattr(response, 'text', '')[:500]
        if status == 401:
            raise LMStudioError(
                f"API Key inválida o no autorizada (HTTP 401).\n"
                f"Verificá la API Key ingresada en el sidebar."
            )
        elif status == 429:
            raise LMStudioError(
                f"Rate limit alcanzado (HTTP 429). Esperá un momento y volvé a intentar.\n"
                f"En planes gratuitos, el límite de requests por minuto es bajo."
            )
        else:
            raise LMStudioError(f"Error HTTP {status}: {e}\nRespuesta: {body}")

    data = response.json()
    return data["choices"][0]["message"]["content"]



def _is_simple_chitchat(question: str) -> bool:
    """
    Detecta de forma rápida si la pregunta es un saludo, agradecimiento, despedida
    o una pregunta meta sobre las capacidades del asistente, para evitar llamar
    innecesariamente al LLM router y reducir latencia o falsos enrutamientos.
    """
    q = _normalize_text(question.strip())
    q = re.sub(r"[^\w\s]", "", q).strip()
    
    chitchat_phrases = {
        "hola", "buenas", "buen dia", "buenas tardes", "buenas noches",
        "gracias", "muchas gracias", "muchisimas gracias", "ok", "listo",
        "entendido", "genial", "perfecto", "excelente", "chau", "adios",
        "hasta luego", "nos vemos", "gracias chau", "ok gracias",
        "quien sos", "que sos", "como funcionas", "ayuda", "help",
        "que podes hacer", "que haces", "que podes responder", "que respondes",
        "que tipo de preguntas podes responderme", "que tipo de preguntas podes responder",
        "que preguntas te puedo hacer", "que preguntas puedo hacer",
        "que podes contestar", "que contestas"
    }
    if q in chitchat_phrases:
        return True
        
    # Coincidencias parciales por regex para variaciones
    meta_patterns = [
        r"^que (?:tipo de )?preguntas (?:puedo|podes|te puedo|me podes)",
        r"^que (?:puedes|podes|haces|hace) (?:hacer|responder|contestar)",
        r"^ayuda\b",
        r"^quien (?:eres|sos)\b"
    ]
    for pattern in meta_patterns:
        if re.search(pattern, q):
            return True
            
    return False


VALID_STRATEGIES = {"direct", "decompose", "multi_query", "hyde"}

def _parse_router_response(raw: str, fallback_question: str) -> tuple[bool, str, str, str]:
    """Parsea la respuesta del router con resiliencia a variaciones de formato
    y limpieza de markdown/comillas.
    Devuelve: (necesita_retrieval, pregunta, strategy, reason)
    """
    m_retrieval = re.search(r"RETRIEVAL:\s*(SI|SÍ|NO)", raw, re.IGNORECASE)
    m_pregunta = re.search(r"(?:PREGUNTA|REFORMULACION|CONSULTA|PREGUNTA REFORMULADA):\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    m_strategy = re.search(r"ESTRATEGIA:\s*(\w+)", raw, re.IGNORECASE)
    m_reason = re.search(r"RAZON:\s*(.+)", raw, re.IGNORECASE)

    # Parse strategy
    strategy = "direct"
    if m_strategy:
        candidate = m_strategy.group(1).strip().lower()
        if candidate in VALID_STRATEGIES:
            strategy = candidate

    # Parse reason
    reason = ""
    if m_reason:
        reason = m_reason.group(1).strip().splitlines()[0].strip()

    # Si la razón indica explícitamente que la pregunta está fuera del dominio, forzar RETRIEVAL: NO
    reason_lower = reason.lower()
    is_out_of_domain = any(
        kw in reason_lower
        for kw in ["ajena", "ajeno", "fuera del dominio", "no pertenece", "no relacionado", "no requiere buscar"]
    )

    if not m_retrieval:
        return (False if is_out_of_domain else True), fallback_question, strategy, reason

    necesita_retrieval = m_retrieval.group(1).upper().startswith("S")
    if is_out_of_domain:
        necesita_retrieval = False

    if not necesita_retrieval:
        return False, fallback_question, "direct", reason or "no necesita buscar"

    if m_pregunta:
        pregunta = m_pregunta.group(1).strip().splitlines()[0].strip()
        # Limpiar comillas y markdown (*, `, ", ')
        pregunta = re.sub(r'^["\'«`*]+|["\'»`*]+$', "", pregunta).strip()
        # Limpiar aclaraciones entre paréntesis al final (ej. "(reformulada)")
        pregunta = re.sub(r"\s*\((?:reformulada|no aplica|autocontenida)\)$", "", pregunta, flags=re.IGNORECASE).strip()
        if pregunta and pregunta.lower() != "(no aplica)":
            return True, pregunta, strategy, reason

    return True, fallback_question, strategy, reason


def route_query(
    history: list,
    question: str,
    base_url: str,
    model: str,
    timeout: int,
    api_key: str = None,
) -> tuple[bool, str, str, str]:
    """
    Decide si la pregunta nueva necesita retrieval en ChromaDB y, si lo
    necesita, la reformula como pregunta autocontenida usando el historial.
    También decide la estrategia de búsqueda óptima (Adaptive RAG).
    """
    q_low = question.lower()
    # Forzado de búsqueda mediante límites de palabras clave inequívocas del dominio RAG
    # Evita falsos positivos en saludos o consultas informales ("hola, quería consultar...")
    import re as _re
    force_patterns = [
        r"\bdocumento[s]?\b",
        r"\bbase\s+de\s+datos\b",
        r"\bbd\b",
        r"\barchivo[s]?\b",
        r"\bpdf[s]?\b",
        r"\bbuscar\s+en\s+los\s+documentos\b",
        r"\bconsultar\s+en\s+los\s+documentos\b",
        r"\bconsultar\s+en\s+la\s+base\b"
    ]
    force_retrieval = any(_re.search(pattern, q_low) for pattern in force_patterns)

    if _is_simple_chitchat(question) and not force_retrieval:
        return False, question, "direct", "saludo o chitchat"

    if not history:
        router_messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
        router_messages.append({"role": "user", "content": f"PREGUNTA NUEVA: {question}"})
        raw = call_lmstudio_chat(
            router_messages, base_url=base_url, model=model,
            temperature=0.0, timeout=timeout, max_tokens=200, api_key=api_key,
        )
        needs, q, strat, reason = _parse_router_response(raw, fallback_question=question)
        if force_retrieval:
            needs = True
            reason = f"Forzado por el usuario ({reason or 'consulta explícita'})"
        if not needs and not _is_simple_chitchat(question):
            return True, question, strat, reason
        return needs, q, strat, reason

    router_messages = [{"role": "system", "content": ROUTER_SYSTEM_PROMPT}]
    router_messages.extend(history)
    router_messages.append({"role": "user", "content": f"PREGUNTA NUEVA: {question}"})

    raw = call_lmstudio_chat(
        router_messages, base_url=base_url, model=model,
        temperature=0.0, timeout=timeout, max_tokens=200, api_key=api_key,
    )
    needs, q, strat, reason = _parse_router_response(raw, fallback_question=question)
    if force_retrieval:
        needs = True
        reason = f"Forzado por el usuario ({reason or 'consulta explícita'})"
    return needs, q, strat, reason