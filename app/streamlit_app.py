#!/usr/bin/env python3
"""
app/streamlit_app.py
---------------------
Interfaz web del asistente RAG sobre el plan de estudios de Ingeniería en
Sistemas de Información (UNLaR). Reusa la misma lógica de recuperación +
prompt + llamada a LMStudio que scripts/05_rag_query.py, pero en formato
chat con Streamlit.

Requisito previo: tener LMStudio corriendo con un modelo cargado y el
servidor local iniciado (por defecto en http://localhost:1234), y haber
corrido antes 03_build_vectordb.py para tener chroma_db/ generado.

Uso:
    streamlit run app/streamlit_app.py

Los parámetros de conexión (rutas, modelo, temperatura) están en la barra
lateral, con los mismos defaults que 05_rag_query.py, así no hace falta
tocar código para cambiarlos.
"""

import re

import streamlit as st

try:
    import chromadb
except ImportError:
    st.error("Falta chromadb. Instalalo con: pip install chromadb --break-system-packages")
    st.stop()

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("Falta sentence-transformers. Instalalo con: pip install sentence-transformers --break-system-packages")
    st.stop()

try:
    import requests
except ImportError:
    st.error("Falta requests. Instalalo con: pip install requests --break-system-packages")
    st.stop()


SYSTEM_PROMPT = (
    "Sos un asistente que responde preguntas sobre el plan de estudios de "
    "Ingeniería en Sistemas de Información de la UNLaR, usando la información "
    "de los fragmentos de contexto que se te proveen.\n\n"
    "Cómo responder:\n"
    "- Basate en los fragmentos relevantes, aunque no todos coincidan entre sí "
    "o algunos no vengan al caso. Si uno solo responde la pregunta, alcanza.\n"
    "- Dá la respuesta completa: si el fragmento tiene varios datos relacionados "
    "con la pregunta (por ejemplo, requisitos para cursar Y para rendir el "
    "final), incluí todos, no solo el primero que encuentres.\n"
    "- Citá de qué fragmento sacaste cada dato (ej. 'según el Fragmento 1...'), "
    "así el usuario puede verificarlo.\n"
    "- Si un fragmento viene de un escaneo con errores de OCR (texto cortado, "
    "tablas desordenadas, palabras sueltas), ignoralo sin mencionarlo como "
    "motivo de duda sobre el resto.\n"
    "- Si de verdad ningún fragmento contiene la respuesta, decilo "
    "directamente en vez de inventar información.\n"
    "- Respondé en español, de forma natural y completa, sin ser telegráfico "
    "ni tampoco extenderte de más."
)


# ---------------------------------------------------------------------------
# Recursos cacheados: el modelo de embeddings y la colección de ChromaDB no
# cambian entre preguntas, así que se cargan una sola vez por sesión/params.
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)


@st.cache_resource(show_spinner="Conectando a ChromaDB...")
def load_collection(db_path: str, collection_name: str):
    client = chromadb.PersistentClient(path=db_path)
    try:
        return client.get_collection(collection_name)
    except Exception as e:
        st.error(
            f"No se pudo abrir la colección '{collection_name}' en '{db_path}'.\n\n"
            f"¿Corriste 03_build_vectordb.py? Detalle: {e}"
        )
        st.stop()


ORDINAL_ANIO = {
    "primer": 1, "1er": 1, "1ro": 1,
    "segundo": 2, "2do": 2,
    "tercer": 3, "tercero": 3, "3er": 3,
    "cuarto": 4, "4to": 4,
    "quinto": 5, "5to": 5,
}


def detect_anio_query(query: str) -> int | None:
    """
    Si la pregunta menciona explícitamente un año de la carrera (ej. "primer año",
    "2do año", "año 3"), devuelve el número de año (1-5). Si no, devuelve None.
    Ver la explicación completa en scripts/05_rag_query.py (mismo mecanismo).
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


def retrieve_chunks(query: str, model: SentenceTransformer, collection, n_results: int):
    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    retrieved = list(zip(docs, metas, distances))

    anio = detect_anio_query(query)
    if anio is not None:
        chunk_id = f"correlativa_anio_{anio:02d}"
        try:
            forced = collection.get(ids=[chunk_id])
        except Exception:
            forced = None
        if forced and forced.get("ids"):
            forced_doc = forced["documents"][0]
            forced_meta = forced["metadatas"][0]
            retrieved = [(forced_doc, forced_meta, None)] + [
                r for r in retrieved if r[0] != forced_doc
            ][: max(n_results - 1, 0)]

    return retrieved


def build_prompt(query: str, retrieved: list) -> str:
    context_blocks = []
    for i, (doc, meta, _dist) in enumerate(retrieved, start=1):
        fuente = meta.get("source", "desconocido")
        context_blocks.append(f"[Fragmento {i} - fuente: {fuente}]\n{doc}")
    context = "\n\n".join(context_blocks)
    return (
        f"Los fragmentos están ordenados del más al menos relevante para la pregunta "
        f"(el Fragmento 1 es el que mejor coincide).\n\n"
        f"CONTEXTO:\n{context}\n\n"
        f"PREGUNTA: {query}\n\n"
        f"Respondé la pregunta basándote en el fragmento (o fragmentos) que "
        f"realmente contengan la información, ignorando los que sean ruido "
        f"o no vengan al caso."
    )


def call_lmstudio(prompt: str, base_url: str, model: str, temperature: float) -> str:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    try:
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"No se pudo conectar a LMStudio en {url}.\n\n"
            f"Revisá que LMStudio esté abierto, con un modelo cargado, y el "
            f"servidor local iniciado (pestaña 'Developer' o 'Local Server')."
        )
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f"LMStudio devolvió un error: {e}\nRespuesta: {response.text[:500]}")

    data = response.json()
    return data["choices"][0]["message"]["content"]


def fuente_label(meta: dict) -> str:
    fuente = meta.get("source", "?")
    pagina = f" (página {meta['page']})" if "page" in meta else ""
    return f"{fuente}{pagina}"


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="RAG Currículum UNLaR", page_icon="🎓", layout="wide")

st.title("🎓 Asistente del plan de estudios — Ing. en Sistemas de Información (UNLaR)")
st.caption(
    "Preguntá sobre correlativas, régimen de cursado o cualquier dato del plan de estudios. "
    "Las respuestas se generan con RAG: se recuperan fragmentos relevantes de ChromaDB y se "
    "arma la respuesta con un LLM local corriendo en LMStudio."
)

with st.sidebar:
    st.header("⚙️ Configuración")

    st.subheader("Base vectorial")
    db_path = st.text_input("Carpeta de ChromaDB", value="./chroma_db")
    collection_name = st.text_input("Colección", value="curriculum")
    embedding_model_name = st.text_input(
        "Modelo de embeddings",
        value="paraphrase-multilingual-MiniLM-L12-v2",
        help="Tiene que ser el mismo modelo usado en 03_build_vectordb.py",
    )
    n_results = st.slider("Fragmentos a recuperar", min_value=1, max_value=10, value=3)

    st.subheader("LMStudio")
    lmstudio_url = st.text_input("URL del servidor", value="http://localhost:1234")
    lmstudio_model = st.text_input(
        "Nombre del modelo cargado",
        value="local-model",
        help="LMStudio suele aceptar cualquier string acá si hay un solo modelo cargado.",
    )
    temperature = st.slider(
        "Temperatura", min_value=0.0, max_value=1.0, value=0.35, step=0.05,
        help="Con modelos más capaces (13B+) 0.3-0.4 da respuestas más naturales sin perder "
             "precisión. Bajala a 0.1-0.2 si notás que empieza a divagar o inventar datos.",
    )

    st.subheader("Debug")
    show_prompt = st.checkbox("Mostrar el prompt completo enviado al LLM", value=False)

    if st.button("🗑️ Borrar historial de chat"):
        st.session_state.messages = []
        st.rerun()

# Carga de recursos (cacheados; solo se recalculan si cambian los parámetros)
embedding_model = load_embedding_model(embedding_model_name)
collection = load_collection(db_path, collection_name)

st.caption(f"📚 {collection.count()} documentos indexados en la colección '{collection_name}'.")

# Historial de chat en la sesión
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("retrieved"):
            with st.expander("📎 Fragmentos recuperados (fuentes)"):
                for i, (doc, meta, dist) in enumerate(msg["retrieved"], start=1):
                    dist_txt = f"{dist:.4f}" if dist is not None else "forzado (palabra clave de año)"
                    st.markdown(f"**[{i}] {fuente_label(meta)}** — distancia: `{dist_txt}`")
                    st.text(doc[:400] + ("..." if len(doc) > 400 else ""))

query = st.chat_input("Escribí tu pregunta sobre el plan de estudios...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Recuperando fragmentos relevantes de ChromaDB..."):
            retrieved = retrieve_chunks(query, embedding_model, collection, n_results)

        prompt = build_prompt(query, retrieved)

        if show_prompt:
            with st.expander("🔍 Prompt enviado al LLM", expanded=False):
                st.text(prompt)

        with st.spinner(f"Consultando LMStudio en {lmstudio_url}..."):
            try:
                answer = call_lmstudio(prompt, lmstudio_url, lmstudio_model, temperature)
            except RuntimeError as e:
                answer = f"⚠️ {e}"

        st.markdown(answer)

        with st.expander("📎 Fragmentos recuperados (fuentes)"):
            for i, (doc, meta, dist) in enumerate(retrieved, start=1):
                dist_txt = f"{dist:.4f}" if dist is not None else "forzado (palabra clave de año)"
                st.markdown(f"**[{i}] {fuente_label(meta)}** — distancia: `{dist_txt}`")
                st.text(doc[:400] + ("..." if len(doc) > 400 else ""))

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "retrieved": retrieved,
    })