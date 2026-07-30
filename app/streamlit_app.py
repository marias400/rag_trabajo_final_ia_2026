#!/usr/bin/env python3
"""
app/streamlit_app.py
---------------------
Interfaz web del asistente RAG sobre el plan de estudios de la UNLaR.

Reusa TODA la logica de rag_core.py (la misma que usa scripts/05_rag_query.py
--chat en consola): recuperacion en ChromaDB, el router conversacional que
decide en cada turno si hace falta volver a buscar en la base o si alcanza
con lo ya conversado, y las llamadas a LMStudio. Nada de eso esta duplicado
aca: si se arregla o mejora en rag_core.py, esta app lo hereda automaticamente.

Uso:
    streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import streamlit as st
import rag_core
import importlib
importlib.reload(rag_core)


N_HISTORY_TURNS = 6


@st.cache_resource(show_spinner="Cargando modelo de embeddings...")
def load_embedding_model(model_name: str):
    return rag_core.get_embedding_model(model_name)


@st.cache_resource(show_spinner="Conectando a ChromaDB...")
def load_collection(db_path: str, collection_name: str):
    try:
        return rag_core.get_chroma_collection(db_path, collection_name)
    except Exception as e:
        st.error(
            f"No se pudo abrir la coleccion '{collection_name}' en '{db_path}'.\n\n"
            f"Corri 03_build_vectordb.py? Detalle: {e}"
        )
        st.stop()


def fuente_label(meta: dict) -> str:
    fuente = meta.get("source", "?")
    pagina = f" (pagina {meta['page']})" if "page" in meta else ""
    return f"{fuente}{pagina}"


def render_fragmentos(retrieved: list, use_hybrid: bool = False, use_reranker: bool = False,
                      use_multi_query: bool = False, use_decompose: bool = False) -> None:
    with st.expander("📎 Fragmentos recuperados (fuentes)"):
        for i, (doc, meta, dist) in enumerate(retrieved, start=1):
            if dist is None:
                dist_txt = "forzado (palabra clave)"
            elif use_reranker:
                dist_txt = f"score: {dist:.4f} (reranker)"
            elif use_multi_query or use_decompose:
                dist_txt = f"score: {dist:.4f} (rrf multi-query)"
            elif use_hybrid:
                dist_txt = f"score: {dist:.4f} (rrf)"
            else:
                dist_txt = f"distancia: {dist:.4f}"
            st.markdown(f"**[{i}] {fuente_label(meta)}** — {dist_txt}")
            st.text(doc[:400] + ("..." if len(doc) > 400 else ""))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="RAG Curriculum UNLaR", page_icon="🎓", layout="wide")

st.title("🎓 Asistente de Planes de Estudios (UNLaR)")
st.caption(
    "Pregunta sobre correlativas, regimen de cursado o cualquier dato de los planes de estudio "
    "(Ing. en Sistemas, Lic. en Sistemas, Ing. Mecatronica). Las respuestas se generan con RAG: "
    "un router decide en cada turno si hace falta volver a buscar en ChromaDB o si alcanza con lo ya conversado, "
    "y un LLM local en LMStudio arma la respuesta final."
)

with st.sidebar:
    st.header("⚙️ Configuracion")

    st.subheader("Base vectorial")
    db_path = st.text_input("Carpeta de ChromaDB", value="./chroma_db")
    collection_name = st.text_input("Coleccion", value="curriculum")
    embedding_model_name = st.text_input(
        "Modelo de embeddings",
        value="paraphrase-multilingual-MiniLM-L12-v2",
        help="Tiene que ser el mismo modelo usado en 03_build_vectordb.py",
    )
    n_results = st.slider("Fragmentos a recuperar", min_value=1, max_value=10, value=5)

    st.subheader("Retrieval Avanzado")
    use_hybrid = st.checkbox("🔀 Busqueda Hibrida (BM25 + Semantica + RRF)", value=True,
                             help="Combina coincidencia exacta de palabras con busqueda semantica.")
    use_reranker = st.checkbox("🏆 Activar Reranker Local", value=True,
                               help="Reordena los fragmentos recuperados con un modelo Cross-Encoder.")
    reranker_model = st.text_input("Modelo Reranker", value="BAAI/bge-reranker-v2-m3") if use_reranker else None

    st.subheader("Query Enhancement")
    use_agent = st.toggle(
        "🤖 Modo Agente (Automático)", value=True,
        help="El sistema analiza cada pregunta y decide automáticamente qué técnica de búsqueda usar."
    )

    if use_agent:
        st.caption("El agente decide la mejor estrategia para cada pregunta.")
        # En modo agente los checkboxes se ignoran
        use_multi_query = False
        use_hyde = False
        use_decompose = False
    else:
        st.caption("Seleccioná manualmente las técnicas. Se pueden combinar.")
        use_multi_query = st.checkbox(
            "🔀 Multi-Query (5 variantes)", value=False,
            help="Genera hasta 5 preguntas alternativas y fusiona los resultados con RRF."
        )
        use_hyde = st.checkbox(
            "💡 HyDE (Hypothetical Document Embeddings)", value=False,
            help="Genera un parrafo hipotetico de respuesta y lo embeddea en lugar de la pregunta."
        )
        use_decompose = st.checkbox(
            "🔍 Query Decomposition", value=False,
            help="Descompone preguntas complejas en sub-preguntas simples."
        )
    
    st.subheader("LMStudio")
    lmstudio_url = st.text_input("URL del servidor", value="http://localhost:1234")
    
    # Intentar obtener modelos cargados
    loaded_models = []
    try:
        import requests
        resp = requests.get(f"{lmstudio_url.rstrip('/')}/v1/models", timeout=2)
        if resp.status_code == 200:
            data = resp.json()
            if "data" in data:
                loaded_models = [m["id"] for m in data["data"] if "embed" not in m["id"].lower()]
    except Exception:
        pass

    if loaded_models:
        lmstudio_model = st.selectbox(
            "Modelo cargado",
            options=loaded_models + ["local-model"],
            index=0,
            help="Selecciona automáticamente el modelo de la lista de los cargados en LMStudio.",
        )
    else:
        lmstudio_model = st.text_input(
            "Nombre del modelo cargado",
            value="local-model",
            help="Usa 'local-model' para apuntar al modelo activo, o escribí el ID explícito.",
        )
    temperature = st.slider(
        "Temperatura", min_value=0.0, max_value=1.0, value=0.20, step=0.05,
        help="Para llama-3.2-3b-instruct se recomienda 0.20.",
    )
    timeout = st.number_input(
        "Timeout respuesta final (s)", min_value=10, max_value=600, value=90, step=10,
    )
    router_timeout = st.number_input(
        "Timeout llamadas cortas (s)", min_value=5, max_value=300, value=30, step=5,
        help="Timeout para llamadas cortas al LLM: router, multi-query, HyDE, decompose.",
    )

    st.subheader("Debug")
    show_prompt = st.checkbox("Mostrar prompt y decision del router", value=False)

    if st.button("🗑️ Borrar historial de chat"):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

# Carga de recursos
embedding_model = load_embedding_model(embedding_model_name)
collection = load_collection(db_path, collection_name)

st.caption(f"📚 {collection.count()} documentos indexados en la coleccion '{collection_name}'.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "history" not in st.session_state:
    st.session_state.history = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            # Mostrar la decisión de enrutamiento
            if show_prompt:
                if msg.get("pregunta_busqueda") and msg.get("necesita_retrieval"):
                    st.caption(f"🧭 router → RETRIEVAL: SI | pregunta reformulada: *{msg['pregunta_busqueda']}*")
                elif msg.get("necesita_retrieval") is False:
                    st.caption("🧭 router → RETRIEVAL: NO")

            # Mostrar la decisión del agente
            if msg.get("strategy") and msg.get("necesita_retrieval"):
                _strat_labels = {
                    "direct": "🎯 Búsqueda directa",
                    "decompose": "🔍 Query Decomposition",
                    "multi_query": "🔀 Multi-Query",
                    "hyde": "💡 HyDE",
                }
                st.caption(f"🤖 agente → {_strat_labels.get(msg['strategy'], msg['strategy'])} | {msg.get('reason', '')}")

            # Mostrar mejoras de query de forma visible si existieron
            if msg.get("sub_queries"):
                with st.expander(f"🔍 Query Decomposition ({len(msg['sub_queries'])} sub-preguntas)", expanded=False):
                    for j, q in enumerate(msg["sub_queries"], start=1):
                        st.markdown(f"**{j}.** {q}")
            if msg.get("variants"):
                with st.expander(f"🔀 Multi-Query ({len(msg['variants'])} variantes)", expanded=False):
                    for j, q in enumerate(msg["variants"], start=1):
                        prefix = "**[original]**" if j == 1 else f"**{j}.**"
                        st.markdown(f"{prefix} {q}")
            if msg.get("hyde_doc"):
                with st.expander("💡 HyDE: documento hipotetico generado", expanded=False):
                    st.text(msg["hyde_doc"])

            if show_prompt and msg.get("current_message"):
                with st.expander("🔍 Prompt enviado al LLM", expanded=False):
                    st.text(msg["current_message"])

            if msg.get("retrieved"):
                render_fragmentos(
                    msg["retrieved"],
                    msg.get("use_hybrid", False),
                    msg.get("use_reranker", False),
                    msg.get("use_multi_query", False),
                    msg.get("use_decompose", False),
                )

question = st.chat_input("Escribi tu pregunta sobre el plan de estudios...")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    recent_history = st.session_state.history[-2 * N_HISTORY_TURNS:]

    with st.chat_message("assistant"):
        retrieved = None
        answer = ""
        hyde_doc = None
        sub_queries = None
        variants = None
        current_message = ""
        try:
            with st.spinner("Analizando pregunta y decidiendo estrategia..."):
                necesita_retrieval, pregunta_busqueda, strategy, reason = rag_core.route_query(
                    recent_history, question, lmstudio_url, lmstudio_model, router_timeout
                )

            # Si modo agente, aplicar la estrategia decidida por el router
            if use_agent and necesita_retrieval:
                use_decompose = (strategy == "decompose")
                use_multi_query = (strategy == "multi_query")
                use_hyde = (strategy == "hyde")

            if show_prompt:
                st.caption(
                    f"🧭 router → RETRIEVAL: {'SI' if necesita_retrieval else 'NO'}"
                    + (f" | pregunta reformulada: *{pregunta_busqueda}*" if necesita_retrieval else "")
                )
            if use_agent and necesita_retrieval:
                strategy_labels = {
                    "direct": "🎯 Búsqueda directa",
                    "decompose": "🔍 Query Decomposition",
                    "multi_query": "🔀 Multi-Query",
                    "hyde": "💡 HyDE",
                }
                st.caption(f"🤖 agente → {strategy_labels.get(strategy, strategy)} | {reason}")

            if necesita_retrieval:
                all_queries = [pregunta_busqueda]

                if use_decompose:
                    with st.spinner("Descomponiendo pregunta en sub-preguntas..."):
                        try:
                            sub_queries = rag_core.decompose_query(
                                pregunta_busqueda, lmstudio_url, lmstudio_model, router_timeout
                            )
                        except rag_core.LMStudioError:
                            sub_queries = []
                    if sub_queries:
                        all_queries.extend(sub_queries)
                        with st.expander(f"🔍 Query Decomposition ({len(sub_queries)} sub-preguntas)", expanded=False):
                            for j, q in enumerate(sub_queries, 1):
                                st.markdown(f"**{j}.** {q}")

                if use_multi_query:
                    with st.spinner("Generando preguntas alternativas (Multi-Query)..."):
                        try:
                            variants_alt = rag_core.generate_multi_queries(
                                pregunta_busqueda, lmstudio_url, lmstudio_model, router_timeout
                            )
                        except rag_core.LMStudioError:
                            variants_alt = []
                    if variants_alt:
                        all_queries.extend(variants_alt)
                        variants = [pregunta_busqueda] + variants_alt
                        with st.expander(f"🔀 Multi-Query ({len(variants)} variantes)", expanded=False):
                            for j, q in enumerate(variants, 1):
                                prefix = "**[original]**" if j == 1 else f"**{j}.**"
                                st.markdown(f"{prefix} {q}")

                if use_hyde:
                    with st.spinner("Generando documento hipotetico (HyDE)..."):
                        try:
                            hyde_doc = rag_core.generate_hyde_document(
                                pregunta_busqueda, lmstudio_url, lmstudio_model, router_timeout
                            )
                            all_queries.append(hyde_doc)
                        except rag_core.LMStudioError:
                            hyde_doc = None
                    if hyde_doc:
                        with st.expander("💡 HyDE: documento hipotetico generado", expanded=False):
                            st.text(hyde_doc)

                with st.spinner(f"Recuperando fragmentos relevantes ({len(all_queries)} queries)..."):
                    if len(all_queries) > 1:
                        retrieved = rag_core.retrieve_with_multi_query(
                            all_queries, db_path, collection_name, embedding_model_name, n_results,
                            use_hybrid=use_hybrid, reranker_model=reranker_model
                        )
                    else:
                        retrieved = rag_core.retrieve_chunks(
                            pregunta_busqueda, db_path, collection_name, embedding_model_name, n_results,
                            use_hybrid=use_hybrid, reranker_model=reranker_model
                        )

                current_message = rag_core.build_rag_user_message(pregunta_busqueda, retrieved)
            else:
                current_message = question

            if show_prompt and necesita_retrieval:
                with st.expander("🔍 Prompt enviado al LLM", expanded=False):
                    st.text(current_message)

            messages = [{"role": "system", "content": rag_core.SYSTEM_PROMPT}]
            messages.extend(recent_history)
            messages.append({"role": "user", "content": current_message})

            with st.spinner(f"Consultando LMStudio en {lmstudio_url}..."):
                answer = rag_core.call_lmstudio_chat(
                    messages, lmstudio_url, lmstudio_model, temperature, timeout
                )
        except rag_core.LMStudioError as e:
            st.error(str(e))
            st.stop()

        st.markdown(answer)
        if retrieved:
            render_fragmentos(retrieved, use_hybrid, use_reranker, use_multi_query, use_decompose)

    st.session_state.history.append({"role": "user", "content": question})
    st.session_state.history.append({"role": "assistant", "content": answer})

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "retrieved": retrieved,
        "use_hybrid": use_hybrid,
        "use_reranker": use_reranker,
        "use_multi_query": use_multi_query,
        "use_decompose": use_decompose,
        "necesita_retrieval": necesita_retrieval,
        "pregunta_busqueda": pregunta_busqueda,
        "hyde_doc": hyde_doc,
        "sub_queries": sub_queries,
        "variants": variants,
        "current_message": current_message if necesita_retrieval else None,
        "strategy": strategy if necesita_retrieval else None,
        "reason": reason if necesita_retrieval else None,
    })
