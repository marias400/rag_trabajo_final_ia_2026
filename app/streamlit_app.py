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
    n = len(retrieved)
    with st.expander(f"📎 Fragmentos recuperados — {n} resultado{'s' if n != 1 else ''}", expanded=False):
        for i, (doc, meta, dist) in enumerate(retrieved, start=1):
            # --- Badge de método ---
            if dist is None:
                badge_color = "#121A21"
                badge_border = "#2196F3"
                badge_icon  = "📌"
                badge_text  = "Forzado (keyword)"
                score_txt   = ""
            elif use_reranker:
                badge_color = "#201A10"
                badge_border = "#f39c12"
                badge_icon  = "🏆"
                badge_text  = "Reranker"
                score_txt   = f"score {dist:.4f}"
            elif use_multi_query or use_decompose:
                badge_color = "#101F15"
                badge_border = "#27ae60"
                badge_icon  = "🔀"
                badge_text  = "RRF Multi-Query"
                score_txt   = f"score {dist:.4f}"
            elif use_hybrid:
                badge_color = "#1B1220"
                badge_border = "#8e44ad"
                badge_icon  = "⚖️"
                badge_text  = "RRF Híbrido"
                score_txt   = f"score {dist:.4f}"
            else:
                badge_color = "#1A1A1A"
                badge_border = "#7f8c8d"
                badge_icon  = "🔎"
                badge_text  = "Semántico"
                score_txt   = f"dist {dist:.4f}"

            fuente = meta.get("source", "?")
            pagina = f" · pág. {meta['page']}" if "page" in meta else ""

            st.markdown(
                f"""
<div style="
    border-left: 4px solid {badge_border};
    background: {badge_color};
    border-radius: 6px;
    padding: 10px 14px 10px 14px;
    margin-bottom: 10px;
    box-shadow: 0 2px 5px rgba(0,0,0,0.2);
">
<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
  <span style="font-weight:700; font-size:0.95em; color: #FFFFFF;">
    {badge_icon} <span style="color:{badge_border}; font-weight: 800;">[{i}]</span> &nbsp;{fuente}{pagina}
  </span>
  <span style="
    background:{badge_border}33;
    color:{badge_border};
    border:1px solid {badge_border}88;
    border-radius:12px;
    padding:2px 10px;
    font-size:0.78em;
    font-weight:700;
    white-space:nowrap;
  ">{badge_text}{(' · ' + score_txt) if score_txt else ''}</span>
</div>
<pre style="
    background: #0D0D0D;
    border: 1px solid #333333;
    border-radius:4px;
    padding:10px;
    font-size:0.85em;
    font-family: 'Courier New', Courier, monospace;
    white-space:pre-wrap;
    word-break:break-word;
    margin:0;
    color:#E0E0E0;
">{doc[:450].replace('<','&lt;').replace('>','&gt;')}{"..." if len(doc) > 450 else ""}</pre>
</div>
""",
                unsafe_allow_html=True,
            )



def render_debug_info(debug_data: dict) -> None:
    """Renderiza en un expander toda la telemetría recolectada del pipeline RAG."""
    with st.expander("📊 Telemetría del Pipeline RAG", expanded=False):
        mode = debug_data.get("mode", "?")
        if mode == "hibrida":
            st.markdown("**Modo:** 🔀 Búsqueda Híbrida (Semántica + BM25 + RRF)")
        elif mode == "solo_semantica":
            st.markdown("**Modo:** 🔎 Búsqueda Solo Semántica")

        # --- Búsqueda Semántica ---
        if "semantica" in debug_data:
            sem = debug_data["semantica"]
            with st.container(border=True):
                st.markdown("#### 🔎 Búsqueda Semántica (ChromaDB)")
                st.write(f"- Candidatos recuperados: **{sem['n_candidatos']}**")
                if sem.get("top_distancias"):
                    st.write("- Top-5 distancias coseno:", sem["top_distancias"])
                if sem.get("top_fuentes"):
                    st.write("- Top-5 fuentes:", sem["top_fuentes"])

        # --- BM25 ---
        if "bm25" in debug_data:
            bm = debug_data["bm25"]
            with st.container(border=True):
                st.markdown("#### 📝 Búsqueda Léxica (BM25 Okapi)")
                st.write("- Tokens de la query:", bm["tokens_query"])
                if bm.get("top_scores"):
                    import pandas as pd
                    df_bm25 = pd.DataFrame(bm["top_scores"])
                    df_bm25.columns = ["Score BM25", "Fuente"]
                    st.dataframe(df_bm25, use_container_width=True, hide_index=True)

        # --- RRF ---
        if "rrf" in debug_data:
            rrf = debug_data["rrf"]
            with st.container(border=True):
                st.markdown("#### ⚖️ Fusión RRF (Reciprocal Rank Fusion)")
                st.write(f"- Documentos únicos fusionados: **{rrf['n_docs_fusionados']}**")
                if rrf.get("top_resultados"):
                    import pandas as pd
                    df_rrf = pd.DataFrame(rrf["top_resultados"])
                    df_rrf.columns = ["Score RRF", "Fuente"]
                    st.dataframe(df_rrf, use_container_width=True, hide_index=True)

        # --- Multi-Query RRF ---
        if "multi_query_rrf" in debug_data:
            mq = debug_data["multi_query_rrf"]
            with st.container(border=True):
                st.markdown("#### 🔀 Multi-Query RRF")
                st.write(f"- Queries fusionadas: **{mq['n_queries']}**")
                st.write(f"- Documentos únicos: **{mq['n_docs_fusionados']}**")
                if mq.get("per_query"):
                    for entry in mq["per_query"]:
                        st.caption(f"• peso={entry['peso']} | {entry['n_resultados']} resultados | `{entry['query']}`")
                if mq.get("top_resultados"):
                    import pandas as pd
                    df_mq = pd.DataFrame(mq["top_resultados"])
                    df_mq.columns = ["Score RRF", "Fuente"]
                    st.dataframe(df_mq, use_container_width=True, hide_index=True)

        # --- Chunks Forzados ---
        if "forced_ids" in debug_data:
            fids = debug_data["forced_ids"]
            with st.container(border=True):
                st.markdown("#### 📌 Chunks Forzados (Asignaturas/Año detectados)")
                if fids:
                    for fid in fids:
                        st.caption(f"• `{fid}`")
                else:
                    st.caption("Ninguno detectado.")

        # --- Reranking ---
        if "reranking" in debug_data:
            with st.container(border=True):
                st.markdown("#### 🏆 Reranking (Cross-Encoder)")
                import pandas as pd
                df_rr = pd.DataFrame(debug_data["reranking"])
                df_rr = df_rr.rename(columns={
                    "rank": "Rank",
                    "score_reranker": "Score Reranker",
                    "fuente": "Fuente",
                    "texto_preview": "Preview",
                })
                st.dataframe(df_rr, use_container_width=True, hide_index=True)


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
    

    # -------------------------------------------------------------------------
    # Proveedor LLM
    # -------------------------------------------------------------------------
    st.subheader("🤖 Proveedor LLM")

    PROVIDERS = {
        "🖥️ Local (LMStudio)": {
            "base_url": "http://localhost:1234",
            "requires_key": False,
            "models": [],  # se detectan dinámicamente
            "help": "Servidor local de LMStudio. No requiere API Key.",
        },
        "⚡ Groq (gratis)": {
            "base_url": "https://api.groq.com/openai",
            "requires_key": True,
            "models": [
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "llama3-70b-8192",
                "gemma2-9b-it",
                "mixtral-8x7b-32768",
            ],
            "help": "Groq ofrece un plan gratuito muy generoso. API Key en console.groq.com",
        },
        "🌐 OpenRouter (modelos gratis)": {
            "base_url": "https://openrouter.ai/api",
            "requires_key": True,
            "models": [
                "meta-llama/llama-3.1-8b-instruct:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "google/gemma-2-9b-it:free",
                "mistralai/mistral-7b-instruct:free",
                "qwen/qwen-2.5-72b-instruct:free",
                "deepseek/deepseek-r1-0528:free",
            ],
            "help": "OpenRouter da acceso a 200+ modelos. Los marcados :free no cuestan nada. API Key en openrouter.ai/keys",
        },
        "🧠 Cerebras (gratis)": {
            "base_url": "https://api.cerebras.ai/v1",
            "requires_key": True,
            "models": [
                "llama-3.3-70b",
                "llama3.1-8b",
                "qwen-3-32b",
            ],
            "help": "Cerebras tiene hardware propio (WSE) ultra rápido y plan gratuito. API Key en cloud.cerebras.ai",
        },
        "💎 Google Gemini (gratis)": {
            "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
            "requires_key": True,
            "models": [
                "gemini-2.0-flash",
                "gemini-2.5-flash",
                "gemini-1.5-flash",
                "gemini-1.5-flash-8b",
            ],
            "help": "Gemini Flash tiene tier gratuito generoso. API Key en aistudio.google.com",
        },
        "🔑 OpenAI (pago)": {
            "base_url": "https://api.openai.com",
            "requires_key": True,
            "models": [
                "gpt-4o-mini",
                "gpt-4o",
                "gpt-4.1-mini",
                "gpt-4.1-nano",
            ],
            "help": "API de OpenAI. Requiere créditos. gpt-4o-mini es el más barato.",
        },
        "🔧 URL personalizada": {
            "base_url": "",
            "requires_key": False,
            "models": [],
            "help": "Cualquier servidor compatible con la API de OpenAI (Ollama, vLLM, etc.).",
        },
    }

    provider_name = st.selectbox(
        "Proveedor",
        options=list(PROVIDERS.keys()),
        index=0,
        help="Elegí el proveedor de LLM a usar.",
    )
    provider = PROVIDERS[provider_name]

    # URL base
    if provider_name == "🔧 URL personalizada":
        lmstudio_url = st.text_input("URL base del servidor", value="http://localhost:1234")
    elif provider_name == "🖥️ Local (LMStudio)":
        lmstudio_url = st.text_input("URL del servidor LMStudio", value=provider["base_url"])
    else:
        lmstudio_url = provider["base_url"]
        st.caption(f"🔗 `{lmstudio_url}`")

    st.caption(provider["help"])

    # API Key
    api_key = None
    if provider["requires_key"]:
        key_input = st.text_input(
            "API Key",
            type="password",
            value=st.session_state.get("llm_api_key", ""),
            help="Se guarda solo en la sesión actual, no se persiste.",
            placeholder="sk-...",
        )
        if key_input:
            st.session_state["llm_api_key"] = key_input
            api_key = key_input
        elif st.session_state.get("llm_api_key"):
            api_key = st.session_state["llm_api_key"]

        if not api_key:
            st.warning("⚠️ Ingresá una API Key para usar este proveedor.")
    else:
        # LMStudio local: intentar detectar modelos cargados
        if "LMStudio" in provider_name:
            loaded_models = []
            try:
                import requests as _req
                resp = _req.get(f"{lmstudio_url.rstrip('/')}/v1/models", timeout=2)
                if resp.status_code == 200:
                    data = resp.json()
                    if "data" in data:
                        loaded_models = [m["id"] for m in data["data"] if "embed" not in m["id"].lower()]
            except Exception:
                pass
            provider["models"] = loaded_models + (["local-model"] if not loaded_models else [])

    # Selector de modelo
    if provider["models"]:
        lmstudio_model = st.selectbox(
            "Modelo",
            options=provider["models"],
            index=0,
            help="Modelos disponibles para este proveedor.",
        )
    else:
        lmstudio_model = st.text_input(
            "Nombre del modelo",
            value="local-model",
            help="ID exacto del modelo (ej: 'llama-3.1-8b-instant').",
        )

    temperature = st.slider(
        "Temperatura", min_value=0.0, max_value=1.0, value=0.20, step=0.05,
    )
    timeout = st.number_input(
        "Timeout respuesta final (s)", min_value=10, max_value=600, value=60, step=10,
    )
    router_timeout = st.number_input(
        "Timeout llamadas cortas (s)", min_value=5, max_value=300, value=20, step=5,
        help="Timeout para el router, multi-query, HyDE y decompose.",
    )


    st.subheader("Debug")
    show_prompt = st.checkbox("Mostrar prompt y decision del router", value=False)
    show_debug_pipeline = st.checkbox("📊 Mostrar telemetría del pipeline", value=False,
                                      help="Muestra los detalles de cada paso: búsqueda semántica, BM25, RRF y reranking.")

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

            if show_debug_pipeline and msg.get("debug_data"):
                render_debug_info(msg["debug_data"])

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
        debug_data = None
        try:
            with st.spinner("Analizando pregunta y decidiendo estrategia..."):
                necesita_retrieval, pregunta_busqueda, strategy, reason = rag_core.route_query(
                    recent_history, question, lmstudio_url, lmstudio_model, router_timeout,
                    api_key=api_key,
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
                                pregunta_busqueda, lmstudio_url, lmstudio_model, router_timeout,
                                api_key=api_key,
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
                                pregunta_busqueda, lmstudio_url, lmstudio_model, router_timeout,
                                api_key=api_key,
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
                                pregunta_busqueda, lmstudio_url, lmstudio_model, router_timeout,
                                api_key=api_key,
                            )
                            all_queries.append(hyde_doc)
                        except rag_core.LMStudioError:
                            hyde_doc = None
                    if hyde_doc:
                        with st.expander("💡 HyDE: documento hipotetico generado", expanded=False):
                            st.text(hyde_doc)

                debug_data = {} if show_debug_pipeline else None
                with st.spinner(f"Recuperando fragmentos relevantes ({len(all_queries)} queries)..."):
                    if len(all_queries) > 1:
                        retrieved = rag_core.retrieve_with_multi_query(
                            all_queries, db_path, collection_name, embedding_model_name, n_results,
                            use_hybrid=use_hybrid, reranker_model=reranker_model,
                            debug_data=debug_data,
                        )
                    else:
                        retrieved = rag_core.retrieve_chunks(
                            pregunta_busqueda, db_path, collection_name, embedding_model_name, n_results,
                            use_hybrid=use_hybrid, reranker_model=reranker_model,
                            debug_data=debug_data,
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

            with st.spinner(f"Generando respuesta ({provider_name})..."):
                answer = rag_core.call_lmstudio_chat(
                    messages, lmstudio_url, lmstudio_model, temperature, timeout,
                    api_key=api_key,
                )
        except rag_core.LMStudioError as e:
            st.error(str(e))
            st.stop()

        st.markdown(answer)
        if retrieved:
            render_fragmentos(retrieved, use_hybrid, use_reranker, use_multi_query, use_decompose)
        if show_debug_pipeline and debug_data:
            render_debug_info(debug_data)

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
        "debug_data": debug_data if show_debug_pipeline else None,
    })
