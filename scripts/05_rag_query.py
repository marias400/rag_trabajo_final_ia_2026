#!/usr/bin/env python3
"""
05_rag_query.py
-----------------
Pipeline de consulta (Parte 2 del flujo): recupera los chunks más relevantes
de ChromaDB, arma el prompt con contexto y llama a LMStudio (servidor local,
API compatible con OpenAI) para generar la respuesta final.

Dos modos:

1) Single-shot (comportamiento original, para pruebas puntuales):
    python scripts/05_rag_query.py --query "¿Qué correlativas tiene Cálculo Numérico?"

2) Chat interactivo (nuevo): mantiene una conversación fluida con memoria
   real (se manda el historial como mensajes de chat al LLM) y un router que
   decide en cada turno si hace falta volver a consultar ChromaDB o si
   alcanza con lo ya conversado. Si hace falta, reformula la pregunta como
   autocontenida (resolviendo pronombres tipo "esa", "la anterior") antes de
   buscar, para que el embedding tenga algo concreto con qué trabajar.
    python scripts/05_rag_query.py --chat

Técnicas de mejora de query (flags independientes, combinables):
  --multi_query  Genera hasta 5 preguntas alternativas y fusiona resultados con RRF.
  --hyde         HyDE: embeddea un documento hipotético de respuesta en lugar de
                 la pregunta (mejor recall semántico en corpus técnico).
  --decompose    Descompone preguntas complejas en sub-preguntas simples.

Toda la lógica de recuperación, router y llamadas a LMStudio vive en
rag_core.py (compartida con otras interfaces, ej. app/streamlit_app.py).
"""

import argparse
import sys

import rag_core


# Cuántos turnos previos (pares pregunta+respuesta) se mandan como historial
# al router y al LLM de respuesta. Limitarlo evita que el prompt crezca sin
# límite en conversaciones largas; 6 turnos alcanza de sobra para resolver
# pronombres y mantener coherencia sin inflar el contexto innecesariamente.
N_HISTORY_TURNS = 6


def print_fragmentos(retrieved: list, args) -> None:
    print("\nFRAGMENTOS RECUPERADOS:")
    print("-" * 80)
    for i, (doc, meta, dist) in enumerate(retrieved, start=1):
        fuente = meta.get("source", "?")
        pagina = f" (página {meta['page']})" if "page" in meta else ""
        
        if dist is None:
            dist_txt = "forzado (palabra clave)"
        elif args.reranker:
            dist_txt = f"score: {dist:.4f} (reranker)"
        elif args.multi_query or args.decompose:
            dist_txt = f"score: {dist:.4f} (rrf multi-query)"
        elif args.hybrid:
            dist_txt = f"score: {dist:.4f} (rrf)"
        else:
            dist_txt = f"distancia: {dist:.4f}"
                
        print(f"[{i}] {fuente}{pagina} - {dist_txt}")
        print(f"    {doc[:200]}...")
    print("-" * 80)


def _build_retrieval(pregunta_busqueda: str, args, label: str = "") -> list:
    """Orquesta la recuperación según los flags activos. Devuelve la lista de chunks."""
    lbl = f"[{label}] " if label else ""

    all_queries = [pregunta_busqueda]

    if args.decompose:
        print(f"{lbl}Descomponiendo la pregunta en sub-preguntas...")
        try:
            sub_queries = rag_core.decompose_query(
                pregunta_busqueda, args.lmstudio_url, args.lmstudio_model, args.router_timeout
            )
        except rag_core.LMStudioError as e:
            print(f"  Advertencia: Error en descomposición ({e}).")
            sub_queries = []
        if sub_queries:
            all_queries.extend(sub_queries)
            if args.show_prompt:
                print(f"  [decompose] {len(sub_queries)} sub-pregunta(s):")
                for j, q in enumerate(sub_queries, 1):
                    print(f"    {j}. {q}")

    if args.multi_query:
        print(f"{lbl}Generando preguntas alternativas (Multi-Query)...")
        try:
            variants = rag_core.generate_multi_queries(
                pregunta_busqueda, args.lmstudio_url, args.lmstudio_model, args.router_timeout
            )
        except rag_core.LMStudioError as e:
            print(f"  Advertencia: Error en multi-query ({e}).")
            variants = []
        if variants:
            all_queries.extend(variants)
            if args.show_prompt:
                print(f"  [multi_query] {len(variants)} variantes:")
                for j, q in enumerate(variants, 1):
                    print(f"    {j}. {q}")

    if args.hyde:
        print(f"{lbl}Generando documento hipotético (HyDE)...")
        try:
            hyde_doc = rag_core.generate_hyde_document(
                pregunta_busqueda, args.lmstudio_url, args.lmstudio_model, args.router_timeout
            )
        except rag_core.LMStudioError as e:
            print(f"  Advertencia: Error en HyDE ({e}).")
            hyde_doc = None
        if hyde_doc:
            all_queries.append(hyde_doc)
            if args.show_prompt:
                print(f"  [hyde] documento hipotético: {hyde_doc[:200]}...")

    print(f"{lbl}Recuperando chunks relevantes ({len(all_queries)} queries)...")
    if len(all_queries) > 1:
        retrieved = rag_core.retrieve_with_multi_query(
            all_queries, args.db_path, args.collection, args.embedding_model, args.n_results,
            use_hybrid=args.hybrid, reranker_model=args.reranker
        )
    else:
        retrieved = rag_core.retrieve_chunks(
            pregunta_busqueda, args.db_path, args.collection, args.embedding_model, args.n_results,
            use_hybrid=args.hybrid, reranker_model=args.reranker
        )

    return retrieved


def run_single_shot(args) -> None:
    pregunta = args.query
    if args.agent:
        print("🤖 [Agente] Analizando la pregunta...")
        try:
            necesita, p_reform, strat, reason = rag_core.route_query(
                [], args.query, args.lmstudio_url, args.lmstudio_model, args.router_timeout
            )
            pregunta = p_reform
            args.decompose = (strat == "decompose")
            args.multi_query = (strat == "multi_query")
            args.hyde = (strat == "hyde")
            
            strat_labels = {
                "direct": "🎯 Búsqueda directa",
                "decompose": "🔍 Query Decomposition",
                "multi_query": "🔀 Multi-Query",
                "hyde": "💡 HyDE",
            }
            print(f"🤖 [Agente] Estrategia elegida: {strat_labels.get(strat, strat)} | Motivo: {reason}")
            if args.show_prompt:
                print(f"  [router] pregunta reformulada: {pregunta}")
        except rag_core.LMStudioError as e:
            sys.exit(f"Error en router: {e}")

    retrieved = _build_retrieval(pregunta, args)

    user_message = rag_core.build_rag_user_message(pregunta, retrieved)
    messages = [
        {"role": "system", "content": rag_core.SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    if args.show_prompt:
        print("\n" + "=" * 80)
        print("PROMPT ENVIADO AL LLM:")
        print("=" * 80)
        print(user_message)
        print("=" * 80 + "\n")

    print(f"Consultando LMStudio en {args.lmstudio_url} (timeout: {args.timeout}s, "
          f"puede tardar bastante con modelos grandes)...\n")
    try:
        answer = rag_core.call_lmstudio_chat(
            messages, args.lmstudio_url, args.lmstudio_model, args.temperature, args.timeout
        )
    except rag_core.LMStudioError as e:
        sys.exit(str(e))

    print("=" * 80)
    print("RESPUESTA DEL LLM:")
    print("=" * 80)
    print(answer)
    print("=" * 80)

    print_fragmentos(retrieved, args)


def run_chat(args) -> None:
    print("Chat interactivo iniciado. Escribí 'salir' para terminar.\n")
    history: list = []

    while True:
        try:
            question = input("Vos: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nChau!")
            break

        if not question:
            continue
        if question.lower() in {"salir", "exit", "quit"}:
            print("Chau!")
            break

        recent_history = history[-2 * N_HISTORY_TURNS:] if N_HISTORY_TURNS > 0 else []

        try:
            necesita_retrieval, pregunta_busqueda, strategy, reason = rag_core.route_query(
                recent_history, question, args.lmstudio_url, args.lmstudio_model, args.router_timeout
            )
        except rag_core.LMStudioError as e:
            print(f"\n⚠️  {e}\n")
            continue

        if args.agent and necesita_retrieval:
            args.decompose = (strategy == "decompose")
            args.multi_query = (strategy == "multi_query")
            args.hyde = (strategy == "hyde")

            strat_labels = {
                "direct": "🎯 Búsqueda directa",
                "decompose": "🔍 Query Decomposition",
                "multi_query": "🔀 Multi-Query",
                "hyde": "💡 HyDE",
            }
            print(f"    🤖 agente → {strat_labels.get(strategy, strategy)} | {reason}")

        if args.show_prompt:
            print(f"    [router] RETRIEVAL: {'SI' if necesita_retrieval else 'NO'}"
                  f"{' | pregunta reformulada: ' + pregunta_busqueda if necesita_retrieval else ''}")

        if necesita_retrieval:
            try:
                retrieved = _build_retrieval(pregunta_busqueda, args, label="retrieval")
            except rag_core.LMStudioError as e:
                print(f"\n⚠️  {e}\n")
                continue
            current_message = rag_core.build_rag_user_message(pregunta_busqueda, retrieved)
        else:
            retrieved = None
            current_message = question

        messages = [{"role": "system", "content": rag_core.SYSTEM_PROMPT}]
        messages.extend(recent_history)
        messages.append({"role": "user", "content": current_message})

        try:
            answer = rag_core.call_lmstudio_chat(
                messages, args.lmstudio_url, args.lmstudio_model, args.temperature, args.timeout
            )
        except rag_core.LMStudioError as e:
            print(f"\n⚠️  {e}\n")
            continue

        print(f"\nAsistente: {answer}\n")
        if retrieved:
            print_fragmentos(retrieved, args)
            print()

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": answer})


def main():
    parser = argparse.ArgumentParser(description="Pipeline RAG: ChromaDB + LMStudio")
    parser.add_argument("--query", help="Pregunta puntual (modo single-shot). Si no se pasa, usá --chat.")
    parser.add_argument("--chat", action="store_true", help="Inicia un chat interactivo con memoria y router")
    parser.add_argument("--db_path", default="./chroma_db", help="Carpeta donde persiste ChromaDB")
    parser.add_argument("--collection", default="curriculum", help="Nombre de la colección")
    parser.add_argument("--embedding_model", default="paraphrase-multilingual-MiniLM-L12-v2",
                         help="Debe ser el mismo modelo usado en 03_build_vectordb.py")
    parser.add_argument("--n_results", type=int, default=5, help="Cantidad de chunks a recuperar")
    parser.add_argument("--lmstudio_url", default="http://localhost:1234",
                         help="URL base del servidor local de LMStudio")
    parser.add_argument("--lmstudio_model", default="local-model",
                         help="Nombre del modelo cargado en LMStudio ('local-model' usa el que esté activo).")
    parser.add_argument("--temperature", type=float, default=0.2,
                         help="Temperatura del LLM para la respuesta final (0.20 recomendado).")
    parser.add_argument("--timeout", type=int, default=90,
                         help="Segundos a esperar la respuesta de LMStudio antes de cortar.")
    parser.add_argument("--router_timeout", type=int, default=30,
                         help="Timeout para llamadas cortas al LLM (router, multi-query, HyDE, decompose).")
    parser.add_argument("--show_prompt", action="store_true",
                         help="Si se pasa, imprime el prompt/decisión del router y sub-queries generadas (útil para debug)")
    # --- Retrieval avanzado ---
    parser.add_argument("--hybrid", action="store_true", help="Usa búsqueda híbrida (BM25 + Semántica + RRF)")
    parser.add_argument("--reranker", default=None, help="Modelo de reranking local (ej. BAAI/bge-reranker-v2-m3)")
    # --- Query Enhancement ---
    parser.add_argument("--agent", action="store_true",
                         help="Modo Agente: analiza la pregunta y decide automáticamente qué técnica aplicar.")
    parser.add_argument("--multi_query", action="store_true",
                         help="Multi-Query: genera hasta 5 preguntas alternativas y fusiona resultados con RRF.")
    parser.add_argument("--hyde", action="store_true",
                         help="HyDE: genera un documento hipotético de respuesta y lo embeddea en lugar de la pregunta.")
    parser.add_argument("--decompose", action="store_true",
                         help="Query Decomposition: descompone preguntas complejas en sub-preguntas simples.")
    args = parser.parse_args()

    if args.chat and args.query:
        sys.exit("Elegí --query (single-shot) o --chat (interactivo), no ambos.")
    if not args.chat and not args.query:
        sys.exit("Falta --query (single-shot) o --chat (interactivo).")

    if args.chat:
        run_chat(args)
    else:
        run_single_shot(args)


if __name__ == "__main__":
    main()