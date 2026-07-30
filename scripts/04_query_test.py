#!/usr/bin/env python3
"""
04_query_test.py
------------------
Prueba rápida de búsqueda en ChromaDB, sin pasar todavía por el LLM.
Sirve para validar que la recuperación de chunks anda bien antes de conectar
LMStudio (Parte 2, pasos 2-3 del flujo de consulta).

Usa rag_core.retrieve_chunks(), la misma función que usan 05_rag_query.py y
streamlit_app.py, para que este script pruebe exactamente la lógica real de
recuperación (incluyendo el forzado del chunk-resumen de año) y no una copia
que se puede desincronizar.

Uso:
    python scripts/04_query_test.py --query "¿Qué correlativas tiene Cálculo Numérico?"
    python scripts/04_query_test.py --query "requisitos de admisión" --n_results 5
"""

import argparse

import rag_core


def main():
    parser = argparse.ArgumentParser(description="Prueba de búsqueda semántica en ChromaDB")
    parser.add_argument("--query", required=True, help="Pregunta a buscar")
    parser.add_argument("--db_path", default="./chroma_db", help="Carpeta donde persiste ChromaDB")
    parser.add_argument("--collection", default="curriculum", help="Nombre de la colección")
    parser.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2",
                         help="Debe ser el mismo modelo usado en 03_build_vectordb.py")
    parser.add_argument("--n_results", type=int, default=5, help="Cantidad de chunks a recuperar")
    parser.add_argument("--hybrid", action="store_true", help="Usa búsqueda híbrida (BM25 + Semántica + RRF)")
    parser.add_argument("--reranker", default=None, help="Modelo de reranking local (ej. BAAI/bge-reranker-v2-m3)")
    args = parser.parse_args()

    retrieved = rag_core.retrieve_chunks(
        args.query, args.db_path, args.collection, args.model, args.n_results,
        use_hybrid=args.hybrid, reranker_model=args.reranker
    )
    collection = rag_core.get_chroma_collection(args.db_path, args.collection)

    print(f"\nPregunta: {args.query}")
    print(f"Total documentos en la colección: {collection.count()}\n")
    print("=" * 80)

    for i, (doc, meta, dist) in enumerate(retrieved, start=1):
        if dist is None:
            dist_txt = "forzado (palabra clave)"
        else:
            if args.reranker:
                dist_txt = f"score: {dist:.4f} (reranker)"
            elif args.hybrid:
                dist_txt = f"score: {dist:.4f} (rrf)"
            else:
                dist_txt = f"distancia: {dist:.4f}"

        print(f"[{i}] {dist_txt}  |  fuente: {meta.get('source', '?')}"
              f"{'  |  página: ' + str(meta['page']) if 'page' in meta else ''}")
        print("-" * 80)
        print(doc[:400])
        print("=" * 80)


if __name__ == "__main__":
    main()