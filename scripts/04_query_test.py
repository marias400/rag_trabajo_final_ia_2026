#!/usr/bin/env python3
"""
04_query_test.py
------------------
Prueba rápida de búsqueda en ChromaDB, sin pasar todavía por el LLM.
Sirve para validar que la recuperación de chunks anda bien antes de conectar
LMStudio (Parte 2, pasos 2-3 del flujo de consulta).

Uso:
    python scripts/04_query_test.py --query "¿Qué correlativas tiene Cálculo Numérico?"
    python scripts/04_query_test.py --query "requisitos de admisión" --n_results 5
"""

import argparse
import sys

try:
    import chromadb
except ImportError:
    sys.exit("Falta chromadb. Instalalo con: pip install chromadb --break-system-packages")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Falta sentence-transformers. Instalalo con: pip install sentence-transformers --break-system-packages")


def main():
    parser = argparse.ArgumentParser(description="Prueba de búsqueda semántica en ChromaDB")
    parser.add_argument("--query", required=True, help="Pregunta a buscar")
    parser.add_argument("--db_path", default="./chroma_db", help="Carpeta donde persiste ChromaDB")
    parser.add_argument("--collection", default="curriculum", help="Nombre de la colección")
    parser.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2",
                         help="Debe ser el mismo modelo usado en 03_build_vectordb.py")
    parser.add_argument("--n_results", type=int, default=3, help="Cantidad de chunks a recuperar")
    args = parser.parse_args()

    model = SentenceTransformer(args.model)
    client = chromadb.PersistentClient(path=args.db_path)
    collection = client.get_collection(args.collection)

    query_embedding = model.encode([args.query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=args.n_results)

    print(f"\nPregunta: {args.query}")
    print(f"Total documentos en la colección: {collection.count()}\n")
    print("=" * 80)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances), start=1):
        print(f"[{i}] distancia: {dist:.4f}  |  fuente: {meta.get('source', '?')}"
              f"{'  |  página: ' + str(meta['page']) if 'page' in meta else ''}")
        print("-" * 80)
        print(doc[:400])
        print("=" * 80)


if __name__ == "__main__":
    main()