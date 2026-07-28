#!/usr/bin/env python3
"""
03_build_vectordb.py
---------------------
Paso 3 y 4 del pipeline RAG: toma los chunks generados (PDFs + correlatividades),
genera embeddings con sentence-transformers y los guarda en ChromaDB persistente.

Uso:
    python scripts/03_build_vectordb.py \
        --input_json ./data/processed/chunks_pdfs.json ./data/processed/chunks_correlatividades.json \
        --db_path ./chroma_db \
        --collection curriculum

Requisitos:
    pip install chromadb sentence-transformers --break-system-packages

Modelo de embeddings: usa "paraphrase-multilingual-MiniLM-L12-v2" por defecto,
que anda bien en español y es liviano (funciona sin GPU). Se puede cambiar con
--model si más adelante querés probar otro.
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import chromadb
except ImportError:
    sys.exit("Falta chromadb. Instalalo con: pip install chromadb --break-system-packages")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Falta sentence-transformers. Instalalo con: pip install sentence-transformers --break-system-packages")


def load_chunks(paths: list[str]) -> list[dict]:
    """Carga y unifica uno o más archivos JSON de chunks en una sola lista."""
    all_chunks = []
    for p in paths:
        path = Path(p)
        if not path.is_file():
            sys.exit(f"No existe el archivo: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for c in data:
            c["_origin_file"] = path.name
        all_chunks.extend(data)
    return all_chunks


def build_metadata(chunk: dict) -> dict:
    """
    ChromaDB no acepta listas ni None en metadata, solo str/int/float/bool.
    Nos quedamos con los campos simples que sirven para mostrar la fuente.
    """
    meta = {"source": chunk.get("source", chunk.get("_origin_file", "desconocido"))}
    if "page" in chunk:
        meta["page"] = chunk["page"]
    if "asignatura" in chunk:
        meta["asignatura"] = chunk["asignatura"]
    return meta


def main():
    parser = argparse.ArgumentParser(description="Genera embeddings y los guarda en ChromaDB")
    parser.add_argument("--input_json", nargs="+", required=True,
                         help="Uno o más archivos JSON de chunks (separados por espacio)")
    parser.add_argument("--db_path", default="./chroma_db", help="Carpeta donde persiste ChromaDB")
    parser.add_argument("--collection", default="curriculum", help="Nombre de la colección en ChromaDB")
    parser.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2",
                         help="Modelo de sentence-transformers a usar")
    parser.add_argument("--reset", action="store_true",
                         help="Si se pasa, borra la colección existente antes de crearla de nuevo")
    args = parser.parse_args()

    chunks = load_chunks(args.input_json)
    if not chunks:
        sys.exit("No se cargó ningún chunk. Revisá las rutas de --input_json.")

    print(f"Chunks cargados: {len(chunks)}")
    print(f"Cargando modelo de embeddings '{args.model}' (puede tardar la primera vez)...")
    model = SentenceTransformer(args.model)

    client = chromadb.PersistentClient(path=args.db_path)

    if args.reset:
        try:
            client.delete_collection(args.collection)
            print(f"Colección '{args.collection}' anterior eliminada.")
        except Exception:
            pass

    collection = client.get_or_create_collection(name=args.collection)

    ids = [c["id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [build_metadata(c) for c in chunks]

    print("Generando embeddings...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32).tolist()

    # upsert en lotes para no golpear memoria con datasets grandes
    batch_size = 200
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            documents=texts[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

    print(f"\nListo. {collection.count()} documentos en la colección '{args.collection}' ({args.db_path})")


if __name__ == "__main__":
    main()