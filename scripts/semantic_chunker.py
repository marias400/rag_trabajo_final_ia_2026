#!/usr/bin/env python3
"""
semantic_chunker.py
--------------------
Módulo auxiliar de **semantic chunking** para el pipeline RAG.

Dada una lista de chunks de texto, fusiona o divide chunks consecutivos según
su similitud semántica:
  - Si dos chunks consecutivos tienen similitud coseno >= threshold → se fusionan
    (pertenecen al mismo tema/idea).
  - Si un chunk supera max_words → se divide en mitades respetando oraciones.
  - Si un chunk queda con menos de min_words → se fusiona con el siguiente.

Uso como módulo (importado por otros scripts):
    from semantic_chunker import semantic_merge

    chunks_out = semantic_merge(
        chunks_in,
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        threshold=0.75,
        max_words=400,
        min_words=30,
    )

Uso directo desde CLI (para pruebas rápidas):
    python scripts/semantic_chunker.py \\
        --input_json ./data/processed/chunks_pdfs.json \\
        --output_json ./data/processed/chunks_pdfs_semantic.json \\
        --threshold 0.75 --max_words 400

Notas sobre el threshold:
  - 0.90+ : casi no fusiona nada (solo oraciones casi idénticas)
  - 0.75   : punto medio recomendado para texto técnico académico en español
  - 0.60   : fusiona bastante; puede crear chunks muy grandes
"""

import argparse
import json
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Tokenizador de oraciones ligero (sin dependencia de NLTK)
# ---------------------------------------------------------------------------

_SENT_RE = re.compile(
    r'(?<=[.!?])\s+'           # corte después de punto/exclamación/interrogación
    r'(?=[A-ZÁÉÍÓÚÜÑ\d])',     # seguido de mayúscula o número
    flags=re.UNICODE,
)


def split_sentences(text: str) -> list[str]:
    """Divide texto en oraciones usando un regex conservador (sin NLTK)."""
    # Proteger abreviaturas comunes en español
    text = re.sub(r'\b(Art|Ord|Prof|Dr|Dra|Ing|Lic|N[°º]|vs|etc|Ej)\.\s', r'\1<PUNKT> ', text)
    sentences = _SENT_RE.split(text)
    # Restaurar abreviaturas
    sentences = [s.replace('<PUNKT>', '.').strip() for s in sentences if s.strip()]
    return sentences or [text.strip()]


# ---------------------------------------------------------------------------
# Carga lazy del modelo de embeddings
# ---------------------------------------------------------------------------

_model_cache: dict = {}


def _get_model(model_name: str):
    if model_name not in _model_cache:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            sys.exit(
                "Falta sentence-transformers.\n"
                "Instalalo con: pip install sentence-transformers"
            )
        print(f"  [semantic_chunker] Cargando modelo '{model_name}'...")
        _model_cache[model_name] = SentenceTransformer(model_name)
    return _model_cache[model_name]


# ---------------------------------------------------------------------------
# Función de similitud coseno (sin scipy, solo numpy)
# ---------------------------------------------------------------------------

def _cosine_sim(a, b) -> float:
    import numpy as np
    a, b = np.array(a), np.array(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


# ---------------------------------------------------------------------------
# Chunking semántico a nivel de oraciones (texto → chunks)
# ---------------------------------------------------------------------------

def _semantic_split_text(
    text: str,
    model,
    threshold: float,
    max_words: int,
    min_words: int,
) -> list[str]:
    """
    Divide un texto en chunks semánticamente coherentes.
    Retorna lista de strings (los chunks).
    """
    sentences = split_sentences(text)
    if len(sentences) <= 1:
        return [text.strip()] if text.strip() else []

    import numpy as np

    # Embeddings de cada oración
    embeddings = model.encode(sentences, show_progress_bar=False)

    # --- Paso 1: detectar fronteras por similitud baja ---
    groups: list[list[str]] = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = _cosine_sim(embeddings[i - 1], embeddings[i])
        if sim >= threshold:
            groups[-1].append(sentences[i])
        else:
            groups.append([sentences[i]])

    chunks_text = [" ".join(g) for g in groups]

    # --- Paso 2: fusionar chunks demasiado cortos con el siguiente ---
    merged: list[str] = []
    i = 0
    while i < len(chunks_text):
        chunk = chunks_text[i]
        if len(chunk.split()) < min_words and i + 1 < len(chunks_text):
            chunks_text[i + 1] = chunk + " " + chunks_text[i + 1]
            i += 1
            continue
        merged.append(chunk)
        i += 1

    # --- Paso 3: dividir chunks demasiado largos por mitad de oración ---
    final: list[str] = []
    for chunk in merged:
        words = chunk.split()
        if len(words) <= max_words:
            final.append(chunk)
        else:
            # dividir en mitad buscando el límite de oración más cercano
            mid = len(words) // 2
            # reconstruir y buscar punto cerca de la mitad
            sentences_in_chunk = split_sentences(chunk)
            cumulative = 0
            split_idx = len(sentences_in_chunk) // 2
            for j, s in enumerate(sentences_in_chunk):
                cumulative += len(s.split())
                if cumulative >= mid:
                    split_idx = j + 1
                    break
            part_a = " ".join(sentences_in_chunk[:split_idx])
            part_b = " ".join(sentences_in_chunk[split_idx:])
            if part_a:
                final.append(part_a)
            if part_b:
                final.append(part_b)

    return [c.strip() for c in final if c.strip()]


# ---------------------------------------------------------------------------
# Función pública principal
# ---------------------------------------------------------------------------

def semantic_merge(
    chunks: list[dict],
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    threshold: float = 0.75,
    max_words: int = 400,
    min_words: int = 30,
) -> list[dict]:
    """
    Recibe una lista de chunks (dicts con campo 'text') y devuelve una lista
    nueva donde los chunks están ajustados semánticamente.

    Estrategia:
      - Toma el texto de cada chunk individual y lo re-divide en oraciones.
      - Luego aplica el algoritmo de similitud para decidir dónde cortar.
      - Preserva todos los campos del chunk original (source, page, id, etc.)
        en el primer chunk resultante. Los chunks adicionales que surjan de
        una división heredan esos campos con un sufijo '_sN' en el id.

    Args:
        chunks:      Lista de dicts de chunks (deben tener 'id' y 'text').
        model_name:  Modelo de sentence-transformers a usar.
        threshold:   Similitud coseno mínima para mantener oraciones juntas.
                     Valores entre 0.65 y 0.85 son razonables.
        max_words:   Límite de palabras por chunk (chunks más largos se dividen).
        min_words:   Mínimo de palabras por chunk (chunks más cortos se fusionan).

    Returns:
        Lista nueva de dicts con los chunks reajustados.
    """
    if not chunks:
        return []

    model = _get_model(model_name)
    result: list[dict] = []

    print(f"  [semantic_chunker] Procesando {len(chunks)} chunks "
          f"(threshold={threshold}, max_words={max_words})...")

    for chunk in chunks:
        text = chunk.get("text", "").strip()
        if not text:
            continue

        sub_texts = _semantic_split_text(text, model, threshold, max_words, min_words)

        if len(sub_texts) == 1:
            # Sin cambios: devolver el chunk original con text limpio
            new_chunk = dict(chunk)
            new_chunk["text"] = sub_texts[0]
            result.append(new_chunk)
        else:
            # El chunk original se dividió en varios
            base_id = chunk.get("id", f"chunk_{len(result)}")
            for j, sub_text in enumerate(sub_texts):
                new_chunk = dict(chunk)
                new_chunk["text"] = sub_text
                new_chunk["id"] = f"{base_id}_s{j}"
                result.append(new_chunk)

    print(f"  [semantic_chunker] {len(chunks)} chunks → {len(result)} chunks semánticos")
    return result


# ---------------------------------------------------------------------------
# CLI para uso directo / pruebas
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aplica semantic chunking a un JSON de chunks existente"
    )
    parser.add_argument("--input_json", required=True,
                        help="JSON de entrada con lista de chunks")
    parser.add_argument("--output_json", required=True,
                        help="JSON de salida con chunks semánticos")
    parser.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2",
                        help="Modelo de sentence-transformers")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Similitud coseno mínima para fusionar oraciones (0-1)")
    parser.add_argument("--max_words", type=int, default=400,
                        help="Máximo de palabras por chunk")
    parser.add_argument("--min_words", type=int, default=30,
                        help="Mínimo de palabras por chunk")
    args = parser.parse_args()

    input_path = Path(args.input_json)
    if not input_path.is_file():
        sys.exit(f"No existe: {input_path}")

    with open(input_path, encoding="utf-8") as f:
        chunks_in = json.load(f)

    print(f"Chunks cargados: {len(chunks_in)}")

    chunks_out = semantic_merge(
        chunks_in,
        model_name=args.model,
        threshold=args.threshold,
        max_words=args.max_words,
        min_words=args.min_words,
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks_out, f, ensure_ascii=False, indent=2)

    print(f"\nResultado: {len(chunks_out)} chunks semánticos guardados en {output_path}")
    avg_words = sum(len(c["text"].split()) for c in chunks_out) / len(chunks_out) if chunks_out else 0
    print(f"Promedio de palabras por chunk: {avg_words:.0f}")


if __name__ == "__main__":
    main()
