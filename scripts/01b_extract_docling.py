#!/usr/bin/env python3
"""
01b_extract_docling.py
-----------------------
Alternativa a 01_extract_pdfs.py: extrae texto de PDFs usando **docling**
(IBM), que entiende la estructura del documento (títulos, secciones, párrafos,
tablas, etc.) en vez de hacer un dump de texto plano.

Ventajas sobre 01_extract_pdfs.py:
  - Los chunks nunca cruzan una sección/heading.
  - El texto de las tablas queda bien formateado (Markdown).
  - Cada chunk tiene metadata de la sección a la que pertenece (heading, level).
  - Opcionalmente aplica semantic chunking para ajustar tamaño semántico.

Uso básico (solo chunking estructural):
    python scripts/01b_extract_docling.py \\
        --input_dir ./data/raw \\
        --output_json ./data/processed/chunks_pdfs.json

Uso con semantic chunking adicional:
    python scripts/01b_extract_docling.py \\
        --input_dir ./data/raw \\
        --output_json ./data/processed/chunks_pdfs.json \\
        --semantic \\
        --threshold 0.75 \\
        --max_words 400

Requisitos:
    pip install docling
    # Los modelos de layout (~600 MB) se descargan automáticamente
    # la primera vez que se corre.

Fallback automático:
    Si docling falla en un PDF específico (corrupto, formato raro), el script
    cae automáticamente a pdfplumber + chunking por párrafo para ese archivo.
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    from docling.document_converter import DocumentConverter
    from docling_core.transforms.chunker import HierarchicalChunker
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False

try:
    import pdfplumber
    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Helpers de limpieza de texto
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normaliza espacios y saltos de línea."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Extracción con docling
# ---------------------------------------------------------------------------

def extract_with_docling(pdf_path: Path) -> list[dict]:
    """
    Usa docling para extraer chunks estructurales del PDF.
    Cada chunk representa una sección/subsección coherente del documento.

    Retorna lista de dicts con campos:
        id, source, page, heading, heading_level, is_table, text
    """
    converter = DocumentConverter()
    chunker = HierarchicalChunker()

    try:
        result = converter.convert(str(pdf_path))
    except Exception as e:
        raise RuntimeError(f"docling no pudo convertir {pdf_path.name}: {e}") from e

    doc = result.document
    raw_chunks = list(chunker.chunk(doc))

    records = []
    for i, chunk in enumerate(raw_chunks):
        # Texto del chunk
        text = clean_text(chunk.text)
        if not text or len(text) < 20:
            continue

        # Metadata de la sección
        headings = getattr(chunk.meta, "headings", None) or []
        heading_str = " > ".join(headings) if headings else ""
        heading_level = len(headings)

        # Detectar si el chunk contiene una tabla (texto con pipes de markdown)
        is_table = "|" in text and text.count("|") >= 4

        # Página: docling puede exponer página en prov
        page = None
        try:
            prov_list = getattr(chunk.meta, "doc_items", [])
            if prov_list:
                first_item = prov_list[0]
                provs = getattr(first_item, "prov", [])
                if provs:
                    page = provs[0].page_no
        except Exception:
            pass

        record = {
            "id": f"{pdf_path.stem}_doc_{i:04d}",
            "source": pdf_path.name,
            "page": page,
            "heading": heading_str,
            "heading_level": heading_level,
            "is_table": is_table,
            "text": text,
        }
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Fallback: extracción con pdfplumber + chunking por párrafo
# ---------------------------------------------------------------------------

def extract_with_pdfplumber_fallback(pdf_path: Path) -> list[dict]:
    """
    Fallback cuando docling falla. Usa pdfplumber y divide por párrafos
    (doble salto de línea), sin chunking fijo de caracteres.
    """
    if not PDFPLUMBER_AVAILABLE:
        print(f"  [SKIP] {pdf_path.name}: docling falló y pdfplumber no está instalado.")
        return []

    records = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw = (page.extract_text() or "").strip()
            if not raw:
                continue
            paragraphs = re.split(r"\n{2,}", raw)
            for j, para in enumerate(paragraphs):
                text = clean_text(para)
                if text and len(text) >= 30:
                    records.append({
                        "id": f"{pdf_path.stem}_fb_p{page_num}_c{j}",
                        "source": pdf_path.name,
                        "page": page_num,
                        "heading": "",
                        "heading_level": 0,
                        "is_table": False,
                        "text": text,
                    })
    return records


# ---------------------------------------------------------------------------
# Procesamiento de un PDF
# ---------------------------------------------------------------------------

def process_pdf(pdf_path: Path, use_semantic: bool, sem_kwargs: dict) -> list[dict]:
    """
    Procesa un PDF con docling (o fallback) y opcionalmente aplica
    semantic chunking sobre los chunks resultantes.
    """
    print(f"\n  Procesando: {pdf_path.name}")

    # 1. Extracción estructural
    if DOCLING_AVAILABLE:
        try:
            records = extract_with_docling(pdf_path)
            print(f"    → docling: {len(records)} chunks estructurales")
        except RuntimeError as e:
            print(f"    [!] {e}")
            print(f"    → Usando fallback pdfplumber...")
            records = extract_with_pdfplumber_fallback(pdf_path)
            print(f"    → fallback: {len(records)} chunks")
    else:
        print(f"    [!] docling no está instalado. Usando fallback pdfplumber...")
        records = extract_with_pdfplumber_fallback(pdf_path)
        print(f"    → fallback: {len(records)} chunks")

    if not records:
        return []

    # 2. Semantic chunking (opcional)
    if use_semantic:
        # Importar desde el mismo directorio de scripts
        import importlib.util, os
        script_dir = Path(__file__).parent
        spec = importlib.util.spec_from_file_location(
            "semantic_chunker", script_dir / "semantic_chunker.py"
        )
        sem_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sem_mod)

        records = sem_mod.semantic_merge(records, **sem_kwargs)
        print(f"    → después de semantic chunking: {len(records)} chunks")

    return records


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extrae texto de PDFs con docling (chunking estructural) "
            "y opcionalmente aplica semantic chunking."
        )
    )
    parser.add_argument("--input_dir", required=True,
                        help="Carpeta con los PDFs a procesar")
    parser.add_argument("--output_json", default="./data/processed/chunks_pdfs.json",
                        help="Archivo JSON de salida")
    parser.add_argument("--semantic", action="store_true",
                        help="Aplicar semantic chunking después del chunking estructural")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Umbral de similitud coseno para semantic chunking (default: 0.75)")
    parser.add_argument("--max_words", type=int, default=400,
                        help="Máximo de palabras por chunk (default: 400)")
    parser.add_argument("--min_words", type=int, default=30,
                        help="Mínimo de palabras por chunk antes de fusionar (default: 30)")
    parser.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2",
                        help="Modelo de sentence-transformers para semantic chunking")
    args = parser.parse_args()

    if not DOCLING_AVAILABLE:
        print(
            "[AVISO] docling no está instalado. Se usará pdfplumber como fallback.\n"
            "Para instalar docling: pip install docling\n"
        )

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"No existe la carpeta: {input_dir}")

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        sys.exit(f"No se encontraron PDFs en {input_dir}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sem_kwargs = {
        "model_name": args.model,
        "threshold": args.threshold,
        "max_words": args.max_words,
        "min_words": args.min_words,
    }

    all_chunks = []
    stats = []

    for pdf_path in pdf_files:
        chunks = process_pdf(pdf_path, args.semantic, sem_kwargs)
        all_chunks.extend(chunks)
        avg_words = (
            sum(len(c["text"].split()) for c in chunks) // len(chunks)
            if chunks else 0
        )
        stats.append((pdf_path.name, len(chunks), avg_words))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # Resumen
    print(f"\n{'Archivo':<55} {'Chunks':>8} {'Prom. palabras':>15}")
    print("-" * 82)
    for name, n_chunks, avg_w in stats:
        print(f"{name:<55} {n_chunks:>8} {avg_w:>15}")
    print("-" * 82)
    print(f"Total chunks generados: {len(all_chunks)}")
    print(f"Modo semantic chunking: {'SÍ' if args.semantic else 'NO'}")
    print(f"Guardado en: {output_path}\n")


if __name__ == "__main__":
    main()
