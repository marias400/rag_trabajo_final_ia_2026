#!/usr/bin/env python3
"""
01_extract_pdfs.py
-------------------
Paso 1 y 2 del pipeline RAG: extrae texto de PDFs y lo divide en chunks.
Detecta automáticamente si un PDF es escaneado (sin capa de texto) y en ese
caso usa OCR (tesseract) en vez de extracción directa.

Uso básico:
    python scripts/01_extract_pdfs.py --input_dir ./data/raw --output_json ./data/processed/chunks_pdfs.json

Uso con semantic chunking (opcional):
    python scripts/01_extract_pdfs.py \\
        --input_dir ./data/raw \\
        --output_json ./data/processed/chunks_pdfs.json \\
        --semantic --threshold 0.75 --max_words 400

Requisitos:
    pip install pdfplumber pytesseract --break-system-packages
    Además necesitás tesseract instalado en el sistema (no es paquete de pip):
      - Windows: instalador en https://github.com/UB-Mannheim/tesseract/wiki
        y agregar el idioma español (spa) durante la instalación
      - Linux:  sudo apt install tesseract-ocr tesseract-ocr-spa poppler-utils

Diseñado para ser genérico: apuntá --input_dir a CUALQUIER carpeta con PDFs
(curriculum, horarios de bus, lo que sea) y el script hace lo mismo.
"""

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import pdfplumber
except ImportError:
    sys.exit("Falta pdfplumber. Instalalo con: pip install pdfplumber --break-system-packages")

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


def configure_tesseract(tesseract_path: str | None):
    """En Windows, pytesseract no siempre encuentra tesseract.exe solo.
    Si se pasa una ruta explícita, se la seteamos acá."""
    if tesseract_path and OCR_AVAILABLE:
        pytesseract.pytesseract.tesseract_cmd = tesseract_path


def has_text_layer(pdf_path: Path) -> bool:
    """True si al menos una de las primeras páginas tiene texto extraíble directamente."""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages[:3]:
            if (page.extract_text() or "").strip():
                return True
    return False


def extract_pages(pdf_path: Path, ocr_lang: str = "spa", ocr_dpi: int = 200) -> list[dict]:
    """
    Devuelve una lista de {'page': N, 'text': str} por cada página con texto.
    Si el PDF no tiene capa de texto (escaneado), usa OCR por página.
    """
    pages = []
    use_ocr = not has_text_layer(pdf_path)

    if use_ocr and not OCR_AVAILABLE:
        print(f"  [!] {pdf_path.name} es un PDF escaneado y falta pytesseract. "
              f"Instalalo con: pip install pytesseract --break-system-packages")
        return []

    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            if use_ocr:
                img = page.to_image(resolution=ocr_dpi).original
                text = pytesseract.image_to_string(img, lang=ocr_lang)
            else:
                text = page.extract_text() or ""
            text = text.strip()
            if text:
                pages.append({"page": i, "text": text})
    if use_ocr:
        print(f"  [OCR] {pdf_path.name} procesado con tesseract ({len(pages)} páginas con texto)")
    return pages


def clean_text(text: str) -> str:
    """Normaliza espacios y saltos de línea raros que deja la extracción de PDF."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Chunking mejorado: prioriza cortar en saltos de párrafo o puntos,
    para evitar partir ideas u oraciones por la mitad.
    """
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:end].strip())
            break
            
        # Intentar cortar en el último \n\n, \n, punto o espacio dentro del límite
        best_cut = -1
        for sep in ["\n\n", "\n", ". ", " "]:
            cut_pos = text.rfind(sep, start, end)
            if cut_pos != -1 and cut_pos > start:
                best_cut = cut_pos + len(sep) if sep == ". " else cut_pos
                break
        
        if best_cut == -1:
            best_cut = end
            
        chunk = text[start:best_cut].strip()
        if chunk:
            chunks.append(chunk)
            
        # Calcular próximo inicio con solapamiento
        start = best_cut - overlap if best_cut - overlap > start else best_cut
        
    return chunks


def process_pdf(pdf_path: Path, chunk_size: int, overlap: int, ocr_lang: str = "spa") -> list[dict]:
    """Extrae y chunkea un PDF entero, devolviendo chunks con metadata."""
    records = []
    pages = extract_pages(pdf_path, ocr_lang=ocr_lang)
    for page_info in pages:
        page_chunks = chunk_text(page_info["text"], chunk_size, overlap)
        for j, chunk in enumerate(page_chunks):
            records.append({
                "id": f"{pdf_path.stem}_p{page_info['page']}_c{j}",
                "source": pdf_path.name,
                "page": page_info["page"],
                "text": chunk,
            })
    return records


def main():
    parser = argparse.ArgumentParser(description="Extrae texto de PDFs y genera chunks para RAG")
    parser.add_argument("--input_dir", required=True, help="Carpeta con los PDFs a procesar")
    parser.add_argument("--output_json", default="chunks.json", help="Archivo JSON de salida")
    parser.add_argument("--chunk_size", type=int, default=800, help="Tamaño de chunk en caracteres")
    parser.add_argument("--overlap", type=int, default=150, help="Solapamiento entre chunks en caracteres")
    parser.add_argument("--ocr_lang", default="spa", help="Idioma para OCR (spa=español, eng=inglés)")
    parser.add_argument("--tesseract_path", default=None,
                         help=r"Ruta al ejecutable tesseract.exe en Windows, ej: C:\Program Files\Tesseract-OCR\tesseract.exe")
    # Semantic chunking (opcional)
    parser.add_argument("--semantic", action="store_true",
                        help="Aplica semantic chunking tras la extracción (fusiona/divide según similitud semántica)")
    parser.add_argument("--threshold", type=float, default=0.75,
                        help="Umbral de similitud coseno para semantic chunking (default: 0.75)")
    parser.add_argument("--max_words", type=int, default=400,
                        help="Máximo de palabras por chunk en semantic chunking (default: 400)")
    parser.add_argument("--min_words", type=int, default=30,
                        help="Mínimo de palabras por chunk en semantic chunking (default: 30)")
    parser.add_argument("--model", default="paraphrase-multilingual-MiniLM-L12-v2",
                        help="Modelo de sentence-transformers para semantic chunking")
    args = parser.parse_args()

    configure_tesseract(args.tesseract_path)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        sys.exit(f"No existe la carpeta: {input_dir}")

    pdf_files = sorted(input_dir.glob("*.pdf"))
    if not pdf_files:
        sys.exit(f"No se encontraron PDFs en {input_dir}")

    output_path = Path(args.output_json)
    if output_path.is_dir():
        output_path = output_path / "chunks_pdfs.json"

    all_chunks = []
    stats = []
    for pdf_path in pdf_files:
        records = process_pdf(pdf_path, args.chunk_size, args.overlap, args.ocr_lang)
        all_chunks.extend(records)
        n_pages = len({r["page"] for r in records})
        avg_len = sum(len(r["text"]) for r in records) // len(records) if records else 0
        stats.append((pdf_path.name, n_pages, len(records), avg_len))

    # Semantic chunking (opcional)
    if args.semantic:
        import importlib.util
        script_dir = Path(__file__).parent
        spec = importlib.util.spec_from_file_location(
            "semantic_chunker", script_dir / "semantic_chunker.py"
        )
        sem_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sem_mod)
        print("\nAplicando semantic chunking...")
        all_chunks = sem_mod.semantic_merge(
            all_chunks,
            model_name=args.model,
            threshold=args.threshold,
            max_words=args.max_words,
            min_words=args.min_words,
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    # Resumen tabular
    print(f"\n{'Archivo':<55} {'Páginas':>8} {'Chunks':>8} {'Prom. chars':>12}")
    print("-" * 87)
    for name, n_pages, n_chunks, avg_len in stats:
        print(f"{name:<55} {n_pages:>8} {n_chunks:>8} {avg_len:>12}")
    print("-" * 87)
    print(f"Total chunks generados: {len(all_chunks)}")
    print(f"Modo semantic chunking: {'SÍ (threshold={:.2f})'.format(args.threshold) if args.semantic else 'NO'}")
    print(f"Guardado en: {output_path}\n")


if __name__ == "__main__":
    main()