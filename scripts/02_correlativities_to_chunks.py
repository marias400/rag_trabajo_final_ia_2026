#!/usr/bin/env python3
"""
02_correlativities_to_chunks.py
--------------------------------
Convierte los JSON estructurados de correlatividades en chunks de texto natural
(uno por asignatura + un resumen por año), listos para vectorizar junto con el
resto de los chunks generados por 01_extract_pdfs.py.

Por qué un script aparte: la tabla de correlatividades es la parte más
consultada por los usuarios ("¿qué correlativas tiene X?") y el OCR la
destruye (mezcla columnas y pierde números). Esta tabla se transcribió/generó
desde los documentos originales y se guarda en formato estructurado para poder
generar texto 100% preciso, en vez de depender del OCR para el dato más
crítico del sistema.

Uso (se generan archivos separados por carrera):
    python scripts/02_correlativities_to_chunks.py \\
        --input_json ./data/structured/correlatividades_ing_sistemas_2024.json \\
                     ./data/structured/correlatividades_lic_sistemas_2024.json \\
                     ./data/structured/correlatividades_ing_mecatronica_2024.json \\
        --output_dir ./data/processed

Salida: por cada archivo de entrada se genera un archivo separado en output_dir
con el mismo nombre stem, ej.:
    chunks_correlatividades_ing_sistemas_2024.json
    chunks_correlatividades_lic_sistemas_2024.json
    chunks_correlatividades_ing_mecatronica_2024.json
"""

import argparse
import json
from pathlib import Path


def nombre_por_numero(asignaturas: dict, n: int) -> str:
    return asignaturas.get(n, f"Asignatura N° {n}")


def lista_a_texto(lista: list, asignaturas: dict) -> str:
    if not lista:
        return "no tiene correlativas (no corresponde)"
    nombres = [f"{n} ({nombre_por_numero(asignaturas, n)})" for n in lista]
    return ", ".join(nombres)


def build_chunk_text(asig: dict, asignaturas: dict, carrera: str, plan: str) -> str:
    regular = lista_a_texto(asig["regular_para_cursar"], asignaturas)
    aprobada = lista_a_texto(asig["aprobada_para_rendir_final"], asignaturas)
    anio_txt = f" Corresponde al {asig['anio']}° año de la carrera." if "anio" in asig else ""
    return (
        f"Correlativas, correlatividades y requisitos de la asignatura: {asig['nombre']} (N° {asig['n']}) de la carrera {carrera}, {plan}.{anio_txt}\n"
        f"Régimen de cursado: {asig['regimen']}.\n"
        f"Correlativas para cursar {asig['nombre']} (tener REGULAR): {regular}.\n"
        f"Correlativas para rendir examen final de {asig['nombre']} (tener APROBADA): {aprobada}."
    )


ORDINAL = {1: "primer", 2: "segundo", 3: "tercer", 4: "cuarto", 5: "quinto"}


def build_year_summary_chunks(data: dict, id_prefix: str) -> list:
    """
    Genera un chunk-resumen por año (uno por cada uno de los años de la carrera),
    con la cantidad y el listado completo de asignaturas de ese año.
    """
    asignaturas = data["asignaturas"]
    if not any("anio" in a for a in asignaturas):
        return []

    por_anio: dict = {}
    for asig in asignaturas:
        por_anio.setdefault(asig["anio"], []).append(asig)

    carrera = data["carrera"]
    plan = data["plan"]
    chunks = []
    for anio in sorted(por_anio):
        materias = por_anio[anio]
        nombres = "; ".join(f"{a['nombre']} (N° {a['n']})" for a in materias)
        ordinal = ORDINAL.get(anio, f"{anio}°")
        text = (
            f"El {ordinal} año de la carrera {carrera}, {plan}, tiene "
            f"{len(materias)} asignaturas. Son las siguientes: {nombres}."
        )
        chunks.append({
            "id": f"{id_prefix}_anio_{anio:02d}",
            "source": data.get("fuente_anio", data["fuente"]),
            "anio": anio,
            "text": text,
        })
    return chunks


def process_one(input_path: Path, output_dir: Path) -> tuple[int, int, Path]:
    """Procesa un único JSON de correlatividades y lo guarda en un archivo separado.
    Devuelve (n_asignaturas, n_anios, ruta_output).
    """
    stem = input_path.stem  # ej. correlatividades_ing_sistemas_2024
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    asignaturas_por_numero = {a["n"]: a["nombre"] for a in data["asignaturas"]}

    chunks = []
    for asig in data["asignaturas"]:
        text = build_chunk_text(asig, asignaturas_por_numero, data["carrera"], data["plan"])
        chunks.append({
            "id": f"{stem}_correlativa_{asig['n']:02d}",
            "source": data["fuente"],
            "asignatura": asig["nombre"],
            "carrera": data["carrera"],
            "text": text,
        })

    year_chunks = build_year_summary_chunks(data, stem)
    chunks.extend(year_chunks)

    output_path = output_dir / f"chunks_{stem}.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    return len(data["asignaturas"]), len(year_chunks), output_path


def main():
    parser = argparse.ArgumentParser(
        description="Genera chunks de texto natural desde los JSON de correlatividades (uno por carrera)"
    )
    parser.add_argument(
        "--input_json", nargs="+", required=True,
        help="Uno o más archivos JSON estructurados de correlatividades"
    )
    parser.add_argument(
        "--output_dir", default="data/processed",
        help="Directorio donde se guardan los archivos de chunks generados"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_asignaturas = 0
    total_anios = 0
    output_paths = []

    for file_path in args.input_json:
        p = Path(file_path)
        if not p.is_file():
            print(f"Advertencia: no se encontró el archivo {file_path}")
            continue

        n_asig, n_anio, out = process_one(p, output_dir)
        total_asignaturas += n_asig
        total_anios += n_anio
        output_paths.append(out)
        print(f"  {p.name}: {n_asig} asignaturas + {n_anio} resúmenes por año -> {out}")

    print(
        f"\nTotal: {total_asignaturas} chunks por asignatura "
        f"+ {total_anios} chunks-resumen por año "
        f"en {len(output_paths)} archivos."
    )


if __name__ == "__main__":
    main()