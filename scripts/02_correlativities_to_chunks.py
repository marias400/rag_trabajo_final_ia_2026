#!/usr/bin/env python3
"""
02_correlativities_to_chunks.py
--------------------------------
Convierte el JSON estructurado y verificado de correlatividades en chunks de
texto natural (uno por asignatura), listos para vectorizar junto con el resto
de los chunks generados por 01_extract_pdfs.py.

Por qué un script aparte: la tabla de correlatividades es la parte más
consultada por los usuarios ("¿qué correlativas tiene X?") y el OCR la
destruye (mezcla columnas y pierde números). Esta tabla se transcribió a mano
desde el documento original y se guarda en formato estructurado para poder
generar texto 100% preciso, en vez de depender del OCR para el dato más
crítico del sistema.

Uso:
    python scripts/02_correlativities_to_chunks.py \
        --input_json ./data/structured/correlatividades_ing_sistemas_2024.json \
        --output_json ./data/processed/chunks_correlatividades.json

Después podés combinar este archivo con chunks.json (el de los PDFs generales)
antes de pasarlos a ChromaDB, o vectorizarlos como una colección separada.
"""

import argparse
import json


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
        f"Asignatura: {asig['nombre']} (N° {asig['n']}) de la carrera {carrera}, {plan}.{anio_txt}\n"
        f"Régimen de cursado: {asig['regimen']}.\n"
        f"Para cursar {asig['nombre']} se necesita tener REGULAR: {regular}.\n"
        f"Para rendir el examen final de {asig['nombre']} se necesita tener APROBADA: {aprobada}."
    )


ORDINAL = {1: "primer", 2: "segundo", 3: "tercer", 4: "cuarto", 5: "quinto"}


def build_year_summary_chunks(data: dict) -> list:
    """
    Genera un chunk-resumen por año (uno por cada uno de los 5 años de la carrera),
    con la cantidad y el listado completo de asignaturas de ese año.

    Por qué hace falta esto además de los chunks por asignatura: preguntas como
    "¿cuántas materias tiene el primer año?" son preguntas AGREGADAS. Ningún chunk
    de una sola asignatura puede responderlas por sí solo, y la búsqueda semántica
    no "suma" varios chunks para vos. Sin un chunk que ya contenga la respuesta
    agregada, esa pregunta queda sin datos para recuperar, sin importar cuántas
    asignaturas tengas bien transcriptas.
    """
    asignaturas = data["asignaturas"]
    if not any("anio" in a for a in asignaturas):
        return []

    por_anio = {}
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
            "id": f"correlativa_anio_{anio:02d}",
            "source": data.get("fuente_anio", data["fuente"]),
            "anio": anio,
            "text": text,
        })
    return chunks


def main():
    parser = argparse.ArgumentParser(description="Genera chunks de texto natural desde el JSON de correlatividades")
    parser.add_argument("--input_json", required=True, help="JSON estructurado de correlatividades")
    parser.add_argument("--output_json", default="correlatividades_chunks.json", help="Archivo de salida")
    args = parser.parse_args()

    with open(args.input_json, encoding="utf-8") as f:
        data = json.load(f)

    asignaturas_por_numero = {a["n"]: a["nombre"] for a in data["asignaturas"]}

    chunks = []
    for asig in data["asignaturas"]:
        text = build_chunk_text(asig, asignaturas_por_numero, data["carrera"], data["plan"])
        chunks.append({
            "id": f"correlativa_{asig['n']:02d}",
            "source": data["fuente"],
            "asignatura": asig["nombre"],
            "text": text,
        })

    year_chunks = build_year_summary_chunks(data)
    chunks.extend(year_chunks)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"Generados {len(chunks) - len(year_chunks)} chunks por asignatura "
          f"+ {len(year_chunks)} chunks-resumen por año -> {args.output_json}")
    print("\nEjemplo (chunk de asignatura):\n")
    print(chunks[0]["text"])
    if year_chunks:
        print("\nEjemplo (chunk-resumen de año):\n")
        print(year_chunks[0]["text"])


if __name__ == "__main__":
    main()