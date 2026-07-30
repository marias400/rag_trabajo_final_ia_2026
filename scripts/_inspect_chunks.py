#!/usr/bin/env python3
"""Extraer datos de materias de los PDFs para Lic y Mecatrónica."""
import json

with open("data/processed/chunks_pdfs.json", encoding="utf-8") as f:
    chunks = json.load(f)

# Buscar chunks con "plan de estudio" o listados de materias en la Ordenanza 235 (Lic)
print("=" * 80)
print("ORDENANZA 235 - LIC SISTEMAS - Plan de estudios / materias")
print("=" * 80)
lic = [c for c in chunks if "Ordenanza_CS_N_235" in c["source"]]
for c in lic:
    t = c["text"].lower()
    if any(kw in t for kw in ["plan de estudio", "código", "codigo", "regimen", "régimen", "lis01", "lis02", "1c", "2c", "anual", "cuatrimestre"]):
        print(f"\n--- page {c['page']} | {c['id']} ---")
        print(c["text"][:1000])
        print()

print("\n\n")
print("=" * 80)
print("ORDENANZA 233 - MECATRÓNICA - Plan de estudios / materias")
print("=" * 80)
mec = [c for c in chunks if "Ordenanza_CS_N_233" in c["source"]]
for c in mec:
    t = c["text"].lower()
    if any(kw in t for kw in ["plan de estudio", "código", "codigo", "eim01", "eim02", "1c", "2c", "anual", "cuatrimestre"]):
        print(f"\n--- page {c['page']} | {c['id']} ---")
        print(c["text"][:1000])
        print()
