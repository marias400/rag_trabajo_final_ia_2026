#!/usr/bin/env python3
"""
05_rag_query.py
-----------------
Pipeline completo de consulta (Parte 2 del flujo): toma una pregunta,
recupera los chunks más relevantes de ChromaDB, arma el prompt con contexto
y llama a LMStudio (servidor local, API compatible con OpenAI) para generar
la respuesta final.

Requisito previo: tener LMStudio corriendo con un modelo cargado y el
servidor local iniciado (por defecto en http://localhost:1234).

Uso:
    python scripts/05_rag_query.py --query "¿Qué correlativas tiene Cálculo Numérico?"
    python scripts/05_rag_query.py --query "requisitos de admisión" --n_results 5 --show_prompt
"""

import argparse
import re
import sys

try:
    import chromadb
except ImportError:
    sys.exit("Falta chromadb. Instalalo con: pip install chromadb --break-system-packages")

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    sys.exit("Falta sentence-transformers. Instalalo con: pip install sentence-transformers --break-system-packages")

try:
    import requests
except ImportError:
    sys.exit("Falta requests. Instalalo con: pip install requests --break-system-packages")


SYSTEM_PROMPT = (
    "Sos un asistente que responde preguntas sobre el plan de estudios de "
    "Ingeniería en Sistemas de Información de la UNLaR, usando la información "
    "de los fragmentos de contexto que se te proveen.\n\n"
    "Cómo responder:\n"
    "- Basate en los fragmentos relevantes, aunque no todos coincidan entre sí "
    "o algunos no vengan al caso. Si uno solo responde la pregunta, alcanza.\n"
    "- Dá la respuesta completa: si el fragmento tiene varios datos relacionados "
    "con la pregunta (por ejemplo, requisitos para cursar Y para rendir el "
    "final), incluí todos, no solo el primero que encuentres.\n"
    "- Citá de qué fragmento sacaste cada dato (ej. 'según el Fragmento 1...'), "
    "así el usuario puede verificarlo.\n"
    "- Si un fragmento viene de un escaneo con errores de OCR (texto cortado, "
    "tablas desordenadas, palabras sueltas), ignoralo sin mencionarlo como "
    "motivo de duda sobre el resto.\n"
    "- Si de verdad ningún fragmento contiene la respuesta, decilo "
    "directamente en vez de inventar información.\n"
    "- Respondé en español, de forma natural y completa, sin ser telegráfico "
    "ni tampoco extenderte de más."
)


ORDINAL_ANIO = {
    "primer": 1, "1er": 1, "1ro": 1,
    "segundo": 2, "2do": 2,
    "tercer": 3, "tercero": 3, "3er": 3,
    "cuarto": 4, "4to": 4,
    "quinto": 5, "5to": 5,
}


def detect_anio_query(query: str) -> int | None:
    """
    Si la pregunta menciona explícitamente un año de la carrera (ej. "primer año",
    "2do año", "año 3"), devuelve el número de año (1-5). Si no, devuelve None.

    Por qué existe esto: en este corpus, la búsqueda semántica pura no separa bien
    las preguntas AGREGADAS ("¿cuántas materias tiene el primer año?") de preguntas
    sobre una asignatura puntual, porque todos los chunks son texto estructurado muy
    parecido entre sí y las distancias terminan amontonadas en un rango muy angosto
    (ver diagnóstico). El chunk-resumen correcto puede terminar en cualquier posición
    del ranking según la pregunta. Como este patrón (pregunta por año) es cerrado y
    predecible -solo 5 años posibles-, es más confiable detectarlo por palabra clave
    y forzar la inclusión del chunk-resumen correspondiente, en vez de confiar en
    que el embedding lo priorice bien.
    """
    q = query.lower()
    if "año" not in q and "anio" not in q and "ano " not in q:
        return None
    for palabra, anio in ORDINAL_ANIO.items():
        if palabra in q:
            return anio
    m = re.search(r"a[nñ]o\s*(?:n[uú]mero\s*)?(\d)\b", q)
    if m:
        return int(m.group(1))
    return None


def retrieve_chunks(query: str, db_path: str, collection_name: str, model_name: str, n_results: int):
    model = SentenceTransformer(model_name)
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_collection(collection_name)

    query_embedding = model.encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]
    retrieved = list(zip(docs, metas, distances))

    anio = detect_anio_query(query)
    if anio is not None:
        chunk_id = f"correlativa_anio_{anio:02d}"
        try:
            forced = collection.get(ids=[chunk_id])
        except Exception:
            forced = None
        if forced and forced.get("ids"):
            forced_doc = forced["documents"][0]
            forced_meta = forced["metadatas"][0]
            ya_esta = any(doc == forced_doc for doc, _meta, _dist in retrieved)
            if ya_esta:
                # Ya vino por búsqueda semántica: lo subimos al frente igual.
                retrieved = [(forced_doc, forced_meta, None)] + [
                    r for r in retrieved if r[0] != forced_doc
                ]
            else:
                # No entró por ranking semántico: lo forzamos al frente,
                # desplazando el último resultado para no crecer el contexto sin límite.
                retrieved = [(forced_doc, forced_meta, None)] + retrieved[: max(n_results - 1, 0)]

    return retrieved


def build_prompt(query: str, retrieved: list) -> str:
    context_blocks = []
    for i, (doc, meta, _dist) in enumerate(retrieved, start=1):
        fuente = meta.get("source", "desconocido")
        context_blocks.append(f"[Fragmento {i} - fuente: {fuente}]\n{doc}")
    context = "\n\n".join(context_blocks)
    return (
        f"Los fragmentos están ordenados del más al menos relevante para la pregunta "
        f"(el Fragmento 1 es el que mejor coincide).\n\n"
        f"CONTEXTO:\n{context}\n\n"
        f"PREGUNTA: {query}\n\n"
        f"Respondé la pregunta basándote en el fragmento (o fragmentos) que "
        f"realmente contengan la información, ignorando los que sean ruido "
        f"o no vengan al caso."
    )


def call_lmstudio(prompt: str, base_url: str, model: str, temperature: float, timeout: int) -> str:
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
    }
    try:
        response = requests.post(url, json=payload, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        sys.exit(
            f"No se pudo conectar a LMStudio en {url}.\n"
            f"Revisá que LMStudio esté abierto, con un modelo cargado, y el "
            f"servidor local iniciado (pestaña 'Developer' o 'Local Server')."
        )
    except requests.exceptions.ReadTimeout:
        sys.exit(
            f"LMStudio no respondió dentro de los {timeout}s.\n"
            f"Con modelos de 13B+ esto puede pasar si no hay suficiente VRAM y el "
            f"modelo corre parcialmente en CPU. Opciones:\n"
            f"  - Volvé a correr con --timeout más alto (ej. --timeout 300)\n"
            f"  - Revisá en LMStudio (pestaña Developer/Local Server) si está "
            f"generando tokens de a poco (lento) o no hay actividad (colgado)\n"
            f"  - Si está colgado, probá recargar el modelo en LMStudio"
        )
    except requests.exceptions.HTTPError as e:
        sys.exit(f"LMStudio devolvió un error: {e}\nRespuesta: {response.text[:500]}")

    data = response.json()
    return data["choices"][0]["message"]["content"]


def main():
    parser = argparse.ArgumentParser(description="Pipeline RAG completo: ChromaDB + LMStudio")
    parser.add_argument("--query", required=True, help="Pregunta del usuario")
    parser.add_argument("--db_path", default="./chroma_db", help="Carpeta donde persiste ChromaDB")
    parser.add_argument("--collection", default="curriculum", help="Nombre de la colección")
    parser.add_argument("--embedding_model", default="paraphrase-multilingual-MiniLM-L12-v2",
                         help="Debe ser el mismo modelo usado en 03_build_vectordb.py")
    parser.add_argument("--n_results", type=int, default=3, help="Cantidad de chunks a recuperar")
    parser.add_argument("--lmstudio_url", default="http://localhost:1234",
                         help="URL base del servidor local de LMStudio")
    parser.add_argument("--lmstudio_model", default="local-model",
                         help="Nombre del modelo cargado en LMStudio (Mistral 7B, LLaMA 3, etc.). "
                              "LMStudio suele aceptar cualquier string acá si hay un solo modelo cargado.")
    parser.add_argument("--temperature", type=float, default=0.35,
                         help="Temperatura del LLM. Con modelos más capaces (13B+) 0.3-0.4 da "
                              "respuestas más naturales sin perder precisión. Bajala a 0.1-0.2 "
                              "si notás que empieza a divagar o inventar datos.")
    parser.add_argument("--timeout", type=int, default=180,
                         help="Segundos a esperar la respuesta de LMStudio antes de cortar. "
                              "Con modelos de 13B+ en VRAM ajustada, subilo a 300-600.")
    parser.add_argument("--show_prompt", action="store_true",
                         help="Si se pasa, imprime el prompt completo enviado al LLM (útil para debug)")
    args = parser.parse_args()

    print("Recuperando chunks relevantes de ChromaDB...")
    retrieved = retrieve_chunks(
        args.query, args.db_path, args.collection, args.embedding_model, args.n_results
    )

    prompt = build_prompt(args.query, retrieved)
    if args.show_prompt:
        print("\n" + "=" * 80)
        print("PROMPT ENVIADO AL LLM:")
        print("=" * 80)
        print(prompt)
        print("=" * 80 + "\n")

    print(f"Consultando LMStudio en {args.lmstudio_url} (timeout: {args.timeout}s, "
          f"puede tardar bastante con modelos grandes)...\n")
    answer = call_lmstudio(prompt, args.lmstudio_url, args.lmstudio_model, args.temperature, args.timeout)

    print("=" * 80)
    print("RESPUESTA DEL LLM:")
    print("=" * 80)
    print(answer)
    print("=" * 80)

    print("\nFRAGMENTOS RECUPERADOS (mostrar también en la interfaz final):")
    print("-" * 80)
    for i, (doc, meta, dist) in enumerate(retrieved, start=1):
        fuente = meta.get("source", "?")
        pagina = f" (página {meta['page']})" if "page" in meta else ""
        dist_txt = f"{dist:.4f}" if dist is not None else "forzado (coincide palabra clave de año)"
        print(f"[{i}] {fuente}{pagina} - distancia: {dist_txt}")
        print(f"    {doc[:200]}...")
    print("-" * 80)


if __name__ == "__main__":
    main()