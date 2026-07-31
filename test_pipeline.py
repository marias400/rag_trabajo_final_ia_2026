import sys
from pathlib import Path
sys.path.insert(0, str(Path("./scripts").resolve()))

import rag_core
import time
from unittest.mock import patch

# Mock responses for LMStudio to avoid timeout issues during testing
def mock_lmstudio_chat(messages, base_url, model, temperature, timeout, max_tokens=None):
    prompt_str = str(messages)
    
    # Mocking the Router Response
    if "ROUTER_SYSTEM_PROMPT" in prompt_str or "RETRIEVAL: SI" in prompt_str or "Sos el enrutador" in prompt_str:
        return "RETRIEVAL: SI\nPREGUNTA: correlativas de Cálculo Numérico Ingeniería en Sistemas\nESTRATEGIA: multi_query\nRAZON: test multi_query"
        
    # Mocking Multi-Query Response
    if "Generá 5 formas diferentes de preguntar lo mismo" in prompt_str:
        return "¿Cuáles son las materias previas de Cálculo Numérico?\n¿Qué necesito aprobar para Cálculo Numérico?\nrequisitos Cálculo Numérico\nmaterias previas Cálculo Numérico\n¿Qué materias son prerequisito de Cálculo Numérico?"
        
    # Mocking HyDE Response
    if "RESPUESTA HIPOTÉTICA:" in prompt_str:
        return "Para cursar Cálculo Numérico se requiere tener aprobada Análisis Matemático II."
        
    # Mocking Query Decomposition Response
    if "sub-preguntas simples" in prompt_str:
        return "¿Qué correlativas tiene Cálculo Numérico?\n¿Cuáles son las materias de Sistemas?"
        
    return "Mocked response from LLM."


@patch("rag_core.call_lmstudio_chat", side_effect=mock_lmstudio_chat)
def main(mock_chat):
    print("=== TEST PIPELINE RAG (MOCKED LLM) ===")
    history = []
    question = "¿Qué correlativas tiene Cálculo Numérico en Sistemas?"
    base_url = "http://localhost:1234"
    model = "local-model"
    timeout = 30
    
    print("\n1. Testing route_query (Router)...")
    try:
        t0 = time.time()
        necesita_retrieval, pregunta_busqueda, strategy, reason = rag_core.route_query(
            history, question, base_url, model, timeout
        )
        t1 = time.time()
        print(f"Time: {t1-t0:.2f}s")
        print(f"Necesita retrieval: {necesita_retrieval}")
        print(f"Pregunta busqueda: {pregunta_busqueda}")
        print(f"Strategy: {strategy}")
        print(f"Reason: {reason}")
    except Exception as e:
        print(f"ERROR in route_query: {e}")
        return

    print("\n2. Testing Query Enhancement (Multi-Query)...")
    try:
        t0 = time.time()
        variants = rag_core.generate_multi_queries(pregunta_busqueda, base_url, model, timeout)
        t1 = time.time()
        print(f"Time: {t1-t0:.2f}s")
        print(f"Variants: {variants}")
    except Exception as e:
        print(f"ERROR in Multi-Query: {e}")
        return

    print("\n3. Testing Retrieval (Hybrid + RRF + Reranker)...")
    db_path = "./chroma_db"
    collection_name = "curriculum"
    embedding_model = "paraphrase-multilingual-MiniLM-L12-v2"
    reranker = "BAAI/bge-reranker-v2-m3"
    
    try:
        t0 = time.time()
        # using the multi-query variants for testing retrieve_with_multi_query
        all_queries = [pregunta_busqueda] + variants
        
        print("  - Running retrieve_with_multi_query (this triggers BM25 + Semantic + RRF internally)...")
        results = rag_core.retrieve_with_multi_query(
            queries=all_queries,
            db_path=db_path,
            collection_name=collection_name,
            model_name=embedding_model,
            n_results=5,
            use_hybrid=True,
            reranker_model=reranker
        )
        t1 = time.time()
        print(f"Time: {t1-t0:.2f}s")
        print(f"Retrieved {len(results)} chunks after Reranking:")
        for i, (doc, meta, score) in enumerate(results):
            print(f"  [{i}] Score: {score:.4f} | Source: {meta.get('source')} (Page {meta.get('page')})")
            print(f"      {doc[:100]}...")
    except Exception as e:
        print(f"ERROR in Retrieval: {e}")
        import traceback
        traceback.print_exc()
        return
        
    print("\n=== PIPELINE OK ===")

if __name__ == '__main__':
    main()
