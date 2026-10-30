"""Agentic RAG para el obligatorio de Taller de Tecnologías 2 (ORT 2026).

Módulos:
    ingest   -- PDFs -> chunks -> embeddings -> índice en Chroma.
    rag      -- búsqueda semántica sobre el índice.
    dataset  -- carga y consultas del dataset de Kaggle.
    agent    -- grafo LangGraph construido a mano (nodos, transiciones, tools).
    memory   -- memoria a corto plazo (por conversación) y a largo plazo (compartida).
"""

__version__ = "0.1.0"
