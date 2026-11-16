"""Agentic RAG para el obligatorio de Taller de Tecnologías 2 (ORT 2026).

Módulos:
    ingest   -- PDFs -> chunks -> embeddings -> índice en Pinecone (+ utilidades compartidas).
    rag      -- búsqueda semántica sobre los papers y tool `query_research_papers`.
    dataset  -- dataset de Kaggle (Netflix Titles) y tool `query_netflix_catalog`.
    agent    -- grafo LangGraph construido a mano (nodos, transiciones, tools).
    memory   -- memoria a corto plazo (SQLite, por conversación) y a largo plazo
                (hechos del usuario en Pinecone, tool `get_user_memory`).
"""

__version__ = "0.1.0"
