"""Recuperación semántica sobre el índice de papers.

Responsable: Integrante A.

Se conecta al índice de Pinecone creado por `ingest` y devuelve los fragmentos
más relevantes para una consulta, junto con sus metadatos de origen. La tool
`query_research_papers` es la que `agent` expone; el agente decide cuándo
invocarla.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from langchain_core.tools import tool

from tt2_agentic_rag.ingest import PAPERS_INDEX_NAME, as_int
from tt2_agentic_rag.ingest import get_vectorstore as _get_vectorstore

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_pinecone import PineconeVectorStore

DEFAULT_K = 4


@lru_cache(maxsize=1)
def get_vectorstore() -> PineconeVectorStore:
    """Vector store de los papers (conexión reutilizada)."""
    return _get_vectorstore(PAPERS_INDEX_NAME)


def search(query: str, k: int = DEFAULT_K) -> list[Document]:
    """Devuelve los `k` chunks más similares a `query`, con sus metadatos."""
    return get_vectorstore().similarity_search(query, k=k)


def format_paper_results(docs: list[Document]) -> str:
    """Formatea los chunks con su fuente (documento y página) para el LLM."""
    if not docs:
        return "No se encontró información relevante en los papers académicos."
    return "\n---\n".join(
        f"[Documento: {d.metadata.get('file_name')} | Página: {as_int(d.metadata.get('page'))}]\n"
        f"{d.page_content}"
        for d in docs
    )


@tool
def query_research_papers(query: str) -> str:
    """Consulta la colección de 6 papers académicos/científicos sobre conceptos teóricos, modelos, metodologías o resultados."""
    return format_paper_results(search(query))
