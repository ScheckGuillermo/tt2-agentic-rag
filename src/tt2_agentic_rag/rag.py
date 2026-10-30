"""Recuperación semántica sobre el índice de papers.

Responsable: Integrante A.

Abre la colección de Chroma creada por `ingest` y devuelve los fragmentos más
relevantes para una consulta, junto con sus metadatos de origen. `agent` expone
esta función como herramienta; el agente decide cuándo invocarla.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document


def search(query: str, k: int = 4) -> list[Document]:
    """Devuelve los `k` chunks más similares a `query`, con sus metadatos."""
    raise NotImplementedError
