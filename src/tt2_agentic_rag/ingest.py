"""Ingesta de papers académicos.

Responsable: Integrante A.

Pipeline: extraer texto de los PDFs (pypdf) -> dividir en chunks -> generar
embeddings con el proveedor elegido -> indexar en una colección de ChromaDB
persistida en disco (CHROMA_DIR).

Cada chunk debe conservar metadatos suficientes para citar la fuente en las
respuestas (archivo, página, etc.); el agente los usa para fundamentar.

La ingesta se ejecuta una vez (o al agregar papers), no en cada consulta.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document


def load_pdfs(papers_dir: Path) -> list[Document]:
    """Extrae el texto de cada PDF de `papers_dir`, con metadatos de origen."""
    raise NotImplementedError


def split_documents(documents: list[Document]) -> list[Document]:
    """Divide los documentos en chunks aptos para embeddings."""
    raise NotImplementedError


def build_index(papers_dir: Path, chroma_dir: Path) -> None:
    """Ejecuta el pipeline completo y persiste el índice en `chroma_dir`."""
    raise NotImplementedError
