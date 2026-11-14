"""Ingesta de papers académicos.

Responsable: Integrante A.

Pipeline: extraer texto de los PDFs (pypdf) -> dividir en chunks -> generar
embeddings (all-MiniLM-L6-v2, local) -> indexar en un índice de Pinecone.

Cada chunk conserva metadatos para citar la fuente en las respuestas
(`file_name`, `page`); el agente los usa para fundamentar.

La ingesta se ejecuta una vez (o al agregar papers), no en cada consulta: si el
índice ya tiene vectores, `build_index` solo se conecta.

También contiene utilidades compartidas con `dataset`, `rag` y `memory`:
`get_embeddings`, `get_pinecone_index` y `clean_text`.
"""

from __future__ import annotations

import os
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
"""Modelo de embeddings. Entrenado casi solo en inglés: conviene consultar en inglés."""

EMBEDDING_DIMENSION = 384
PAPERS_INDEX_NAME = "research-papers"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120


# ---------------------------------------------------------------------------
# Utilidades compartidas
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """Devuelve el modelo de embeddings (se carga una sola vez por proceso)."""
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_pinecone_client() -> Pinecone:
    """Cliente de Pinecone autenticado con `PINECONE_API_KEY`."""
    from pinecone import Pinecone

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("Falta la variable de entorno PINECONE_API_KEY (ver .env.example).")
    return Pinecone(api_key=api_key)


def get_pinecone_index(name: str, dimension: int = EMBEDDING_DIMENSION) -> str:
    """Crea el índice serverless `name` si no existe y espera a que esté listo.

    Devuelve el nombre del índice, para pasarlo a `PineconeVectorStore`.
    """
    from pinecone import ServerlessSpec

    pc = get_pinecone_client()
    if name not in [idx.name for idx in pc.list_indexes()]:
        print(f"Creando índice '{name}' en Pinecone...")
        pc.create_index(
            name=name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(name).status["ready"]:
            time.sleep(1)
        print(f"Índice '{name}' listo.")
    return name


def index_vector_count(name: str) -> int:
    """Cantidad de vectores guardados en el índice."""
    stats = get_pinecone_client().Index(name).describe_index_stats()
    return int(stats.get("total_vector_count", 0))


def get_vectorstore(index_name: str) -> PineconeVectorStore:
    """Vector store de LangChain conectado a un índice existente (o recién creado)."""
    from langchain_pinecone import PineconeVectorStore

    get_pinecone_index(index_name)
    return PineconeVectorStore(index_name=index_name, embedding=get_embeddings())


def index_documents(
    documents: list[Document],
    ids: list[str],
    index_name: str,
    *,
    force: bool = False,
) -> PineconeVectorStore:
    """Indexa `documents` en `index_name` salvo que ya tenga vectores.

    Los `ids` deben ser deterministas: reindexar con `force=True` sobrescribe
    los mismos vectores en lugar de duplicarlos.
    """
    vectorstore = get_vectorstore(index_name)
    existing = index_vector_count(index_name)
    if existing and not force:
        print(f"El índice '{index_name}' ya tiene {existing} vectores. Conectando sin reindexar.")
        return vectorstore

    print(f"Indexando {len(documents)} documentos en '{index_name}'...")
    vectorstore.add_documents(documents, ids=ids)
    print(f"¡{len(documents)} documentos indexados en '{index_name}'!")
    return vectorstore


def as_int(value: object) -> object:
    """Pinecone devuelve los números de los metadatos como float (3.0 -> 3)."""
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def clean_text(s: object) -> str:
    """Colapsa espacios y saltos de línea repetidos (preserva el vocabulario)."""
    if s is None or not str(s).strip():
        return ""
    return re.sub(r"\s+", " ", str(s)).strip()


# ---------------------------------------------------------------------------
# Pipeline de papers
# ---------------------------------------------------------------------------


def load_pdfs(papers_dir: Path) -> list[Document]:
    """Extrae el texto de cada PDF de `papers_dir` (una página por documento)."""
    from langchain_community.document_loaders import PyPDFLoader

    raw_docs: list[Document] = []
    for pdf_path in sorted(Path(papers_dir).glob("*.pdf")):
        pages = PyPDFLoader(str(pdf_path)).load()
        for page in pages:
            page.metadata["file_name"] = pdf_path.name
            page.metadata["source_type"] = "research_paper"
        raw_docs.extend(pages)
        print(f"Cargado: {pdf_path.name} ({len(pages)} páginas)")
    return raw_docs


def split_documents(documents: list[Document]) -> list[Document]:
    """Divide en chunks, normaliza espacios y deja metadatos citables."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks: list[Document] = []
    for chunk in splitter.split_documents(documents):
        cleaned = clean_text(chunk.page_content)
        if not cleaned:
            continue
        chunks.append(
            Document(
                page_content=cleaned,
                metadata={
                    "file_name": str(chunk.metadata.get("file_name", "desconocido.pdf")),
                    # PyPDFLoader cuenta las páginas desde 0; +1 para que coincida con el PDF
                    "page": int(chunk.metadata.get("page", 0)) + 1,
                    "source_type": "research_paper",
                },
            )
        )
    return chunks


def chunk_ids(chunks: list[Document]) -> list[str]:
    """IDs deterministas `archivo:página:n` para no duplicar al reindexar."""
    counters: dict[tuple[str, int], int] = {}
    ids: list[str] = []
    for chunk in chunks:
        key = (chunk.metadata["file_name"], chunk.metadata["page"])
        n = counters.get(key, 0)
        counters[key] = n + 1
        ids.append(f"{key[0]}:{key[1]}:{n}")
    return ids


def build_index(
    papers_dir: Path,
    index_name: str = PAPERS_INDEX_NAME,
    *,
    force: bool = False,
) -> PineconeVectorStore:
    """Ejecuta el pipeline completo y devuelve el vector store de papers.

    Si el índice ya tiene vectores y `force` es False, no relee los PDFs.
    """
    if index_vector_count(get_pinecone_index(index_name)) and not force:
        return index_documents([], [], index_name)

    chunks = split_documents(load_pdfs(papers_dir))
    return index_documents(chunks, chunk_ids(chunks), index_name, force=True)
