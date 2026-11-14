"""Acceso al dataset de Kaggle: Netflix Titles.

Responsable: Integrante C.

Preparación del dataset para búsqueda semántica:

- Los nulos de las columnas de texto se reemplazan por cadenas vacías para no
  romper la concatenación; se colapsan espacios y saltos de línea repetidos sin
  tocar el vocabulario.
- Las partes que definen la obra (título, tipo, géneros, director, elenco y
  descripción) se unen en un solo texto con etiquetas simples ("Title: ...")
  para que el modelo de embeddings distinga cada dato.
- País, año y tipo se guardan como metadatos, para filtrar sin "ensuciar" la
  búsqueda conceptual.

Cada título se indexa en Pinecone con su `show_id` como ID. La tool
`query_netflix_catalog` devuelve pocos registros formateados, nunca el
DataFrame completo.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.tools import tool

from tt2_agentic_rag.ingest import as_int, clean_text, index_documents
from tt2_agentic_rag.ingest import get_vectorstore as _get_vectorstore

if TYPE_CHECKING:
    import pandas as pd
    from langchain_pinecone import PineconeVectorStore

NETFLIX_INDEX_NAME = "netflix-catalog"
DEFAULT_K = 4
TEXT_COLUMNS = ["title", "type", "director", "cast", "listed_in", "description"]


def load_dataset(path: Path) -> pd.DataFrame:
    """Lee el dataset desde disco."""
    import pandas as pd

    return pd.read_csv(path)


def prepare_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Limpia el dataset y arma la columna `text_for_embedding`."""
    df = df.copy()
    for col in TEXT_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna("").apply(clean_text)

    df["text_for_embedding"] = (
        "Title: " + df["title"] + ". "
        + "Type: " + df["type"] + ". "
        + "Genres: " + df["listed_in"] + ". "
        + "Director: " + df["director"] + ". "
        + "Cast: " + df["cast"] + ". "
        + "Description: " + df["description"]
    ).str.strip()

    df["country"] = df["country"].fillna("Unknown")
    df["release_year"] = df["release_year"].fillna(0).astype(int)
    return df


def to_documents(df: pd.DataFrame) -> list[Document]:
    """Un `Document` por título: texto para embeddings + metadatos filtrables."""
    return [
        Document(
            page_content=row["text_for_embedding"],
            metadata={
                "show_id": str(row.get("show_id", idx)),
                "title": str(row["title"]),
                "type": str(row["type"]),
                "country": str(row["country"]),
                "release_year": int(row["release_year"]),
                "source_type": "kaggle_netflix",
            },
        )
        for idx, row in df.iterrows()
    ]


def build_index(
    df: pd.DataFrame,
    index_name: str = NETFLIX_INDEX_NAME,
    *,
    force: bool = False,
) -> PineconeVectorStore:
    """Indexa el catálogo (ya preparado) salvo que el índice tenga vectores."""
    documents = to_documents(df)
    ids = [doc.metadata["show_id"] for doc in documents]
    return index_documents(documents, ids, index_name, force=force)


@lru_cache(maxsize=1)
def get_vectorstore() -> PineconeVectorStore:
    """Vector store del catálogo de Netflix (conexión reutilizada)."""
    return _get_vectorstore(NETFLIX_INDEX_NAME)


def search(query: str, k: int = DEFAULT_K, filter: dict | None = None) -> list[Document]:
    """Búsqueda semántica, opcionalmente filtrada por metadatos (país, tipo, año)."""
    return get_vectorstore().similarity_search(query, k=k, filter=filter)


def format_netflix_results(docs: list[Document]) -> str:
    """Formatea los registros con título, tipo, año y país para el LLM."""
    if not docs:
        return "No se encontraron títulos relacionados en el catálogo de Netflix."
    return "\n---\n".join(
        f"[Registro Netflix | Título: {d.metadata.get('title')} ({d.metadata.get('type')}) | "
        f"Año: {as_int(d.metadata.get('release_year'))} | País: {d.metadata.get('country')}]\n"
        f"{d.page_content}"
        for d in docs
    )


@tool
def query_netflix_catalog(query: str) -> str:
    """Consulta el catálogo de Netflix sobre películas, series, actores, directores, géneros o sinopsis."""
    return format_netflix_results(search(query))
