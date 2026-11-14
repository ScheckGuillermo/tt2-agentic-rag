"""Memoria del agente.

Responsable: Integrante C.

- Corto plazo: historial de cada conversación, con un checkpointer de LangGraph
  persistido en SQLite (MEMORY_DB_PATH). Cada conversación se identifica con su
  `thread_id`: son independientes y sobreviven a reinicios.
- Largo plazo: hechos y preferencias del usuario compartidos entre
  conversaciones. Al cierre de cada turno, `LongTermMemory.save_from_turn`
  le pide al LLM que extraiga hechos relevantes, resuelve contradicciones con
  los hechos previos y los guarda en un índice de Pinecone filtrable por
  `user_id`. El agente los consulta con la tool `get_user_memory`.

`agent` recibe el checkpointer y el store al compilar el grafo.
"""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool

from tt2_agentic_rag.ingest import get_vectorstore

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.store.base import BaseStore

MEMORY_INDEX_NAME = "user-memory"
DEFAULT_USER_ID = "default_user"

EXTRACTION_PROMPT = """Analiza la siguiente interacción entre un usuario y un asistente.
Identifica y extrae exclusivamente hechos clave, preferencias, gustos o datos personales sobre el usuario.
- Si no hay información relevante sobre el usuario o el mensaje es trivial, responde exactamente: 'NO_INFO'.
- Si hay información útil, resúmela en una o dos oraciones declarativas concisas en tercera persona.

Interacción:
{conversation}
"""

CONFLICT_PROMPT = """Tienes un hecho nuevo sobre un usuario y una lista de hechos previamente guardados.
Determina qué hechos viejos quedan obsoletos o contradicen directamente al hecho nuevo.

Hecho nuevo:
"{new_fact}"

Hechos guardados:
{saved_facts}

Instrucciones:
- Si algún hecho viejo contradice o es reemplazado por el nuevo, responde únicamente con los índices numéricos correspondientes separados por coma (ejemplo: '1' o '1, 3').
- Si ningún hecho viejo entra en contradicción con el nuevo, responde exactamente: 'NONE'.
"""


def _connect(db_path: Path) -> sqlite3.Connection:
    # Autocommit (isolation_level=None): sin él, SqliteStore deja una
    # transacción abierta y bloquea al checkpointer que comparte el archivo.
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path, check_same_thread=False, isolation_level=None)


def get_checkpointer(db_path: Path) -> BaseCheckpointSaver:
    """Devuelve el checkpointer persistente para la memoria a corto plazo."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    return SqliteSaver(_connect(db_path))


def get_store(db_path: Path) -> BaseStore:
    """Devuelve el store persistente de LangGraph (clave-valor compartido)."""
    from langgraph.store.sqlite import SqliteStore

    store = SqliteStore(_connect(db_path))
    store.setup()
    return store


def _user_id(config: RunnableConfig | None) -> str:
    return (config or {}).get("configurable", {}).get("user_id", DEFAULT_USER_ID)


class LongTermMemory:
    """Memoria semántica a largo plazo por usuario, persistida en Pinecone."""

    def __init__(self, llm: BaseChatModel, index_name: str = MEMORY_INDEX_NAME, k: int = 5):
        self.llm = llm
        self.k = k
        self.vectorstore = get_vectorstore(index_name)

    def search(self, query: str, user_id: str) -> list[Document]:
        """Hechos guardados del usuario más parecidos a `query`."""
        return self.vectorstore.similarity_search(query, k=self.k, filter={"user_id": user_id})

    def save_from_turn(self, messages: list[BaseMessage], user_id: str = DEFAULT_USER_ID) -> str | None:
        """Extrae hechos del turno y actualiza la memoria resolviendo contradicciones.

        Devuelve el hecho guardado, o None si el turno no aportó información.
        """
        formatted = [
            f"{'Usuario' if isinstance(m, HumanMessage) else 'Asistente'}: {m.content}"
            for m in messages
            if getattr(m, "content", None)
        ]
        if not formatted:
            return None

        prompt = EXTRACTION_PROMPT.format(conversation="\n".join(formatted))
        new_fact = str(self.llm.invoke([HumanMessage(content=prompt)]).content).strip()
        if not new_fact or "NO_INFO" in new_fact:
            return None

        previous = [d for d in self.search(new_fact, user_id) if d.metadata.get("id")]
        if previous:
            saved = "\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(previous))
            prompt = CONFLICT_PROMPT.format(new_fact=new_fact, saved_facts=saved)
            conflict = str(self.llm.invoke([HumanMessage(content=prompt)]).content).strip()
            if "NONE" not in conflict:
                indices = [int(i.strip()) - 1 for i in conflict.split(",") if i.strip().isdigit()]
                stale = [previous[i].metadata["id"] for i in indices if 0 <= i < len(previous)]
                if stale:
                    self.vectorstore.delete(ids=stale)

        fact_id = str(uuid.uuid4())
        doc = Document(
            page_content=new_fact,
            metadata={"id": fact_id, "user_id": user_id, "created_at": datetime.now().isoformat()},
        )
        self.vectorstore.add_documents([doc], ids=[fact_id])
        print(f"  [Memoria LTM guardada para '{user_id}']: {new_fact}")
        return new_fact

    def as_tool(self) -> BaseTool:
        """Tool `get_user_memory`; toma el `user_id` del config del grafo."""
        memory = self

        @tool
        def get_user_memory(query: str, config: RunnableConfig) -> str:
            """Busca y recupera hechos, datos personales o preferencias guardadas sobre el usuario en conversaciones anteriores."""
            docs = memory.search(query, _user_id(config))
            if not docs:
                return "No hay recuerdos previos guardados para este usuario."
            return "\n".join(f"- {d.page_content}" for d in docs)

        return get_user_memory
