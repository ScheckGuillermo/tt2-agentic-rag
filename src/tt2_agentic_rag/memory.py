"""Memoria del agente.

Responsable: Integrante C.

- Corto plazo: historial de cada conversación. Se implementa con un checkpointer
  de LangGraph persistido en disco (SQLite en MEMORY_DB_PATH); cada conversación
  se identifica con su `thread_id`, de modo que las conversaciones son
  independientes y sobreviven a reinicios.
- Largo plazo: información compartida entre conversaciones (por ejemplo datos
  o preferencias del usuario), mediante un store de LangGraph persistente.

`agent` recibe ambos objetos al compilar el grafo.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.store.base import BaseStore


def get_checkpointer(db_path: Path) -> BaseCheckpointSaver:
    """Devuelve el checkpointer persistente para la memoria a corto plazo."""
    raise NotImplementedError


def get_store(db_path: Path) -> BaseStore:
    """Devuelve el store persistente para la memoria a largo plazo."""
    raise NotImplementedError
