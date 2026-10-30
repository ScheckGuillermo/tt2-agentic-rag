"""Agente construido a mano con LangGraph.

Responsable: Integrante B.

Define el estado, los nodos, las aristas condicionales y las herramientas
(búsqueda en papers vía `rag`, consultas vía `dataset`). El LLM decide qué
herramientas invocar (tool calling).

Restricción del obligatorio: NO usar `create_agent()` ni `create_react_agent()`;
el grafo se arma explícitamente con `StateGraph`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.store.base import BaseStore


def build_graph(checkpointer: BaseCheckpointSaver, store: BaseStore) -> CompiledStateGraph:
    """Construye y compila el grafo del agente con la memoria de `memory`."""
    raise NotImplementedError
