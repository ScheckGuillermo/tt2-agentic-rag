"""Punto de entrada para la interfaz web Gradio (ejecución local)."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_groq import ChatGroq

from tt2_agentic_rag.agent import build_agent
from tt2_agentic_rag.dataset import query_netflix_catalog
from tt2_agentic_rag.memory import LongTermMemory, get_checkpointer, get_store
from tt2_agentic_rag.rag import query_research_papers
from tt2_agentic_rag.ui import create_demo

ROOT = Path(__file__).resolve().parent


def main() -> None:
    load_dotenv(ROOT / ".env")

    for key in ("GROQ_API_KEY", "PINECONE_API_KEY"):
        if not os.getenv(key):
            raise ValueError(f"Falta {key} en .env (ver .env.example).")

    memory_db_path = ROOT / (os.getenv("MEMORY_DB_PATH") or "storage/memory.sqlite")
    llm_model = os.getenv("LLM_MODEL") or "qwen/qwen3.8-27b"

    llm = ChatGroq(model=llm_model, temperature=0, max_tokens=500)
    checkpointer = get_checkpointer(memory_db_path)
    store = get_store(memory_db_path)
    long_term_memory = LongTermMemory(llm)

    agent = build_agent(
        llm=llm,
        tools=[query_netflix_catalog, query_research_papers, long_term_memory.as_tool()],
        checkpointer=checkpointer,
        store=store,
    )

    demo = create_demo(
        agent,
        save_memory_fn=lambda msgs, uid: long_term_memory.save_from_turn(msgs, user_id=uid),
    )
    demo.launch()


if __name__ == "__main__":
    main()
