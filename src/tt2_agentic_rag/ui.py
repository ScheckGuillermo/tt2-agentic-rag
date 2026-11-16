"""Interfaz web opcional con Gradio para el agente Agentic RAG.

Expone `create_demo`, reutilizable tanto desde `app.py` como desde
`notebooks/obligatorio.ipynb`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, Any

import gradio as gr
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph


SaveMemoryFn = Callable[[Sequence[BaseMessage], str], Any]


def _clean_user_id(user_id: str | None, default: str) -> str:
    return (user_id or default).strip() or default


def _default_thread(user_id: str) -> str:
    return f"hilo-1-{user_id}"


def _load_thread_history(agent: CompiledStateGraph, thread_id: str) -> list[dict[str, str]]:
    """Reconstruye el historial de Gradio desde el checkpointer de LangGraph."""
    state = agent.get_state({"configurable": {"thread_id": thread_id}})
    messages: Sequence[BaseMessage] = state.values.get("messages", []) if state and state.values else []

    history: list[dict[str, str]] = []
    pending_tools: list[str] = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            pending_tools.clear()
            history.append({"role": "user", "content": str(msg.content)})
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                pending_tools.extend(call["name"] for call in msg.tool_calls)
            elif msg.content:
                tools_note = ", ".join(f"`{t}`" for t in pending_tools) if pending_tools else "ninguna"
                content = f"{msg.content}\n\n---\n*Herramientas usadas: {tools_note}*"
                history.append({"role": "assistant", "content": content})
                pending_tools.clear()
    return history


def create_demo(
    agent: CompiledStateGraph,
    save_memory_fn: SaveMemoryFn | None = None,
    *,
    title: str = "Agentic RAG — Papers & Netflix",
    default_user_id: str = "nicolas",
) -> gr.Blocks:
    """Construye la app Gradio conectada al grafo compilado y a la memoria."""
    initial_thread = _default_thread(default_user_id)

    def respond(
        message: str,
        history: list[dict[str, str]],
        thread_id: str,
        user_id: str,
    ) -> tuple[str, list[dict[str, str]]]:
        text = (message or "").strip()
        if not text:
            return "", history

        clean_user = _clean_user_id(user_id, default_user_id)
        clean_thread = (thread_id or _default_thread(clean_user)).strip() or _default_thread(clean_user)
        config = {"configurable": {"thread_id": clean_thread, "user_id": clean_user}}

        state = agent.get_state(config)
        prev_count = len(state.values.get("messages", [])) if state and state.values else 0

        result = agent.invoke({"messages": [HumanMessage(content=text)]}, config=config)
        new_messages = result["messages"][prev_count:]
        tools_used = [
            call["name"]
            for msg in new_messages
            if isinstance(msg, AIMessage)
            for call in msg.tool_calls
        ]

        bot_reply = result["messages"][-1]
        if save_memory_fn is not None:
            save_memory_fn([HumanMessage(content=text), bot_reply], clean_user)

        tools_note = ", ".join(f"`{t}`" for t in tools_used) if tools_used else "ninguna"
        reply_text = f"{bot_reply.content}\n\n---\n*Herramientas usadas: {tools_note}*"

        updated_history = [
            *history,
            {"role": "user", "content": text},
            {"role": "assistant", "content": reply_text},
        ]
        return "", updated_history

    def switch_user(
        user_id: str,
        threads_by_user: dict[str, list[str]],
    ) -> tuple[str, gr.Dropdown, dict[str, list[str]], list[dict[str, str]], str]:
        clean_user = _clean_user_id(user_id, default_user_id)
        updated_map = dict(threads_by_user)
        user_threads = updated_map.get(clean_user)
        if not user_threads:
            user_threads = [_default_thread(clean_user)]
            updated_map[clean_user] = user_threads

        active_thread = user_threads[-1]
        return (
            clean_user,
            gr.Dropdown(choices=user_threads, value=active_thread),
            updated_map,
            _load_thread_history(agent, active_thread),
            f"Usuario activo: **`{clean_user}`**",
        )

    def new_conversation(
        user_id: str,
        threads_by_user: dict[str, list[str]],
    ) -> tuple[gr.Dropdown, dict[str, list[str]], list[dict[str, str]]]:
        clean_user = _clean_user_id(user_id, default_user_id)
        updated_map = dict(threads_by_user)
        existing = list(updated_map.get(clean_user, []))
        next_id = f"hilo-{len(existing) + 1}-{clean_user}"
        existing.append(next_id)
        updated_map[clean_user] = existing
        return gr.Dropdown(choices=existing, value=next_id), updated_map, []

    def switch_thread(thread_id: str) -> list[dict[str, str]]:
        return _load_thread_history(agent, thread_id)

    with gr.Blocks(title=title) as demo:
        gr.Markdown(
            f"# {title}\n"
            "Consultá sobre los papers académicos, el catálogo de Netflix o mantené una conversación general."
        )

        active_user_state = gr.State(default_user_id)
        threads_by_user_state = gr.State({default_user_id: [initial_thread]})

        with gr.Group():
            with gr.Row():
                user_input = gr.Textbox(
                    label="Usuario (user_id)",
                    value=default_user_id,
                    scale=3,
                )
                switch_user_btn = gr.Button("Cambiar usuario", scale=1)
                active_user_md = gr.Markdown(f"Usuario activo: **`{default_user_id}`**")

        with gr.Row():
            thread_dropdown = gr.Dropdown(
                label="Conversaciones del usuario (thread_id)",
                choices=[initial_thread],
                value=initial_thread,
                allow_custom_value=True,
                scale=4,
            )
            new_chat_btn = gr.Button("Nueva conversación", scale=1)

        chatbot = gr.Chatbot(type="messages", height=460, label="Chat")

        with gr.Row():
            msg_box = gr.Textbox(
                placeholder="Escribí tu consulta...",
                show_label=False,
                scale=6,
            )
            send_btn = gr.Button("Enviar", variant="primary", scale=1)

        msg_box.submit(
            respond,
            inputs=[msg_box, chatbot, thread_dropdown, active_user_state],
            outputs=[msg_box, chatbot],
        )
        send_btn.click(
            respond,
            inputs=[msg_box, chatbot, thread_dropdown, active_user_state],
            outputs=[msg_box, chatbot],
        )
        user_input.submit(
            switch_user,
            inputs=[user_input, threads_by_user_state],
            outputs=[active_user_state, thread_dropdown, threads_by_user_state, chatbot, active_user_md],
        )
        switch_user_btn.click(
            switch_user,
            inputs=[user_input, threads_by_user_state],
            outputs=[active_user_state, thread_dropdown, threads_by_user_state, chatbot, active_user_md],
        )
        new_chat_btn.click(
            new_conversation,
            inputs=[active_user_state, threads_by_user_state],
            outputs=[thread_dropdown, threads_by_user_state, chatbot],
        )
        thread_dropdown.change(
            switch_thread,
            inputs=[thread_dropdown],
            outputs=[chatbot],
        )

    return demo
