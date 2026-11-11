"""Pruebas del agente con un modelo simulado (sin consumir APIs externas)."""

from __future__ import annotations

import os
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END

from tt2_agentic_rag.agent import build_agent, route_after_llm


# ---------------------------------------------------------------------------
# Dobles de prueba
# ---------------------------------------------------------------------------
class ScriptedChatModel(BaseChatModel):
    """Modelo falso: devuelve respuestas guionadas y registra lo que recibe."""

    responses: list[AIMessage]
    calls: list[list[BaseMessage]] = []
    bound_tool_names: list[str] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> ScriptedChatModel:
        self.bound_tool_names = [t.name for t in tools]
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.calls.append(list(messages))
        if not self.responses:
            raise AssertionError("El modelo fue invocado más veces de lo esperado")
        return ChatResult(generations=[ChatGeneration(message=self.responses.pop(0))])


def call(name: str, call_id: str, **args: Any) -> dict[str, Any]:
    return {"name": name, "args": args, "id": call_id, "type": "tool_call"}


def ai(content: str = "", *tool_calls: dict[str, Any]) -> AIMessage:
    return AIMessage(content=content, tool_calls=list(tool_calls))


@tool
def add(a: int, b: int) -> int:
    """Suma dos enteros."""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiplica dos enteros."""
    return a * b


@tool
def boom(x: str) -> str:
    """Herramienta que siempre falla."""
    raise RuntimeError("falla simulada")


def ask(agent, text: str, config: RunnableConfig | None = None) -> list[BaseMessage]:
    return agent.invoke({"messages": [HumanMessage(text)]}, config)["messages"]


# ---------------------------------------------------------------------------
# Pruebas
# ---------------------------------------------------------------------------
def test_responde_sin_herramientas():
    model = ScriptedChatModel(responses=[ai("¡Hola! ¿En qué te ayudo?")])
    agent = build_agent(model, [add])

    messages = ask(agent, "Hola")

    assert len(model.calls) == 1
    assert [type(m) for m in messages] == [HumanMessage, AIMessage]
    assert messages[-1].content == "¡Hola! ¿En qué te ayudo?"


def test_ejecuta_herramienta_y_devuelve_resultado_al_llm():
    model = ScriptedChatModel(responses=[ai("", call("add", "c1", a=2, b=3)), ai("Da 5.")])
    agent = build_agent(model, [add])

    messages = ask(agent, "¿Cuánto es 2+3?")

    tool_msg = messages[2]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.tool_call_id == "c1"
    assert tool_msg.content == "5"
    # La segunda llamada al LLM recibe el resultado de la herramienta.
    assert len(model.calls) == 2
    assert model.calls[1][-1] == tool_msg
    assert messages[-1].content == "Da 5."


def test_multiples_herramientas_en_una_respuesta():
    model = ScriptedChatModel(
        responses=[
            ai("", call("add", "c1", a=1, b=2), call("multiply", "c2", a=3, b=4)),
            ai("3 y 12."),
        ]
    )
    agent = build_agent(model, [add, multiply])

    messages = ask(agent, "Sumá 1+2 y multiplicá 3*4")

    tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
    assert [(m.tool_call_id, m.content) for m in tool_msgs] == [("c1", "3"), ("c2", "12")]
    assert len(model.calls) == 2


def test_varias_iteraciones_llm_tools_llm():
    model = ScriptedChatModel(
        responses=[
            ai("", call("add", "c1", a=2, b=3)),
            ai("", call("multiply", "c2", a=5, b=10)),
            ai("(2+3)*10 = 50"),
        ]
    )
    agent = build_agent(model, [add, multiply])

    messages = ask(agent, "¿(2+3)*10?")

    assert len(model.calls) == 3
    assert [type(m).__name__ for m in messages] == [
        "HumanMessage", "AIMessage", "ToolMessage", "AIMessage", "ToolMessage", "AIMessage",
    ]
    assert messages[4].content == "50"


def test_herramienta_desconocida_no_rompe_el_grafo():
    model = ScriptedChatModel(responses=[ai("", call("no_existe", "c1")), ai("No pude.")])
    agent = build_agent(model, [add])

    messages = ask(agent, "Usá una herramienta rara")

    tool_msg = messages[2]
    assert isinstance(tool_msg, ToolMessage)
    assert tool_msg.status == "error"
    assert tool_msg.tool_call_id == "c1"
    assert "no_existe" in tool_msg.content and "add" in tool_msg.content
    assert messages[-1].content == "No pude."


def test_excepcion_en_herramienta_se_informa_al_llm():
    model = ScriptedChatModel(responses=[ai("", call("boom", "c1", x="a")), ai("Hubo un error.")])
    agent = build_agent(model, [boom])

    messages = ask(agent, "Probá boom")

    tool_msg = messages[2]
    assert tool_msg.status == "error"
    assert "falla simulada" in tool_msg.content
    assert model.calls[1][-1] == tool_msg
    assert messages[-1].content == "Hubo un error."


def test_argumentos_invalidos_se_informan_como_error():
    model = ScriptedChatModel(responses=[ai("", call("add", "c1", a="x")), ai("Error de argumentos.")])
    agent = build_agent(model, [add])

    messages = ask(agent, "Sumá algo")

    assert messages[2].status == "error"
    assert messages[-1].content == "Error de argumentos."


def test_route_after_llm():
    assert route_after_llm({"messages": [ai("final")]}) == END
    assert route_after_llm({"messages": [ai("", call("add", "c1", a=1, b=1))]}) == "tools"
    assert route_after_llm({"messages": [HumanMessage("hola")]}) == END


def test_finaliza_cuando_no_hay_llamadas_pendientes():
    model = ScriptedChatModel(responses=[ai("", call("add", "c1", a=1, b=1)), ai("2")])
    agent = build_agent(model, [add])

    messages = ask(agent, "1+1")

    assert isinstance(messages[-1], AIMessage)
    assert not messages[-1].tool_calls
    assert model.responses == []  # no hubo llamadas extra


def test_limite_de_iteraciones_fuerza_respuesta_final():
    looping = [ai("", call("add", f"c{i}", a=1, b=1)) for i in range(2)]
    model = ScriptedChatModel(responses=[*looping, ai("Respuesta parcial.")])
    agent = build_agent(model, [add], max_iterations=3)

    messages = ask(agent, "Loop")

    assert len(model.calls) == 3
    # La última llamada recibe el aviso de límite y termina sin tool_calls.
    assert isinstance(model.calls[-1][-1], SystemMessage)
    assert "límite" in model.calls[-1][-1].content
    assert messages[-1].content == "Respuesta parcial."


def test_system_prompt_se_antepone_y_no_se_guarda():
    model = ScriptedChatModel(responses=[ai("ok")])
    agent = build_agent(model, [add], system_prompt="SOS UN TEST")

    messages = ask(agent, "Hola")

    assert isinstance(model.calls[0][0], SystemMessage)
    assert model.calls[0][0].content == "SOS UN TEST"
    assert not any(isinstance(m, SystemMessage) for m in messages)


def test_tools_se_reciben_dinamicamente():
    def resta(a: int, b: int) -> int:
        """Resta b de a."""
        return a - b

    model = ScriptedChatModel(responses=[ai("", call("resta", "c1", a=5, b=2)), ai("3")])
    agent = build_agent(model, [resta])

    messages = ask(agent, "5-2")

    assert model.bound_tool_names == ["resta"]
    assert messages[2].content == "3"


def test_nombres_duplicados_fallan():
    with pytest.raises(ValueError):
        build_agent(ScriptedChatModel(responses=[]), [add, add])


def test_config_llega_a_la_herramienta():
    @tool
    def whoami(config: RunnableConfig) -> str:
        """Devuelve el user_id de la configuración."""
        return config["configurable"]["user_id"]

    model = ScriptedChatModel(responses=[ai("", call("whoami", "c1")), ai("listo")])
    agent = build_agent(model, [whoami])

    messages = ask(agent, "¿quién soy?", {"configurable": {"user_id": "u-42"}})

    assert messages[2].content == "u-42"


def test_checkpointer_separa_conversaciones_por_thread_id():
    model = ScriptedChatModel(responses=[ai("r1"), ai("r2"), ai("r3")])
    agent = build_agent(model, [add], checkpointer=InMemorySaver())
    thread_a = {"configurable": {"thread_id": "a"}}
    thread_b = {"configurable": {"thread_id": "b"}}

    ask(agent, "mensaje A1", thread_a)
    ask(agent, "mensaje B1", thread_b)
    messages_a = ask(agent, "mensaje A2", thread_a)

    assert [m.content for m in messages_a] == ["mensaje A1", "r1", "mensaje A2", "r3"]
    # El modelo, en la tercera llamada, no ve nada del hilo B.
    assert all("B1" not in str(m.content) for m in model.calls[2])


# ---------------------------------------------------------------------------
# Prueba opcional con un modelo real (no corre por defecto)
# ---------------------------------------------------------------------------
@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("RUN_LLM_TESTS") != "1" or not os.getenv("GROQ_API_KEY"),
    reason="Requiere RUN_LLM_TESTS=1 y GROQ_API_KEY",
)
def test_modelo_real_groq():
    langchain_groq = pytest.importorskip("langchain_groq")
    llm = langchain_groq.ChatGroq(
        model=os.getenv("LLM_MODEL") or "qwen/qwen3.8-27b", temperature=0, max_tokens=300
    )
    agent = build_agent(llm, [multiply])

    messages = ask(agent, "Usá la herramienta para calcular 12 por 7.")

    assert any(isinstance(m, ToolMessage) and m.content == "84" for m in messages)
    assert "84" in messages[-1].content
