"""El grafo y la conversacion.

Topologia de radios (hub-and-spoke): una unica arista condicional, la que sale
del coordinador. `retrieve` y `vision` siempre vuelven a el y nunca hablan entre
ellos.

                            START
                              |
                              v
        +--------------> coordinator <---------------+
        |               /     |     \\                |
        |     "retrieve"   "vision"  "respond"       |
        |             /       |        \\             |
        |            v        v         v            |
        +------- retrieve   vision    respond        |
                     |         |         |           |
                     +---------+---------+           |
                               |                     |
                               +---------------------+
                                                     |
                                                   END

La alternativa -una cadena `retrieve -> vision -> respond` con saltos
condicionales entre nodos- funciona con dos agentes y se convierte en spaghetti
con el tercero, porque la logica de enrutado queda repartida por todos. Con
radios hay **un unico sitio** donde se decide y un unico sitio donde se aplica
el presupuesto.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from uuid import UUID

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from imagent.agents.budget import Budget
from imagent.agents.coordinator import CoordinatorAgent
from imagent.agents.responder import ResponderAgent
from imagent.agents.retrieval import RetrievalAgent
from imagent.agents.routing import IncompleteReason, Route
from imagent.agents.state import AgentState, new_turn
from imagent.agents.vision import VisionAgent
from imagent.config import Settings
from imagent.domain.errors import DegradationEvent
from imagent.providers.registry import Providers

NODOS = {
    Route.RETRIEVE: "retrieve",
    Route.VISION: "vision",
    Route.RESPOND: "respond",
}


def _siguiente(state: AgentState) -> str:
    """La arista condicional, entera.

    Una linea porque la decision ya la tomo el nodo coordinador llamando a
    `route()`. Una arista de LangGraph no puede escribir en el estado, asi que
    si la decision viviera aqui no habria donde dejar el recorte de imagenes ni
    el motivo de incompletitud.
    """
    return NODOS[state["next_route"]]


def build_graph(
    providers: Providers,
    *,
    settings: Settings,
    checkpointer: BaseCheckpointSaver,
) -> object:
    """Construye y compila el grafo.

    El checkpointer se inyecta, como los proveedores: en runtime es SqliteSaver
    y en los tests MemorySaver, y nada del grafo se entera de la diferencia.
    """
    builder = StateGraph(AgentState)

    builder.add_node(
        "coordinator",
        CoordinatorAgent(
            providers,
            timeouts=settings.timeouts,
            catalogue_limit=settings.coordinator_catalogue_limit,
        ),
    )
    builder.add_node(
        "retrieve",
        RetrievalAgent(providers, timeouts=settings.timeouts, top_k=settings.retrieval_top_k),
    )
    builder.add_node("vision", VisionAgent(providers, timeouts=settings.timeouts))
    builder.add_node("respond", ResponderAgent(providers, timeouts=settings.timeouts))

    builder.add_edge(START, "coordinator")
    builder.add_conditional_edges("coordinator", _siguiente, NODOS.values())
    # Los radios siempre vuelven al centro. Ningun agente decide a donde ir
    # despues de si mismo.
    builder.add_edge("retrieve", "coordinator")
    builder.add_edge("vision", "coordinator")
    builder.add_edge("respond", END)

    return builder.compile(checkpointer=checkpointer)


@asynccontextmanager
async def open_checkpointer(settings: Settings) -> AsyncIterator[BaseCheckpointSaver]:
    """El checkpointer de runtime (decision D8-B).

    SqliteSaver y no MemorySaver: con MemorySaver el historial muere al
    reiniciar y, con varios workers de uvicorn, cada worker tiene el suyo, asi
    que el usuario pierde el hilo de la conversacion segun a que worker le toque
    su siguiente mensaje. Es un bug muy desagradable de diagnosticar porque
    parece aleatorio.

    Es un context manager porque la conexion hay que cerrarla; lo gestiona el
    lifespan de la aplicacion.
    """
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    async with AsyncSqliteSaver.from_conn_string(str(settings.sqlite_path)) as saver:
        yield saver


def memory_checkpointer() -> BaseCheckpointSaver:
    """El de los tests: sin fichero, sin limpieza y sin esperas."""
    return MemorySaver()


@dataclass(frozen=True)
class TurnResult:
    """Lo que sale de un turno."""

    answer: str
    incomplete: bool
    incomplete_reason: IncompleteReason | None
    degradations: list[DegradationEvent]
    image_ids: list[UUID]
    """Las imagenes de las que ha ido la respuesta."""


class Conversation:
    """Envoltorio del grafo: un turno de conversacion.

    Existe para que la capa HTTP no tenga que saber ni construir el estado
    inicial ni poner el `recursion_limit`. Olvidar cualquiera de las dos cosas
    no falla de forma evidente, que es la peor manera de fallar.
    """

    def __init__(self, graph: object, *, settings: Settings) -> None:
        self._graph = graph
        self._settings = settings

    @property
    def _recursion_limit(self) -> int:
        """El cinturon de seguridad, no el freno.

        El freno es el presupuesto de iteraciones, que corta con elegancia y
        devuelve una respuesta marcada como incompleta. Esto lanza
        GraphRecursionError y solo deberia saltar si la contabilidad del
        presupuesto tiene un bug. Se deja con holgura suficiente para que el
        camino normal nunca lo roce.
        """
        return self._settings.budget.max_graph_iterations * 2 + 4

    async def ask(self, *, thread_id: str, question: str) -> TurnResult:
        entrada = new_turn(
            question=question,
            budget=Budget.for_turn(self._settings.budget),
        )

        final: AgentState = await self._graph.ainvoke(  # type: ignore[attr-defined]
            entrada,
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": self._recursion_limit,
            },
        )

        motivo = final.get("incomplete_reason")
        return TurnResult(
            answer=final.get("answer") or "",
            incomplete=motivo is not None,
            incomplete_reason=motivo,
            degradations=list(final.get("degradations") or []),
            image_ids=list(final.get("focus_image_ids") or []),
        )
