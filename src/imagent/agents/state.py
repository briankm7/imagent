"""El estado del grafo, y la frontera entre lo que persiste y lo que no.

Decision D12-A: **el unico canal con reducer es `messages`**. Todo lo demas es
last-write-wins y cada nodo devuelve la lista entera ya mezclada.

La regla se lee en el tipo: si un campo lleva `Annotated[..., reducer]`, acumula
y sobrevive entre turnos; si no lo lleva, lo sobrescribe quien lo escriba y el
turno nuevo lo reinicia.

La alternativa idiomatica -reducers en todos los canales acumulativos- se
descarto por un detalle que solo se ve al usarlo: **escribir `[]` en un canal
con reducer de suma no lo vacia, lo deja igual**. Reiniciar la evidencia entre
turnos exigiria un valor centinela que ademas tendria que sobrevivir a la
serializacion del checkpointer. El precio de D12-A es que los nodos mezclan a
mano; a cambio, el reinicio entre turnos es trivial y visible.
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict
from uuid import UUID

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from imagent.agents.budget import Budget
from imagent.agents.routing import CoordinatorDecision, IncompleteReason, Route
from imagent.domain.errors import DegradationEvent
from imagent.domain.models import RetrievedImage, VisionFinding


class AgentState(TypedDict):
    """Lo que viaja por el grafo.

    Los campos estan agrupados por VIDA, no por tema: es la distincion que hay
    que tener clara para explicar como se gestiona el contexto.
    """

    # -- Persiste entre turnos ------------------------------------------------
    messages: Annotated[list[AnyMessage], add_messages]
    """El historial. Unico canal acumulativo: cada turno añade y nada borra."""

    focus_image_ids: list[UUID]
    """Sobre que imagenes va la conversacion ahora mismo.

    Es lo que hace que "y de esa, que pone abajo a la derecha?" signifique algo.
    No lleva reducer, pero persiste igual: al no escribirlo, el turno siguiente
    conserva el valor que dejo el anterior. Ese es exactamente el comportamiento
    que se quiere -se arrastra hasta que alguien lo cambia- y sale gratis.
    """

    # -- Se reinicia en cada turno --------------------------------------------
    question: str
    """La pregunta tal cual la escribio el usuario."""

    budget: Budget
    """Presupuesto del turno. Se reinicia SIEMPRE: si se arrastrase, la tercera
    pregunta de una conversacion no tendria derecho a mirar ninguna imagen."""

    queries_done: list[str]
    """Consultas ya lanzadas en este turno, para el corte de D13-B."""

    retrieved: list[RetrievedImage]
    """Evidencia barata acumulada en este turno."""

    vision_findings: list[VisionFinding]
    """Evidencia cara: lo que se descubrio volviendo a mirar."""

    degradations: list[DegradationEvent]
    """Lo que fallo y se degrado. Sale en la respuesta, no solo en los logs."""

    decision: CoordinatorDecision | None
    """Lo ultimo que propuso el coordinador."""

    next_route: Route
    """A donde va el grafo, YA decidido por el router.

    Lo escribe el nodo coordinador y la arista condicional solo lo lee. Es
    deliberado: una arista condicional de LangGraph devuelve un nombre de nodo y
    **no puede escribir en el estado**, asi que si la arista llamara a `route()`
    no habria donde dejar el recorte de imagenes ni el motivo de incompletitud.
    Calculando la ruta dentro del nodo, la arista se queda en una linea y toda
    la logica sigue siendo comprobable sin grafo.
    """

    vision_allowance: int
    """Imagenes autorizadas por el router, YA recortadas por el presupuesto."""

    incomplete_reason: IncompleteReason | None
    answer: str | None


PERSISTENT_KEYS = frozenset({"messages", "focus_image_ids"})
"""Los campos que un turno nuevo NO reinicia."""


def new_turn(*, question: str, budget: Budget) -> dict[str, Any]:
    """El estado de entrada de un turno.

    Se pasa entero a `graph.ainvoke()`: `messages` se acumula por su reducer y
    el resto sobrescribe lo que dejo el turno anterior. Asi el reinicio no
    necesita un nodo dedicado ni un centinela; es el propio input del turno.

    Que esta funcion exista y no se construya el diccionario a mano en la capa
    HTTP importa: olvidar un campo aqui significaria arrastrar evidencia de un
    turno al siguiente, y eso no falla, solo responde peor. Hay un test que
    comprueba que cubre todos los campos no persistentes.
    """
    from langchain_core.messages import HumanMessage

    return {
        "messages": [HumanMessage(content=question)],
        "question": question,
        "budget": budget,
        "queries_done": [],
        "retrieved": [],
        "vision_findings": [],
        "degradations": [],
        "decision": None,
        "next_route": Route.RESPOND,
        "vision_allowance": 0,
        "incomplete_reason": None,
        "answer": None,
    }


def focus_of(state: AgentState) -> list[UUID]:
    """El foco de la conversacion, tolerando que todavia no exista.

    En LangGraph, un canal al que **nadie ha escrito nunca** no aparece en el
    estado que recibe el nodo: no llega como None, es que la clave no esta. En
    el primer turno de una conversacion nadie ha escrito `focus_image_ids`, asi
    que leerlo con corchetes revienta con un KeyError.

    Es el precio exacto de que este campo persista sin reducer: se arrastra
    gratis entre turnos, pero el primero hay que tratarlo con cuidado. El resto
    de campos no lo necesitan porque `new_turn` los escribe todos en cada turno.
    """
    return list(state.get("focus_image_ids") or [])


def merge_retrieved(
    current: list[RetrievedImage], nuevos: list[RetrievedImage]
) -> list[RetrievedImage]:
    """Une dos tandas de evidencia por imagen, de mayor a menor puntuacion.

    Hace falta porque D13-B permite reformular y volver a buscar: la segunda
    busqueda no reemplaza a la primera, la completa. Si una imagen sale en las
    dos, se queda la puntuacion mas alta y se unen los aspectos que casaron, que
    es la informacion que el coordinador usa para juzgar si le basta.

    Es la logica que en la version con reducers viviria dentro del reducer. Al
    ser una funcion normal se puede probar sola, sin grafo.
    """
    por_imagen: dict[UUID, RetrievedImage] = {r.image_id: r for r in current}

    for nuevo in nuevos:
        anterior = por_imagen.get(nuevo.image_id)
        if anterior is None:
            por_imagen[nuevo.image_id] = nuevo
            continue

        por_imagen[nuevo.image_id] = nuevo.model_copy(
            update={
                "score": max(anterior.score, nuevo.score),
                "matched_aspects": {**anterior.matched_aspects, **nuevo.matched_aspects},
            }
        )

    # Desempate por image_id: dos imagenes con la misma puntuacion no pueden
    # ordenarse distinto en dos ejecuciones o los tests serian inestables.
    return sorted(por_imagen.values(), key=lambda r: (-r.score, str(r.image_id)))
