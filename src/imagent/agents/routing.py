"""La tabla de enrutado del grafo.

Aqui esta la decision central del proyecto: cuando escalar al agente de vision
-que es caro- y cuando rendirse y responder con lo que haya.

`route()` es una funcion PURA. No toca el estado del grafo, no llama a ningun
modelo y no depende de LangGraph. Recibe exactamente tres cosas: lo que propuso
el coordinador, lo que queda de presupuesto y lo que ya se ha buscado en este
turno. Esa firma es deliberada: se puede probar la logica de enrutado entera sin
construir un grafo, sin proveedores y sin generar una sola palabra de prosa.

La mitad de codigo de la decision D3-C esta en este fichero. El coordinador
devuelve una accion de un enum cerrado; a partir de ahi manda esto.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from imagent.agents.budget import Budget
from imagent.domain.text import fingerprint


class Route(StrEnum):
    """A donde va el grafo despues del coordinador."""

    RETRIEVE = "retrieve"
    VISION = "vision"
    RESPOND = "respond"


class IncompleteReason(StrEnum):
    """Por que una respuesta sale marcada como incompleta.

    Cadenas estables: salen en los logs y en la respuesta HTTP, asi que
    renombrarlas rompe a quien las este leyendo.
    """

    BUDGET_ITERATIONS = "budget_iterations"
    """Se agotaron los super-steps del grafo."""

    BUDGET_VISION = "budget_vision"
    """Habia que volver a mirar, pero no quedaba presupuesto de vision."""

    NO_CANDIDATES = "no_candidates"
    """El coordinador queria mirar pero no senalo ninguna imagen."""

    NO_EVIDENCE = "no_evidence"
    """El coordinador se rinde: la evidencia no basta y no hay nada mas que hacer."""

    REPEATED_QUERY = "repeated_query"
    """El coordinador pidio repetir una busqueda ya hecha en este turno."""

    INVALID_PLAN = "invalid_plan"
    """El plan no se sostiene (buscar sin consulta, o no haber plan)."""

    RESPONSE_DEGRADED = "response_degraded"
    """No se pudo redactar la respuesta y se devolvio la evidencia en crudo.

    A diferencia de los demas, este motivo no lo pone `route()`: lo pone el nodo
    de redaccion cuando el modelo de texto le falla. Vive aqui igualmente porque
    lo que enumera este enum es "por que la respuesta no es completa", y esa es
    una de las razones."""


class CoordinatorDecision(BaseModel):
    """Lo que el coordinador devuelve como salida estructurada.

    Enum cerrado en `action`: el modelo no puede inventarse una accion, y si
    devuelve algo fuera del enum la validacion de pydantic lo rechaza antes de
    que llegue al grafo. Es la primera mitad del hibrido de D3-C; la segunda es
    `route()`, que puede ignorar lo que diga esto.
    """

    model_config = ConfigDict(frozen=True)

    action: Route
    sufficient: bool
    """Si la evidencia recogida basta para responder.

    No decide la ruta -de eso se encarga `action`-, pero distingue una respuesta
    normal de una respuesta a medias cuando la ruta ya es RESPOND.
    """

    standalone_query: str = ""
    """La pregunta con las referencias resueltas: "y de esa, que pone abajo?"
    convertido en algo que se puede buscar sin el historial delante.

    Es lo unico que el agente de recuperacion recibe de la conversacion. La
    gestion de contexto consiste sobre todo en decidir cuanto NO pasar."""

    candidate_image_ids: list[UUID] = Field(default_factory=list)
    """Que imagenes merece la pena volver a mirar, en orden de preferencia.
    El presupuesto recorta esta lista; el modelo no decide cuantas son."""

    gap: str = ""
    """Que falta, en palabras. Se le pasa al agente de vision como pregunta."""

    reasoning: str = ""
    """Para los logs. No se le enseña al usuario."""


@dataclass(frozen=True, slots=True)
class RoutingOutcome:
    """Lo que decide el router: a donde ir y con cuanto."""

    route: Route
    vision_images: int = 0
    """Cuantas imagenes quedan autorizadas, YA recortadas por el presupuesto."""

    incomplete_reason: IncompleteReason | None = None

    @property
    def incomplete(self) -> bool:
        return self.incomplete_reason is not None


def route(
    *,
    decision: CoordinatorDecision | None,
    budget: Budget,
    queries_done: Collection[str],
) -> RoutingOutcome:
    """Decide el siguiente paso del grafo.

    El ORDEN de las comprobaciones es parte del diseño: los cortes de
    presupuesto van los primeros, antes de mirar siquiera que pidio el modelo.
    Si fueran despues, una propuesta del coordinador podria colarse por delante
    de un limite, y el limite dejaria de serlo.
    """
    # 1. Presupuesto de iteraciones. Va el primero de todo: da igual lo que
    #    diga el plan, si no quedan vueltas se responde con lo que haya.
    if not budget.can_iterate:
        return RoutingOutcome(Route.RESPOND, incomplete_reason=IncompleteReason.BUDGET_ITERATIONS)

    # 2. Guarda contra un plan que no llego. No deberia pasar nunca -el nodo
    #    coordinador siempre escribe uno-, pero responder a medias es mejor que
    #    reventar el turno entero por un bug.
    if decision is None:
        return RoutingOutcome(Route.RESPOND, incomplete_reason=IncompleteReason.INVALID_PLAN)

    if decision.action is Route.RETRIEVE:
        return _route_retrieve(decision, queries_done)

    if decision.action is Route.VISION:
        return _route_vision(decision, budget)

    # 3. El coordinador quiere responder. `sufficient` distingue una respuesta
    #    normal de una en la que se rinde sabiendo que le falta informacion.
    if decision.sufficient:
        return RoutingOutcome(Route.RESPOND)
    return RoutingOutcome(Route.RESPOND, incomplete_reason=IncompleteReason.NO_EVIDENCE)


def _route_retrieve(decision: CoordinatorDecision, queries_done: Collection[str]) -> RoutingOutcome:
    """Decision D13-B: se puede volver a buscar, pero no repetir una busqueda.

    Reformular es comportamiento legitimo de un coordinador ("busca caligrafia
    en vez de escrito a mano"). Repetir la misma consulta no lo es: no puede dar
    un resultado distinto y solo quema iteraciones. Se corta aqui, en codigo, en
    vez de pedirle al prompt que no lo haga.

    La comparacion usa `fingerprint` y no `normalize`: si solo se quitaran
    tildes y mayusculas, añadir una interrogacion bastaria para saltarse el
    corte y repetir la busqueda igualmente.
    """
    consulta = fingerprint(decision.standalone_query)
    if not consulta:
        return RoutingOutcome(Route.RESPOND, incomplete_reason=IncompleteReason.INVALID_PLAN)

    if consulta in {fingerprint(q) for q in queries_done}:
        return RoutingOutcome(Route.RESPOND, incomplete_reason=IncompleteReason.REPEATED_QUERY)

    return RoutingOutcome(Route.RETRIEVE)


def _route_vision(decision: CoordinatorDecision, budget: Budget) -> RoutingOutcome:
    """La escalada, y sus dos formas de no ocurrir.

    Se distinguen a proposito: "no quedaba presupuesto" y "el coordinador no
    senalo ninguna imagen" llevan al mismo sitio pero significan cosas
    distintas, y quien lea los logs necesita saber cual de las dos fue.
    """
    if not decision.candidate_image_ids:
        return RoutingOutcome(Route.RESPOND, incomplete_reason=IncompleteReason.NO_CANDIDATES)

    if not budget.can_look:
        return RoutingOutcome(Route.RESPOND, incomplete_reason=IncompleteReason.BUDGET_VISION)

    autorizadas = budget.allowance(len(decision.candidate_image_ids))
    if autorizadas == 0:
        return RoutingOutcome(Route.RESPOND, incomplete_reason=IncompleteReason.BUDGET_VISION)

    return RoutingOutcome(Route.VISION, vision_images=autorizadas)
