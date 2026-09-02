"""Agente coordinador.

Es el unico que habla con el usuario y el unico que decide. Aqui esta la primera
mitad de la decision D3-C: se le pide al modelo una salida estructurada con un
enum cerrado, y **despues el codigo la valida, la recorta y decide de verdad**.

El nodo hace cuatro cosas, en este orden:

1. comprueba si le queda presupuesto de iteraciones. Si no, ni siquiera llama al
   modelo: una llamada cuyo resultado no se puede obedecer es una llamada tirada;
2. gasta una iteracion;
3. pide un plan al modelo, con degradacion a un plan de reserva determinista;
4. **sanea** el plan y llama a `route()`, que puede ignorarlo.

El paso 4 es el que hace que esto sea un sistema con limites en vez de un
sistema que le pide por favor a un modelo que se porte bien.
"""

from __future__ import annotations

import asyncio
from typing import Any

from imagent.agents import formatting
from imagent.agents.prompts import load
from imagent.agents.routing import CoordinatorDecision, IncompleteReason, Route, route
from imagent.agents.state import AgentState, focus_of
from imagent.config import TimeoutSettings
from imagent.domain.errors import (
    DegradationEvent,
    RepositoryError,
    TextError,
    degrade_on,
)
from imagent.domain.models import ImageRecord, ImageStatus
from imagent.providers.registry import Providers

EVENTO_SIN_CATALOGO = "coordinator.catalogue_failed"
EVENTO_SIN_PLAN = "coordinator.plan_failed"
EVENTO_CANDIDATAS_INVENTADAS = "coordinator.dropped_unknown_candidates"


class CoordinatorAgent:
    """Nodo `coordinator` del grafo."""

    def __init__(
        self, providers: Providers, *, timeouts: TimeoutSettings, catalogue_limit: int
    ) -> None:
        self._providers = providers
        self._timeouts = timeouts
        self._catalogue_limit = catalogue_limit

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        presupuesto = state["budget"]

        # Sin iteraciones no se llama al modelo. Pedirle un plan que no se va a
        # poder obedecer seria pagar por nada.
        if not presupuesto.can_iterate:
            return {
                "next_route": Route.RESPOND,
                "vision_allowance": 0,
                "incomplete_reason": IncompleteReason.BUDGET_ITERATIONS,
            }

        presupuesto = presupuesto.spend_iteration()

        degradaciones: list[DegradationEvent] = []

        # El catalogo se pide ANTES del plan: es lo que le permite al
        # coordinador señalar una imagen que la busqueda no le ha devuelto.
        catalogo: list[ImageRecord] = []
        with degrade_on(RepositoryError, event=EVENTO_SIN_CATALOGO, sink=degradaciones):
            catalogo = await self._providers.repository.list(statuses=[ImageStatus.INDEXED])

        decision = self._plan_de_reserva(state)
        with degrade_on((TextError, TimeoutError), event=EVENTO_SIN_PLAN, sink=degradaciones):
            async with asyncio.timeout(self._timeouts.coordinator_seconds):
                decision = await self._pedir_plan(state, presupuesto, catalogo)

        decision = self._sanear(decision, state, catalogo, degradaciones)
        resultado = route(
            decision=decision,
            budget=presupuesto,
            queries_done=state["queries_done"],
            has_evidence=bool(state["retrieved"] or state["vision_findings"]),
        )

        return {
            "budget": presupuesto,
            "decision": decision,
            "next_route": resultado.route,
            "vision_allowance": resultado.vision_images,
            "incomplete_reason": resultado.incomplete_reason,
            "degradations": [*state["degradations"], *degradaciones],
        }

    # -- El plan --------------------------------------------------------------
    async def _pedir_plan(
        self, state: AgentState, presupuesto: Any, catalogo: list[ImageRecord]
    ) -> CoordinatorDecision:
        usuario = load("coordinator_user").format(
            question=state["question"],
            catalogue=formatting.catalogo(catalogo, self._catalogue_limit),
            history=formatting.historial(state["messages"]),
            focus=formatting.identificadores(focus_of(state)),
            queries=", ".join(state["queries_done"]) or "ninguna",
            retrieved=formatting.evidencia(state["retrieved"]),
            findings=formatting.hallazgos(state["vision_findings"]),
            budget=(
                f"{presupuesto.iterations_left} pasos, "
                f"{presupuesto.vision_images_left} imagenes por mirar"
            ),
        )
        return await self._providers.text.structured(
            system=load("coordinator_system"),
            user=usuario,
            schema=CoordinatorDecision,
        )

    def _plan_de_reserva(self, state: AgentState) -> CoordinatorDecision:
        """Que hacer si el modelo de texto no responde.

        Degrada en vez de tumbar el turno: un corte pasajero en la llamada de
        planificacion no tiene por que tirar la evidencia que ya se recogio. El
        plan de reserva es determinista: buscar si aun no se ha buscado nada, y
        responder a medias si ya hay algo.
        """
        if not state["retrieved"] and not state["queries_done"]:
            return CoordinatorDecision(
                action=Route.RETRIEVE,
                sufficient=False,
                standalone_query=state["question"],
            )
        return CoordinatorDecision(action=Route.RESPOND, sufficient=False)

    def _sanear(
        self,
        decision: CoordinatorDecision,
        state: AgentState,
        catalogo: list[ImageRecord],
        degradaciones: list[DegradationEvent],
    ) -> CoordinatorDecision:
        """Recorta el plan a lo que el sistema puede sostener.

        Dos correcciones, las dos silenciosas para el modelo y ruidosas para los
        logs:

        - una `standalone_query` vacia se sustituye por la pregunta original, que
          es peor consulta pero es una consulta;
        - los identificadores de imagen que no aparecen en la evidencia se
          tiran. Un modelo puede inventarse un UUID con toda la seguridad del
          mundo, y mandar a mirar una imagen que no existe gastaria presupuesto
          para no encontrar nada.
        """
        conocidas = (
            {r.image_id for r in state["retrieved"]}
            | set(focus_of(state))
            | {r.id for r in catalogo}
        )
        validas = [i for i in decision.candidate_image_ids if i in conocidas]

        if len(validas) != len(decision.candidate_image_ids):
            degradaciones.append(
                DegradationEvent(
                    event=EVENTO_CANDIDATAS_INVENTADAS,
                    detail=(
                        f"se descartaron {len(decision.candidate_image_ids) - len(validas)} "
                        "candidatas que no estaban en la evidencia"
                    ),
                    error_type="HallucinatedCandidate",
                )
            )

        return CoordinatorDecision(
            action=decision.action,
            sufficient=decision.sufficient,
            standalone_query=decision.standalone_query.strip() or state["question"],
            candidate_image_ids=validas,
            gap=decision.gap,
            reasoning=decision.reasoning,
        )
