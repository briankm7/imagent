"""Nodo de redaccion.

Segunda mitad del agente coordinador: el que pone las palabras. Esta separado
del que decide porque asi el enrutado se puede probar entero sin generar una
sola frase, que es lo que hace barata la suite de tests de la pieza central.

Contrato de fallo: DEGRADA. Lo escribi primero como fatal, siguiendo la tabla
de contratos, y estaba mal. El argumento de "si no hay respuesta no hay nada que
devolver" solo se sostiene si la unica forma de responder es un modelo de texto,
y no lo es: la evidencia ya esta recogida y se puede listar. Un 502 aqui tira
las llamadas de vision que ya se han pagado en este mismo turno, que es
exactamente lo que el proyecto entero intenta evitar.

La respuesta de reserva no inventa nada: dice que no se ha podido redactar y
enumera lo que se encontro. Es peor que la buena y mejor que ninguna, y sale
marcada como incompleta para que nadie la confunda con una respuesta normal.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import UUID

from langchain_core.messages import AIMessage

from imagent.agents import formatting
from imagent.agents.prompts import load
from imagent.agents.routing import IncompleteReason
from imagent.agents.state import AgentState
from imagent.config import TimeoutSettings
from imagent.domain.errors import DegradationEvent, TextError, degrade_on
from imagent.providers.registry import Providers

EVENTO_SIN_REDACCION = "responder.narration_failed"


class ResponderAgent:
    """Nodo `respond` del grafo."""

    def __init__(self, providers: Providers, *, timeouts: TimeoutSettings) -> None:
        self._providers = providers
        self._timeouts = timeouts

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        motivo = state["incomplete_reason"]
        degradaciones: list[DegradationEvent] = []
        texto = ""

        with degrade_on((TextError, TimeoutError), event=EVENTO_SIN_REDACCION, sink=degradaciones):
            async with asyncio.timeout(self._timeouts.responder_seconds):
                texto = await self._redactar(state, motivo)

        if not texto:
            texto = respuesta_de_reserva(state)
            # El motivo original se conserva si ya habia uno: "se agoto el
            # presupuesto de vision" explica mas que "fallo la redaccion".
            motivo = motivo or IncompleteReason.RESPONSE_DEGRADED

        salida: dict[str, Any] = {
            "answer": texto,
            "incomplete_reason": motivo,
            # Unico canal con reducer: aqui se AÑADE al historial, no se
            # reemplaza. Es lo que hace que el turno siguiente tenga contexto.
            "messages": [AIMessage(content=texto)],
            "degradations": [*state["degradations"], *degradaciones],
        }

        # El foco de la conversacion se actualiza con las imagenes de las que
        # ha ido esta respuesta, y solo si ha ido de alguna: si el turno no
        # encontro nada, se conserva el foco anterior para que "y de esa"
        # siga significando lo mismo que antes de preguntar.
        foco = [h.image_id for h in state["vision_findings"]] or [
            r.image_id for r in state["retrieved"]
        ]
        if foco:
            salida["focus_image_ids"] = foco

        return salida

    async def _redactar(self, state: AgentState, motivo: IncompleteReason | None) -> str:
        usuario = load("responder_user").format(
            question=state["question"],
            retrieved=formatting.evidencia(state["retrieved"]),
            findings=formatting.hallazgos(state["vision_findings"]),
            status=explicar(motivo),
        )
        return await self._providers.text.complete(system=load("responder_system"), user=usuario)


def respuesta_de_reserva(state: AgentState) -> str:
    """Lo que se devuelve cuando el modelo de texto no responde.

    Enumera la evidencia y no la interpreta. Que sea fea es deliberado: tiene
    que notarse que no es la respuesta normal.
    """
    nombres: dict[UUID, str] = {r.image_id: r.filename for r in state["retrieved"]}

    lineas = ["No he podido redactar la respuesta. Esto es lo que he encontrado:"]

    for imagen in state["retrieved"]:
        lineas.append(f"- {imagen.filename}: {imagen.description}")

    for hallazgo in state["vision_findings"]:
        nombre = nombres.get(hallazgo.image_id, str(hallazgo.image_id))
        lineas.append(f"- {nombre} (al volver a mirar): {hallazgo.answer}")

    if len(lineas) == 1:
        lineas.append("- nada relevante.")

    return "\n".join(lineas)


def explicar(motivo: IncompleteReason | None) -> str:
    """Traduce el motivo a lenguaje natural para el redactor.

    Se traduce en vez de pasarle el valor del enum porque el enum es un contrato
    para logs y para la API, no texto para un modelo.
    """
    if motivo is None:
        return "completa"

    explicaciones = {
        IncompleteReason.BUDGET_ITERATIONS: (
            "incompleta: se agoto el numero de pasos permitidos antes de terminar"
        ),
        IncompleteReason.BUDGET_VISION: (
            "incompleta: habria hecho falta volver a mirar mas imagenes, "
            "pero se agoto el presupuesto para hacerlo"
        ),
        IncompleteReason.NO_CANDIDATES: (
            "incompleta: haria falta volver a mirar alguna imagen, pero no se "
            "ha podido determinar cual"
        ),
        IncompleteReason.NO_EVIDENCE: (
            "incompleta: la informacion disponible no basta para responder"
        ),
        IncompleteReason.REPEATED_QUERY: (
            "incompleta: no se ha encontrado una forma nueva de buscar lo que falta"
        ),
        IncompleteReason.INVALID_PLAN: (
            "incompleta: hubo un problema interno al planificar la busqueda"
        ),
        IncompleteReason.RESPONSE_DEGRADED: ("incompleta: no se ha podido redactar la respuesta"),
    }
    return explicaciones[motivo]
