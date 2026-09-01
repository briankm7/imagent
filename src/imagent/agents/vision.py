"""Agente de vision bajo demanda.

El caro. Es el unico nodo que vuelve a mirar una imagen, y el unico cuyo trabajo
el presupuesto acota de verdad.

Contrato de fallo: DEGRADA, y **por imagen**. Que la segunda de tres miradas
falle no puede tirar las otras dos: se responde con lo que se consiguio y el
fallo queda registrado. Un lote que es todo o nada aqui seria pagar tres
llamadas para tirar el resultado por una.

Decision D5: siempre en lote. No hay parada temprana. Para "¿en alguna hay algo
escrito a mano?" se podria parar en el primer si, pero para "¿en cuales aparece
un coche?" hay que verlas todas, y distinguir los dos casos exige que el
coordinador clasifique la pregunta. Queda para despues.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from uuid import UUID

from imagent.agents.budget import Budget
from imagent.agents.state import AgentState
from imagent.config import TimeoutSettings
from imagent.domain.errors import (
    DegradationEvent,
    StorageError,
    VisionError,
    degrade_on,
)
from imagent.domain.models import ImageRecord, ImageStatus, VisionFinding
from imagent.providers.base import ImageBlob
from imagent.providers.registry import Providers

EVENTO_FALLO = "vision.on_demand.failed"
EVENTO_SIN_REGISTRO = "vision.on_demand.missing_record"
EVENTO_SIN_TIEMPO = "vision.on_demand.out_of_time"


class VisionAgent:
    """Nodo `vision` del grafo."""

    def __init__(self, providers: Providers, *, timeouts: TimeoutSettings) -> None:
        self._providers = providers
        self._timeouts = timeouts

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        decision = state["decision"]

        # El recorte ya lo hizo el router; aqui solo se obedece. Que este nodo
        # no vuelva a decidir cuantas imagenes mira es lo que mantiene la
        # decision en un unico sitio.
        candidatas = decision.candidate_image_ids[: state["vision_allowance"]] if decision else []
        pregunta = (decision.gap if decision and decision.gap else state["question"]).strip()

        hallazgos: list[VisionFinding] = []
        degradaciones: list[DegradationEvent] = []

        # El presupuesto se lleva actualizado imagen a imagen, no de una vez al
        # final. Es lo que hace que `time_allowance` de la tercera mirada sepa
        # lo que gastaron las dos primeras.
        restante = state["budget"]

        for indice, image_id in enumerate(candidatas):
            # El presupuesto de TIEMPO puede agotarse a mitad del lote aunque
            # queden imagenes autorizadas: se para y se deja constancia.
            if not restante.can_look:
                degradaciones.append(
                    DegradationEvent(
                        event=EVENTO_SIN_TIEMPO,
                        detail=f"quedaban {len(candidatas) - indice} imagenes por mirar",
                        error_type="BudgetExhausted",
                    )
                )
                break

            record = await self._registro(image_id, degradaciones)
            if record is None:
                # No se cobra: una candidata cuyo registro no existe no llega a
                # costar una llamada al modelo.
                continue

            inicio = time.monotonic()
            with degrade_on(
                (VisionError, StorageError, TimeoutError),
                event=EVENTO_FALLO,
                sink=degradaciones,
            ):
                hallazgos.append(await self._mirar(record, pregunta, restante))

            # Se cobra dentro o fuera del `with`: una mirada que fallo por
            # timeout ya ha consumido tiempo real y probablemente una llamada al
            # proveedor. No cobrarla convertiria un fallo en presupuesto gratis.
            restante = restante.spend_vision(images=1, seconds=time.monotonic() - inicio)

        return {
            "vision_findings": [*state["vision_findings"], *hallazgos],
            "budget": restante,
            "degradations": [*state["degradations"], *degradaciones],
        }

    async def _registro(
        self, image_id: UUID, degradaciones: list[DegradationEvent]
    ) -> ImageRecord | None:
        record = await self._providers.repository.get(image_id)
        if record is not None and record.status is ImageStatus.INDEXED:
            return record

        degradaciones.append(
            DegradationEvent(
                event=EVENTO_SIN_REGISTRO,
                detail=f"no hay registro indexado para {image_id}",
                error_type="InconsistentIndex",
            )
        )
        return None

    async def _mirar(
        self, record: ImageRecord, pregunta: str, presupuesto: Budget
    ) -> VisionFinding:
        datos = await self._providers.blobs.get_bytes(record.id)
        blob = ImageBlob(data=datos, media_type=record.media_type)

        # El timeout efectivo es el minimo entre el de una mirada y lo que quede
        # de presupuesto de tiempo: la ultima imagen del lote no puede gastar
        # mas de lo que sobra aunque su timeout individual sea mayor.
        limite = presupuesto.time_allowance(self._timeouts.vision_on_demand_seconds)
        async with asyncio.timeout(limite):
            respuesta = await self._providers.vision.answer_about(blob, pregunta)

        return VisionFinding(
            image_id=record.id,
            filename=record.filename,
            question=pregunta,
            answer=respuesta,
            model_name=self._providers.vision.model_name,
        )
