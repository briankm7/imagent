"""Agente de recuperacion.

El barato. Busca sobre lo que ya esta indexado y **nunca mira una imagen**. Esa
restriccion no es una limitacion, es la razon de que este agente exista aparte:
el trabajo visual se hace una vez, en la ingesta, y este se repite en cada
pregunta.

Contrato de fallo: DEGRADA. Si el almacen no responde, la consulta sigue
adelante sin evidencia y el coordinador decidira que hacer con eso. Tumbar el
turno porque una busqueda fallo seria peor que responder "no he podido
consultar el indice".

Lo que este agente recibe de la conversacion es una sola cadena: la
`standalone_query` que el coordinador ya ha resuelto. No ve el historial. Eso es
gestion de contexto: decidir cuanto NO pasar.
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any
from uuid import UUID

from imagent.agents.state import AgentState, merge_retrieved
from imagent.config import TimeoutSettings
from imagent.domain.errors import (
    DegradationEvent,
    EmbeddingError,
    VectorStoreError,
    degrade_on,
)
from imagent.domain.models import Aspect, ImageStatus, RetrievedImage
from imagent.providers.base import ScoredPoint
from imagent.providers.registry import Providers

EVENTO_FALLO = "retrieval.failed"
EVENTO_HUERFANO = "retrieval.orphan_points"


class RetrievalAgent:
    """Nodo `retrieve` del grafo."""

    def __init__(self, providers: Providers, *, timeouts: TimeoutSettings, top_k: int) -> None:
        self._providers = providers
        self._timeouts = timeouts
        self._top_k = top_k

    async def __call__(self, state: AgentState) -> dict[str, Any]:
        decision = state["decision"]
        consulta = decision.standalone_query if decision is not None else state["question"]

        degradaciones: list[DegradationEvent] = []
        encontradas: list[RetrievedImage] = []

        with degrade_on(
            (EmbeddingError, VectorStoreError, TimeoutError),
            event=EVENTO_FALLO,
            sink=degradaciones,
        ):
            async with asyncio.timeout(self._timeouts.retrieval_seconds):
                encontradas = await self._buscar(consulta, degradaciones)

        return {
            "retrieved": merge_retrieved(state["retrieved"], encontradas),
            # La consulta se apunta AUNQUE haya fallado. Repetirla en el mismo
            # turno no puede dar un resultado distinto: fallaria igual y solo
            # habria quemado otra iteracion.
            "queries_done": [*state["queries_done"], consulta],
            "degradations": [*state["degradations"], *degradaciones],
        }

    async def _buscar(
        self, consulta: str, degradaciones: list[DegradationEvent]
    ) -> list[RetrievedImage]:
        vector = await self._providers.embeddings.embed_query(consulta)

        # Se piden mas puntos que imagenes porque cada imagen tiene hasta tres
        # (decision D2-B) y podrian casar varios de la misma. Sin este factor,
        # una sola imagen muy relevante ocuparia todo el limite.
        puntos = await self._providers.store.search(
            vector,
            limit=self._top_k * len(Aspect),
            aspects=None,
            image_ids=None,
        )

        return await self._agrupar(puntos, degradaciones)

    async def _agrupar(
        self, puntos: list[ScoredPoint], degradaciones: list[DegradationEvent]
    ) -> list[RetrievedImage]:
        """Colapsa los puntos por imagen y los completa con el registro.

        Los aspectos que casaron se conservan: no es lo mismo que una imagen
        salga por su descripcion que por su OCR, y es justo lo que el
        coordinador necesita para juzgar si la evidencia le basta.
        """
        por_imagen: dict[UUID, dict[Aspect, float]] = defaultdict(dict)
        for punto in puntos:
            anterior = por_imagen[punto.image_id].get(punto.aspect)
            if anterior is None or punto.score > anterior:
                por_imagen[punto.image_id][punto.aspect] = punto.score

        resultados: list[RetrievedImage] = []
        for image_id, aspectos in por_imagen.items():
            record = await self._providers.repository.get(image_id)

            if record is None or record.status is not ImageStatus.INDEXED:
                # Puntos sin registro detras, o de un registro que ya no esta
                # indexado: es la divergencia entre los dos almacenes hecha
                # visible. Se salta la imagen y se deja constancia en vez de
                # devolver un resultado que no se puede describir.
                degradaciones.append(
                    DegradationEvent(
                        event=EVENTO_HUERFANO,
                        detail=f"puntos sin registro indexado para {image_id}",
                        error_type="InconsistentIndex",
                    )
                )
                continue

            analisis = record.analysis
            resultados.append(
                RetrievedImage(
                    image_id=image_id,
                    filename=record.filename,
                    score=max(aspectos.values()),
                    matched_aspects=aspectos,
                    description=analisis.description if analisis else "",
                    ocr_text=analisis.ocr_text if analisis else "",
                    objects=list(analisis.objects) if analisis else [],
                )
            )

        resultados.sort(key=lambda r: (-r.score, str(r.image_id)))
        return resultados[: self._top_k]
