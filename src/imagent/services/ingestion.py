"""Pipeline de ingesta (decision D1-A).

No es un agente y no es un grafo: es una secuencia lineal sin ramificaciones.
Que llame al mismo VisionProvider que usa el agente de vision no la convierte en
uno; lo que comparten es el proveedor, que es lo unico que tiene sentido
compartir.

El orden de los pasos NO es arbitrario. Es una eleccion sobre como se rompe:

    guardar bytes -> analizar -> guardar analisis -> escribir puntos -> marcar INDEXED

Dos propiedades salen de ese orden concreto:

1. el analisis se persiste EN CUANTO existe, antes de indexar. Si el almacen de
   vectores falla despues, el registro queda FAILED pero conserva el analisis, y
   un reintento no vuelve a pagar la llamada cara de vision;
2. los puntos se escriben ANTES de marcar el registro como INDEXED. Si el
   proceso muere en medio quedan puntos huerfanos que un reintento sobrescribe,
   porque su id es determinista. Al reves tendrias un registro que afirma estar
   indexado sin puntos detras: un hueco que ninguna consulta revelaria.

Contrato de fallo: cualquier fallo aqui es FATAL para esa imagen y se propaga.
Degradar seria indexar un hueco y que el usuario creyera que su imagen esta
dentro. Pero antes de propagar se persiste el registro en FAILED con la fase, de
forma que el fallo queda visible y diagnosticable en vez de desaparecer con la
excepcion.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from imagent.config import TimeoutSettings
from imagent.domain.errors import ImagentError, InvalidImageError, VisionError
from imagent.domain.models import ImageRecord, ImageStatus, IngestionStage
from imagent.providers.base import ImageBlob
from imagent.providers.registry import Providers
from imagent.services.indexing import build_points


@dataclass(frozen=True)
class IngestionResult:
    record: ImageRecord
    deduplicated: bool
    """True si los bytes ya estaban y se devuelve el registro existente (D10-A)."""


class IngestionService:
    def __init__(self, providers: Providers, *, timeouts: TimeoutSettings) -> None:
        self._providers = providers
        self._timeouts = timeouts
        self._store_ready = False

    async def ingest(self, *, filename: str, media_type: str, data: bytes) -> IngestionResult:
        if not data:
            raise InvalidImageError("la imagen esta vacia")
        if not media_type.startswith("image/"):
            raise InvalidImageError(f"tipo no soportado: {media_type!r}")
        # El limite de tamaño NO se comprueba aqui: la capa HTTP puede rechazar
        # mientras recibe, antes de tener los bytes enteros en memoria. Aqui ya
        # es tarde para ahorrar nada.

        blob = ImageBlob(data=data, media_type=media_type)

        if (existente := await self._buscar_duplicado(blob)) is not None:
            return IngestionResult(record=existente, deduplicated=True)

        record = ImageRecord(
            content_hash=blob.content_hash,
            filename=filename,
            media_type=media_type,
            size_bytes=len(data),
        )
        await self._providers.repository.save(record)

        async with self._fase(record, IngestionStage.STORE):
            await self._providers.blobs.put(record.id, blob)

        async with self._fase(record, IngestionStage.ANALYSIS):
            async with asyncio.timeout(self._timeouts.vision_ingestion_seconds):
                analysis = await self._providers.vision.describe(blob)

            # Un analisis sin nada indexable no es un exito silencioso: la imagen
            # quedaria registrada y no la encontraria ninguna consulta. Se trata
            # como fallo de la fase de analisis, que es de donde viene.
            if not analysis.description.strip() and not analysis.ocr_text.strip():
                raise VisionError("el analisis no produjo ningun texto indexable")

        record = record.with_analysis(analysis)
        await self._providers.repository.save(record)

        async with self._fase(record, IngestionStage.EMBEDDING):
            points = await build_points(
                image_id=record.id,
                analysis=analysis,
                embeddings=self._providers.embeddings,
            )

        async with self._fase(record, IngestionStage.INDEXING):
            await self._asegurar_almacen()
            await self._providers.store.upsert(points)

        record = record.indexed()
        await self._providers.repository.save(record)
        return IngestionResult(record=record, deduplicated=False)

    async def _buscar_duplicado(self, blob: ImageBlob) -> ImageRecord | None:
        """Decision D10-A: si los bytes ya estan, se devuelve el registro existente.

        Un registro FAILED NO cuenta como duplicado. Si contase, una imagen que
        fallo al indexarse quedaria condenada: cada reintento devolveria el
        registro roto en vez de intentarlo otra vez.
        """
        existente = await self._providers.repository.find_by_hash(blob.content_hash)
        if existente is None or existente.status is ImageStatus.FAILED:
            return None
        return existente

    async def _asegurar_almacen(self) -> None:
        """`ensure_ready` una vez por instancia, no una por subida.

        Con Qdrant real seria un viaje de red por imagen. La carrera entre dos
        subidas simultaneas es inofensiva: la operacion es idempotente y lo peor
        que pasa es que se llame dos veces.
        """
        if not self._store_ready:
            await self._providers.store.ensure_ready()
            self._store_ready = True

    @asynccontextmanager
    async def _fase(self, record: ImageRecord, stage: IngestionStage) -> AsyncIterator[None]:
        """Persiste el registro como FAILED con la fase y vuelve a lanzar.

        Es la contrapartida de `degrade_on`: alli el error se traga y se sigue;
        aqui se deja constancia y se propaga. Que sean dos mecanismos distintos y
        con nombres distintos es deliberado, porque son dos contratos distintos.

        Se captura TimeoutError ademas de ImagentError porque un timeout tambien
        es un fallo de la fase, y sin el la excepcion escaparia sin dejar rastro
        en el registro.
        """
        try:
            yield
        except (ImagentError, TimeoutError) as exc:
            await self._providers.repository.save(record.failed(stage, str(exc)))
            raise
