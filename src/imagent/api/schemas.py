"""Contrato HTTP.

Separado de `domain/models.py` a proposito. Parece duplicacion y es una
frontera: lo de dentro cambia cuando haga falta, y lo de aqui es lo que ve quien
consume la API. Con los mismos modelos en ambos sitios, renombrar un campo
interno rompe el front sin que nada avise.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field

from imagent.agents.graph import TurnResult
from imagent.agents.routing import IncompleteReason
from imagent.domain.models import ImageRecord, ImageStatus, IngestionStage
from imagent.services.ingestion import IngestionResult


class ImageOut(BaseModel):
    """Una imagen, tal como la ve el front."""

    id: UUID
    filename: str
    media_type: str
    size_bytes: int
    created_at: datetime
    status: ImageStatus

    description: str = ""
    ocr_text: str = ""
    objects: list[str] = Field(default_factory=list)

    failed_stage: IngestionStage | None = None
    failure_reason: str | None = None
    """El fallo se expone, no se esconde: una imagen que no se pudo indexar tiene
    que verse en la lista, y verse por que."""

    @classmethod
    def de(cls, record: ImageRecord) -> ImageOut:
        analisis = record.analysis
        return cls(
            id=record.id,
            filename=record.filename,
            media_type=record.media_type,
            size_bytes=record.size_bytes,
            created_at=record.created_at,
            status=record.status,
            description=analisis.description if analisis else "",
            ocr_text=analisis.ocr_text if analisis else "",
            objects=list(analisis.objects) if analisis else [],
            failed_stage=record.failed_stage,
            failure_reason=record.failure_reason,
        )


class UploadOut(BaseModel):
    image: ImageOut
    deduplicated: bool
    """True si estos bytes ya estaban y se devuelve el registro que ya existia."""

    @classmethod
    def de(cls, resultado: IngestionResult) -> UploadOut:
        return cls(image=ImageOut.de(resultado.record), deduplicated=resultado.deduplicated)


class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    thread_id: str | None = None
    """Si no viene, se abre una conversacion nueva y se devuelve el id."""


class WarningOut(BaseModel):
    """Un camino degradado, tal como lo ve quien consume la API.

    Sale en la respuesta y no solo en los logs. Un fallo silencioso es peor que
    una caida ruidosa, y un fallo que solo esta en el log es silencioso para
    todo el que no tenga acceso al log.
    """

    event: str
    detail: str


class ChatOut(BaseModel):
    thread_id: str
    answer: str
    incomplete: bool
    incomplete_reason: IncompleteReason | None = None
    warnings: list[WarningOut] = Field(default_factory=list)
    image_ids: list[UUID] = Field(default_factory=list)

    @classmethod
    def de(cls, thread_id: str, resultado: TurnResult) -> ChatOut:
        return cls(
            thread_id=thread_id,
            answer=resultado.answer,
            incomplete=resultado.incomplete,
            incomplete_reason=resultado.incomplete_reason,
            warnings=[WarningOut(event=d.event, detail=d.detail) for d in resultado.degradations],
            image_ids=resultado.image_ids,
        )


class ErrorOut(BaseModel):
    error: str
    detail: str


class HealthOut(BaseModel):
    status: str
    provider_mode: str
    vector_store_mode: str
    images_indexed: int
