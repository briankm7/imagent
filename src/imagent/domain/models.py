"""Modelos de dominio.

Este modulo no importa FastAPI, ni LangGraph, ni el cliente de Qdrant. Es el
vocabulario del proyecto y tiene que poder usarse desde cualquier capa.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Self
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from imagent.domain.errors import InvalidStateTransition


class Aspect(StrEnum):
    """Faceta indexable de una imagen (decision D2-B).

    Cada imagen produce un punto de Qdrant por aspecto, no uno solo con todo
    concatenado: una descripcion de 80 palabras diluye un OCR de 4, y la
    pregunta "que pone en el cartel" recuperaria mal.
    """

    DESCRIPTION = "description"
    OCR = "ocr"
    OBJECTS = "objects"


class IngestionStage(StrEnum):
    """Fases de la ingesta. Sirven para decir DONDE murio un registro fallido."""

    STORE = "store"
    ANALYSIS = "analysis"
    EMBEDDING = "embedding"
    INDEXING = "indexing"


class ImageStatus(StrEnum):
    """Estado de un registro de imagen (decision P2-B).

    Cuatro estados y no dos porque ANALYZED es el punto donde ya has pagado la
    llamada cara de vision. Si el indexado falla despues, el analisis sigue
    guardado y reintentar es gratis. Sin ese estado intermedio, cualquier fallo
    posterior a la vision te obliga a volver a pagarla.
    """

    PENDING = "pending"
    ANALYZED = "analyzed"
    INDEXED = "indexed"
    FAILED = "failed"


class ImageAnalysis(BaseModel):
    """Lo que el agente de vision extrae en la ingesta. Esto es lo que se indexa."""

    model_config = ConfigDict(frozen=True)

    description: str
    ocr_text: str = ""
    objects: list[str] = Field(default_factory=list)

    model_name: str
    """Que modelo lo produjo. Procedencia: sin esto no puedes reindexar cuando
    cambies de modelo, ni distinguir un analisis viejo de uno nuevo."""


class ImageRecord(BaseModel):
    """Una imagen ingerida y su estado en el pipeline.

    Inmutable a proposito: las transiciones devuelven un registro nuevo en vez
    de mutar. Asi una funcion no puede dejarte el registro a medias si revienta
    por la mitad, y el estado anterior sigue siendo valido.
    (Si prefieres un modelo mutable con un unico `transition_to`, se cambia en
    unas diez lineas: dimelo.)
    """

    model_config = ConfigDict(frozen=True)

    # Identidad opaca (decision P3-C): el id no dice nada del contenido...
    id: UUID = Field(default_factory=uuid4)
    # ...y el hash va aparte, indexado. Deduplicar pasa a ser una POLITICA que
    # decide el servicio de ingesta, no una consecuencia del esquema.
    content_hash: str = Field(pattern=r"^[0-9a-f]{64}$")

    filename: str
    media_type: str
    size_bytes: int = Field(gt=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    status: ImageStatus = ImageStatus.PENDING
    analysis: ImageAnalysis | None = None
    failed_stage: IngestionStage | None = None
    failure_reason: str | None = None

    @model_validator(mode="after")
    def _check_invariants(self) -> Self:
        """Hace irrepresentables los estados imposibles.

        Un registro que dice estar indexado pero no tiene analisis es peor que
        un error: es un hueco silencioso en el indice.
        """
        if self.status is ImageStatus.FAILED:
            if self.failed_stage is None:
                raise ValueError("un registro FAILED tiene que decir en que fase murio")
        elif self.failed_stage is not None:
            raise ValueError("solo un registro FAILED puede llevar failed_stage")

        if self.status in {ImageStatus.ANALYZED, ImageStatus.INDEXED} and self.analysis is None:
            raise ValueError(f"status={self.status} exige un analisis")

        return self

    def _evolve(self, **changes: object) -> ImageRecord:
        """Copia validada.

        Ojo: `model_copy(update=...)` NO vuelve a pasar por los validadores, asi
        que se saltaria las invariantes de arriba. Reconstruir es mas lento y
        aqui da exactamente igual; lo que no da igual es que un estado imposible
        pueda existir.
        """
        return ImageRecord(**{**self.model_dump(), **changes})

    def with_analysis(self, analysis: ImageAnalysis) -> ImageRecord:
        """PENDING -> ANALYZED. A partir de aqui, la vision ya esta pagada."""
        if self.status is not ImageStatus.PENDING:
            raise InvalidStateTransition(f"with_analysis desde {self.status}, se esperaba pending")
        return self._evolve(status=ImageStatus.ANALYZED, analysis=analysis)

    def indexed(self) -> ImageRecord:
        """ANALYZED -> INDEXED. Ya es consultable."""
        if self.status is not ImageStatus.ANALYZED:
            raise InvalidStateTransition(f"indexed desde {self.status}, se esperaba analyzed")
        return self._evolve(status=ImageStatus.INDEXED)

    def failed(self, stage: IngestionStage, reason: str) -> ImageRecord:
        """-> FAILED, conservando el analisis si ya existia.

        Conservarlo es el motivo de que ImageStatus tenga cuatro valores: un
        fallo de indexado no debe tirar a la basura una llamada de vision que ya
        has pagado.
        """
        if self.status is ImageStatus.INDEXED:
            raise InvalidStateTransition("un registro ya indexado no puede pasar a failed")
        return self._evolve(
            status=ImageStatus.FAILED,
            failed_stage=stage,
            failure_reason=reason,
        )


class RetrievedImage(BaseModel):
    """Una imagen recuperada, ya colapsada.

    D2-B produce varios puntos por imagen; el agente de recuperacion los agrupa
    y devuelve esto. `matched_aspects` se conserva porque es la evidencia que el
    coordinador necesita para juzgar si la recuperacion basta: no es lo mismo
    casar por descripcion que casar por OCR.
    """

    model_config = ConfigDict(frozen=True)

    image_id: UUID
    filename: str
    score: float
    matched_aspects: dict[Aspect, float] = Field(default_factory=dict)

    description: str = ""
    ocr_text: str = ""
    objects: list[str] = Field(default_factory=list)


class VisionFinding(BaseModel):
    """Respuesta del agente de vision al volver a mirar UNA imagen."""

    model_config = ConfigDict(frozen=True)

    image_id: UUID
    filename: str = ""
    """Se copia del registro al mirar.

    Redundante con el id y util igualmente: quien redacta la respuesta tiene que
    poder citar "en libreta.png" y no un UUID, y quien lee los logs tambien.
    """

    question: str
    answer: str
    model_name: str
