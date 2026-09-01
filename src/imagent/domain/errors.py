"""Errores del dominio y el mecanismo de degradacion.

Decision de diseño (P1-C). Los errores se clasifican por **dominio**
(que subsistema fallo), no por contrato (fatal o degradable). El motivo es que
el contrato no es una propiedad del error: un `VisionError` es **fatal** durante
la ingesta (si no analizas la imagen, indexas un hueco invisible) y es
**degradable** en la re-mirada bajo demanda (respondes peor, pero respondes).
La misma clase, dos contratos. Meterlo en la jerarquia obligaria a envolver y
relanzar segun quien llame.

El contrato lo pone entonces quien llama, pero no de cualquier manera: solo a
traves de `degrade_on`, que es la unica forma legitima de que un error deje de
propagarse. Eso da tres cosas a la vez:

1. cada punto de degradacion es una linea grepeable (`grep -rn degrade_on`);
2. degradar **siempre** emite un log WARNING con nombre de evento estable, asi
   que el camino degradado es audible por construccion y no por disciplina;
3. lo que no esta en la lista de tipos propaga. Un bug tuyo nunca se degrada.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, MutableSequence
from contextlib import contextmanager

from pydantic import BaseModel, ConfigDict

from imagent.observability.logging import log_extra

logger = logging.getLogger("imagent.degradation")


# ---------------------------------------------------------------------------
# Jerarquia
# ---------------------------------------------------------------------------
class ImagentError(Exception):
    """Raiz de todo lo que este proyecto lanza a proposito."""


class ProviderError(ImagentError):
    """Fallo al hablar con un proveedor externo (modelo o almacen)."""


class VisionError(ProviderError):
    """El modelo multimodal no ha podido analizar una imagen."""


class TextError(ProviderError):
    """El modelo de texto no ha podido razonar o generar."""


class EmbeddingError(ProviderError):
    """No se han podido calcular embeddings."""


class RepositoryError(ProviderError):
    """El repositorio de registros no responde o rechaza la operacion."""


class VectorStoreError(ProviderError):
    """El almacen de vectores no responde o rechaza la operacion."""


class StorageError(ImagentError):
    """Fallo escribiendo o leyendo los bytes de una imagen."""


class InvalidImageError(ImagentError):
    """Lo que se ha subido no sirve como imagen.

    Es culpa de quien llama, no del entorno, asi que nunca degrada: la respuesta
    correcta es rechazar la subida y decir por que.
    """


class InvalidStateTransition(ImagentError):
    """Se ha intentado una transicion de estado imposible.

    Nunca se degrada: esto es un bug, no una condicion del entorno.
    """


# ---------------------------------------------------------------------------
# Degradacion
# ---------------------------------------------------------------------------
class DegradationEvent(BaseModel):
    """Rastro de que algo fallo y el sistema siguio adelante.

    Viaja en el estado del grafo y sale tambien en la respuesta HTTP, no solo
    en los logs: si solo estuviera en el log, en una demo no se veria.
    """

    model_config = ConfigDict(frozen=True)

    event: str
    """Nombre estable y jerarquico, p.ej. 'vision.on_demand.failed'."""

    detail: str
    error_type: str


@contextmanager
def degrade_on(
    errors: type[Exception] | tuple[type[Exception], ...],
    *,
    event: str,
    sink: MutableSequence[DegradationEvent],
) -> Iterator[None]:
    """Marca un punto donde un fallo concreto NO debe tumbar al llamador.

    El valor por defecto lo pone quien llama, antes del bloque, para que sea
    visible en la lectura:

        findings: list[VisionFinding] = []
        with degrade_on(VisionError, event="vision.on_demand.failed", sink=degradations):
            findings = await vision.answer_about(image, question)

    Solo se capturan los tipos de `errors`. Cualquier otra excepcion propaga,
    incluidas las que heredan de BaseException (KeyboardInterrupt, CancelledError):
    un cancelado no es una degradacion, es un cancelado.
    """
    try:
        yield
    except errors as exc:
        degradation = DegradationEvent(
            event=event,
            detail=str(exc) or exc.__class__.__name__,
            error_type=exc.__class__.__name__,
        )
        sink.append(degradation)
        logger.warning(
            "camino degradado: %s (%s)",
            event,
            degradation.error_type,
            extra=log_extra(event=event, error_type=degradation.error_type),
            exc_info=exc,
        )
