"""Tests del mecanismo de degradacion.

Lo que se defiende aqui es el contrato de fallo, no el logging de Python.
"""

from __future__ import annotations

import logging

import pytest

from imagent.domain.errors import (
    DegradationEvent,
    ImagentError,
    VectorStoreError,
    VisionError,
    degrade_on,
)
from imagent.observability.logging import CLAVE, log_extra


def _falla_la_vision() -> str:
    raise VisionError("el modelo devolvio 503")


def test_sin_fallo_no_registra_nada() -> None:
    sink: list[DegradationEvent] = []

    with degrade_on(VisionError, event="vision.on_demand.failed", sink=sink):
        pass

    assert sink == []


def test_captura_el_error_declarado_y_conserva_el_valor_por_defecto() -> None:
    sink: list[DegradationEvent] = []
    resultado = "valor por defecto"

    with degrade_on(VisionError, event="vision.on_demand.failed", sink=sink):
        resultado = _falla_la_vision()

    assert resultado == "valor por defecto"
    assert len(sink) == 1
    assert sink[0].event == "vision.on_demand.failed"
    assert sink[0].error_type == "VisionError"
    assert "503" in sink[0].detail


def test_un_error_no_declarado_propaga() -> None:
    """Un bug tuyo nunca se degrada."""
    sink: list[DegradationEvent] = []

    with (
        pytest.raises(ZeroDivisionError),
        degrade_on(VisionError, event="vision.on_demand.failed", sink=sink),
    ):
        _ = 1 / 0

    assert sink == []


def test_un_error_de_otro_dominio_propaga() -> None:
    """Declarar VisionError no te protege de que se caiga Qdrant."""
    sink: list[DegradationEvent] = []

    with (
        pytest.raises(VectorStoreError),
        degrade_on(VisionError, event="vision.on_demand.failed", sink=sink),
    ):
        raise VectorStoreError("connection refused")


def test_acepta_una_tupla_de_tipos() -> None:
    sink: list[DegradationEvent] = []

    with degrade_on((VisionError, TimeoutError), event="vision.on_demand.timeout", sink=sink):
        raise TimeoutError("25s")

    assert sink[0].error_type == "TimeoutError"


def test_la_cancelacion_nunca_se_degrada() -> None:
    """KeyboardInterrupt hereda de BaseException: un cancelado no es una degradacion."""
    sink: list[DegradationEvent] = []

    with (
        pytest.raises(KeyboardInterrupt),
        degrade_on(Exception, event="lo.que.sea", sink=sink),
    ):
        raise KeyboardInterrupt

    assert sink == []


def test_degradar_siempre_es_audible(caplog: pytest.LogCaptureFixture) -> None:
    """Un camino degradado que no deja rastro es peor que una caida."""
    sink: list[DegradationEvent] = []

    with (
        caplog.at_level(logging.WARNING, logger="imagent.degradation"),
        degrade_on(VisionError, event="vision.on_demand.failed", sink=sink),
    ):
        raise VisionError("timeout")

    registro = caplog.records[0]
    assert registro.levelno == logging.WARNING
    # Los campos estructurados viajan bajo una sola clave: `logging` reserva
    # nombres como `filename` o `module` y ponerlos sueltos en `extra` lanza un
    # KeyError EN LA LLAMADA AL LOGGER. Intentar observar rompeia la peticion.
    assert getattr(registro, CLAVE)["event"] == "vision.on_demand.failed"
    assert registro.exc_info is not None


def test_todo_error_propio_cuelga_de_una_raiz_comun() -> None:
    """`except ImagentError` tiene que atrapar cualquier fallo deliberado."""
    assert issubclass(VisionError, ImagentError)
    assert issubclass(VectorStoreError, ImagentError)


def test_se_pueden_loguear_campos_con_nombres_reservados(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """`filename` es un atributo reservado del LogRecord.

    Pasarlo suelto por `extra` no se ignora: lanza KeyError en la propia llamada
    al logger, o sea que intentar observar algo rompe lo que observabas. Bajo
    CLAVE no puede pasar.
    """
    logger = logging.getLogger("imagent.prueba")

    with caplog.at_level(logging.INFO, logger="imagent.prueba"):
        logger.info("subida", extra=log_extra(filename="coche.jpg", module="ingesta"))

    assert getattr(caplog.records[0], CLAVE)["filename"] == "coche.jpg"
