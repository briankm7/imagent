"""Tests del dominio.

La maquina de estados de ImageRecord es la que decide que pasa cuando la
ingesta se rompe a mitad. Cada test de aqui fija una de esas garantias.
"""

from __future__ import annotations

import hashlib

import pytest
from pydantic import ValidationError

from imagent.domain.errors import InvalidStateTransition
from imagent.domain.models import (
    ImageAnalysis,
    ImageRecord,
    ImageStatus,
    IngestionStage,
)

HASH = hashlib.sha256(b"unos bytes de imagen").hexdigest()


def _record() -> ImageRecord:
    return ImageRecord(
        content_hash=HASH,
        filename="cartel.jpg",
        media_type="image/jpeg",
        size_bytes=2048,
    )


def _analysis() -> ImageAnalysis:
    return ImageAnalysis(
        description="Un cartel azul en una calle",
        ocr_text="SE VENDE",
        objects=["cartel", "calle"],
        model_name="fake-vision-1",
    )


def test_una_imagen_recien_subida_esta_pendiente() -> None:
    record = _record()

    assert record.status is ImageStatus.PENDING
    assert record.analysis is None
    assert record.failed_stage is None


def test_camino_feliz_pending_analyzed_indexed() -> None:
    record = _record().with_analysis(_analysis()).indexed()

    assert record.status is ImageStatus.INDEXED
    assert record.analysis is not None
    assert record.analysis.ocr_text == "SE VENDE"


def test_las_transiciones_no_mutan_el_registro_anterior() -> None:
    pendiente = _record()

    analizado = pendiente.with_analysis(_analysis())

    assert pendiente.status is ImageStatus.PENDING
    assert analizado.status is ImageStatus.ANALYZED
    assert analizado.id == pendiente.id


def test_el_registro_es_inmutable() -> None:
    with pytest.raises(ValidationError):
        _record().status = ImageStatus.INDEXED  # type: ignore[misc]


def test_fallar_el_indexado_conserva_el_analisis() -> None:
    """El motivo entero de que ImageStatus tenga cuatro valores.

    La llamada de vision ya esta pagada; un fallo posterior no puede tirarla.
    """
    analizado = _record().with_analysis(_analysis())

    roto = analizado.failed(IngestionStage.INDEXING, "qdrant: connection refused")

    assert roto.status is ImageStatus.FAILED
    assert roto.failed_stage is IngestionStage.INDEXING
    assert roto.analysis is not None
    assert roto.analysis.description == "Un cartel azul en una calle"


def test_fallar_antes_de_analizar_no_deja_analisis() -> None:
    roto = _record().failed(IngestionStage.ANALYSIS, "timeout de 60s")

    assert roto.analysis is None
    assert roto.failure_reason == "timeout de 60s"


def test_no_se_puede_indexar_sin_analizar() -> None:
    with pytest.raises(InvalidStateTransition, match="pending"):
        _record().indexed()


def test_no_se_puede_analizar_dos_veces() -> None:
    analizado = _record().with_analysis(_analysis())

    with pytest.raises(InvalidStateTransition, match="analyzed"):
        analizado.with_analysis(_analysis())


def test_un_registro_indexado_no_puede_pasar_a_fallido() -> None:
    indexado = _record().with_analysis(_analysis()).indexed()

    with pytest.raises(InvalidStateTransition):
        indexado.failed(IngestionStage.INDEXING, "tarde")


def test_estado_indexado_sin_analisis_es_irrepresentable() -> None:
    """Un registro que dice estar indexado sin analisis es un hueco silencioso."""
    with pytest.raises(ValidationError, match="exige un analisis"):
        ImageRecord(
            content_hash=HASH,
            filename="x.jpg",
            media_type="image/jpeg",
            size_bytes=1,
            status=ImageStatus.INDEXED,
        )


def test_estado_fallido_sin_fase_es_irrepresentable() -> None:
    with pytest.raises(ValidationError, match="en que fase murio"):
        ImageRecord(
            content_hash=HASH,
            filename="x.jpg",
            media_type="image/jpeg",
            size_bytes=1,
            status=ImageStatus.FAILED,
        )


def test_una_fase_de_fallo_sin_estado_fallido_es_irrepresentable() -> None:
    with pytest.raises(ValidationError, match="solo un registro FAILED"):
        ImageRecord(
            content_hash=HASH,
            filename="x.jpg",
            media_type="image/jpeg",
            size_bytes=1,
            failed_stage=IngestionStage.STORE,
        )


def test_el_hash_tiene_que_ser_un_sha256() -> None:
    with pytest.raises(ValidationError):
        ImageRecord(
            content_hash="no-soy-un-hash",
            filename="x.jpg",
            media_type="image/jpeg",
            size_bytes=1,
        )


def test_dos_subidas_del_mismo_contenido_son_registros_distintos() -> None:
    """P3-C: el id es opaco; deduplicar lo decide el servicio, no el esquema."""
    primera, segunda = _record(), _record()

    assert primera.id != segunda.id
    assert primera.content_hash == segunda.content_hash
