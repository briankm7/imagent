"""Tests del repositorio de imagenes.

Los mismos tests corren contra las DOS implementaciones: la de memoria y la de
SQLite. Es lo unico que da derecho a decir que son intercambiables; un fake que
se comporta distinto que la implementacion real no protege de nada.

El contrato que mas importa aqui es el ORDEN: "la tercera imagen" de una
conversacion depende de el.
"""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from imagent.domain.models import ImageAnalysis, ImageRecord, ImageStatus, IngestionStage
from imagent.providers.base import ImageRepository
from imagent.providers.fake.repository import InMemoryImageRepository
from imagent.providers.real.sqlite import SqliteImageRepository

AYER = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)
HOY = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)


def _hash(semilla: str) -> str:
    return hashlib.sha256(semilla.encode()).hexdigest()


def registro(
    nombre: str = "foto.jpg",
    *,
    creado: datetime | None = None,
    contenido: str | None = None,
) -> ImageRecord:
    return ImageRecord(
        content_hash=_hash(contenido if contenido is not None else nombre),
        filename=nombre,
        media_type="image/jpeg",
        size_bytes=1024,
        created_at=creado if creado is not None else HOY,
    )


def _analisis() -> ImageAnalysis:
    return ImageAnalysis(description="algo", model_name="fake-vision-1")


@pytest.fixture(params=["memoria", "sqlite"])
async def repo(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[ImageRepository]:
    if request.param == "memoria":
        yield InMemoryImageRepository()
        return

    # SQLite SI se prueba en CI: `aiosqlite` entra como dependencia del
    # checkpointer, que es base. No hace falta ningun servicio.
    repositorio = SqliteImageRepository(tmp_path / "imagent.sqlite")
    yield repositorio
    await repositorio.close()


async def test_guardar_y_recuperar(repo: ImageRepository) -> None:
    record = registro()
    await repo.save(record)

    assert await repo.get(record.id) == record


async def test_un_id_desconocido_devuelve_none(repo: ImageRepository) -> None:
    assert await repo.get(registro().id) is None


async def test_guardar_sobrescribe_por_id(repo: ImageRepository) -> None:
    """Las transiciones devuelven un registro nuevo con el mismo id."""
    pendiente = registro()
    await repo.save(pendiente)

    await repo.save(pendiente.with_analysis(_analisis()))

    recuperado = await repo.get(pendiente.id)
    assert recuperado is not None
    assert recuperado.status is ImageStatus.ANALYZED
    assert len(await repo.list()) == 1


async def test_buscar_por_hash(repo: ImageRepository) -> None:
    record = registro(contenido="bytes concretos")
    await repo.save(record)

    assert await repo.find_by_hash(record.content_hash) == record
    assert await repo.find_by_hash(_hash("otra cosa")) is None


async def test_buscar_por_hash_devuelve_el_mas_antiguo(repo: ImageRepository) -> None:
    """Es el que ya tiene la vision pagada, que es el util para deduplicar."""
    vieja = registro("a.jpg", creado=AYER, contenido="mismos bytes")
    nueva = registro("b.jpg", creado=HOY, contenido="mismos bytes")
    await repo.save(nueva)
    await repo.save(vieja)

    assert await repo.find_by_hash(vieja.content_hash) == vieja


async def test_list_ordena_por_fecha_de_subida(repo: ImageRepository) -> None:
    """Sin este orden, "la tercera" no significa nada."""
    tercera = registro("c.jpg", creado=HOY)
    primera = registro("a.jpg", creado=AYER)
    segunda = registro("b.jpg", creado=AYER + timedelta(hours=1))
    for record in (tercera, primera, segunda):
        await repo.save(record)

    assert [r.filename for r in await repo.list()] == ["a.jpg", "b.jpg", "c.jpg"]


async def test_el_desempate_va_por_orden_de_subida(repo: ImageRepository) -> None:
    """Dos subidas del mismo instante conservan el orden en que llegaron.

    Desempatar por id seria estable dentro de una ejecucion y aleatorio entre
    ejecuciones, porque un UUID no tiene nada que ver con cuando se subio la
    imagen. Y "la tercera" dejaria de significar nada.
    """
    segunda = registro("b.jpg", creado=HOY, contenido="b")
    primera = registro("a.jpg", creado=HOY, contenido="a")
    await repo.save(segunda)
    await repo.save(primera)

    assert [r.filename for r in await repo.list()] == ["b.jpg", "a.jpg"]


async def test_actualizar_un_registro_no_lo_mueve_de_sitio(
    repo: ImageRepository,
) -> None:
    """El numero de orden es el de subida, no el de la ultima escritura."""
    primera = registro("a.jpg", creado=HOY, contenido="a")
    segunda = registro("b.jpg", creado=HOY, contenido="b")
    await repo.save(primera)
    await repo.save(segunda)

    await repo.save(primera.with_analysis(_analisis()))

    assert [r.filename for r in await repo.list()] == ["a.jpg", "b.jpg"]


async def test_list_filtra_por_estado(repo: ImageRepository) -> None:
    pendiente = registro("pendiente.jpg", contenido="1")
    indexada = registro("indexada.jpg", contenido="2").with_analysis(_analisis()).indexed()
    fallida = registro("fallida.jpg", contenido="3").failed(IngestionStage.ANALYSIS, "503")
    for record in (pendiente, indexada, fallida):
        await repo.save(record)

    consultables = await repo.list(statuses=[ImageStatus.INDEXED])

    assert [r.filename for r in consultables] == ["indexada.jpg"]


async def test_list_filtra_por_rango_de_fechas(repo: ImageRepository) -> None:
    """El filtro de "las de ayer": vive aqui, no en Qdrant (decision P8-B)."""
    de_ayer = registro("ayer.jpg", creado=AYER, contenido="1")
    de_hoy = registro("hoy.jpg", creado=HOY, contenido="2")
    await repo.save(de_ayer)
    await repo.save(de_hoy)

    resultado = await repo.list(created_before=AYER + timedelta(hours=12))

    assert [r.filename for r in resultado] == ["ayer.jpg"]


async def test_list_combina_filtros(repo: ImageRepository) -> None:
    vieja_indexada = (
        registro("vieja.jpg", creado=AYER, contenido="1").with_analysis(_analisis()).indexed()
    )
    nueva_indexada = (
        registro("nueva.jpg", creado=HOY, contenido="2").with_analysis(_analisis()).indexed()
    )
    nueva_pendiente = registro("pendiente.jpg", creado=HOY, contenido="3")
    for record in (vieja_indexada, nueva_indexada, nueva_pendiente):
        await repo.save(record)

    resultado = await repo.list(
        statuses=[ImageStatus.INDEXED], created_after=AYER + timedelta(hours=1)
    )

    assert [r.filename for r in resultado] == ["nueva.jpg"]


async def test_un_repositorio_vacio_devuelve_lista_vacia(repo: ImageRepository) -> None:
    assert await repo.list() == []
