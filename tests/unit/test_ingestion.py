"""Tests del pipeline de ingesta.

El test que mas defiende el diseño es `test_el_orden_de_escritura_es_el_seguro`:
una sola asercion que fija las dos propiedades por las que los pasos estan
ordenados como estan.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any
from uuid import UUID

import pytest

from imagent.config import TimeoutSettings
from imagent.domain.errors import (
    InvalidImageError,
    StorageError,
    VectorStoreError,
    VisionError,
)
from imagent.domain.models import (
    Aspect,
    ImageAnalysis,
    ImageRecord,
    ImageStatus,
    IngestionStage,
)
from imagent.providers.base import ImageBlob, IndexedPoint
from imagent.providers.fake.blobs import InMemoryBlobStore
from imagent.providers.fake.embeddings import FakeEmbeddingProvider
from imagent.providers.fake.repository import InMemoryImageRepository
from imagent.providers.fake.text import FakeTextProvider
from imagent.providers.fake.vectorstore import InMemoryVectorStore
from imagent.providers.fake.vision import FakeVisionProvider, VisionScript
from imagent.providers.registry import Providers
from imagent.services.ingestion import IngestionResult, IngestionService

DIMENSIONS = 64
BYTES_COCHE = b"\xff\xd8\xff bytes de la foto del coche"
COCHE = ImageBlob(data=BYTES_COCHE, media_type="image/jpeg")

GUION_COCHE = VisionScript(
    description="Un coche rojo aparcado en la acera",
    ocr_text="SE VENDE",
    objects=("coche", "cartel"),
)


def montar(
    *,
    vision: Any = None,
    store: Any = None,
    repository: Any = None,
    blobs: Any = None,
) -> Providers:
    return Providers(
        vision=vision or FakeVisionProvider({COCHE.content_hash: GUION_COCHE}),
        text=FakeTextProvider(),
        embeddings=FakeEmbeddingProvider(dimensions=DIMENSIONS),
        store=store or InMemoryVectorStore(dimensions=DIMENSIONS),
        repository=repository or InMemoryImageRepository(),
        blobs=blobs or InMemoryBlobStore(),
    )


def servicio(providers: Providers, **timeouts: float) -> IngestionService:
    return IngestionService(providers, timeouts=TimeoutSettings(**timeouts))


async def subir(
    service: IngestionService, blob: ImageBlob = COCHE, nombre: str = "coche.jpg"
) -> IngestionResult:
    return await service.ingest(filename=nombre, media_type=blob.media_type, data=blob.data)


@pytest.fixture
def providers() -> Providers:
    return montar()


@pytest.fixture
def service(providers: Providers) -> IngestionService:
    return servicio(providers)


# ---------------------------------------------------------------------------
# Camino feliz
# ---------------------------------------------------------------------------
async def test_una_imagen_ingerida_queda_indexada(
    service: IngestionService, providers: Providers
) -> None:
    resultado = await subir(service)

    assert resultado.deduplicated is False
    assert resultado.record.status is ImageStatus.INDEXED
    assert resultado.record.content_hash == COCHE.content_hash
    assert resultado.record.size_bytes == len(BYTES_COCHE)
    assert await providers.repository.get(resultado.record.id) == resultado.record


async def test_los_bytes_quedan_guardados(service: IngestionService, providers: Providers) -> None:
    resultado = await subir(service)

    assert await providers.blobs.get_bytes(resultado.record.id) == BYTES_COCHE


async def test_se_indexa_un_punto_por_aspecto(
    service: IngestionService, providers: Providers
) -> None:
    """Decision D2-B, comprobada de punta a punta."""
    await subir(service)

    aspectos = sorted(p.aspect for p in providers.store.snapshot())
    assert aspectos == [Aspect.DESCRIPTION, Aspect.OBJECTS, Aspect.OCR]


async def test_los_aspectos_vacios_no_generan_punto() -> None:
    """Un OCR vacio produciria el vector de reserva y casaria con cualquier cosa."""
    sin_texto = ImageBlob(data=b"foto de un paisaje", media_type="image/png")
    providers = montar(
        vision=FakeVisionProvider(
            {sin_texto.content_hash: VisionScript(description="Un paisaje de montaña")}
        )
    )

    await subir(servicio(providers), sin_texto)

    assert [p.aspect for p in providers.store.snapshot()] == [Aspect.DESCRIPTION]


async def test_los_aspectos_se_embeben_en_una_sola_llamada(
    service: IngestionService, providers: Providers
) -> None:
    """Con un proveedor real, tres viajes de red menos por imagen."""
    await subir(service)

    assert len(providers.embeddings.documents_calls) == 1
    assert providers.embeddings.documents_calls[0] == [
        "Un coche rojo aparcado en la acera",
        "SE VENDE",
        "coche, cartel",
    ]


async def test_el_texto_indexado_es_crudo(service: IngestionService, providers: Providers) -> None:
    """Decision D11-A: nada de prefijos como 'Texto visible en la imagen:'."""
    await subir(service)

    ocr = next(p for p in providers.store.snapshot() if p.aspect is Aspect.OCR)
    assert ocr.text == "SE VENDE"


# ---------------------------------------------------------------------------
# El orden de escritura
# ---------------------------------------------------------------------------
async def test_el_orden_de_escritura_es_el_seguro() -> None:
    """Las dos propiedades del pipeline, en una sola asercion.

    - `save:analyzed` va ANTES de `upsert`: la llamada cara de vision se
      persiste en cuanto existe, asi que un fallo de indexado no la tira.
    - `upsert` va ANTES de `save:indexed`: si el proceso muere en medio quedan
      puntos huerfanos que el reintento sobrescribe, en vez de un registro que
      afirma estar indexado sin puntos detras.
    """
    diario: list[str] = []

    class RepoEspia(InMemoryImageRepository):
        async def save(self, record: ImageRecord) -> None:
            diario.append(f"save:{record.status.value}")
            await super().save(record)

    class StoreEspia(InMemoryVectorStore):
        async def upsert(self, points: Sequence[IndexedPoint]) -> None:
            diario.append("upsert")
            await super().upsert(points)

    providers = montar(repository=RepoEspia(), store=StoreEspia(dimensions=DIMENSIONS))

    await subir(servicio(providers))

    assert diario == ["save:pending", "save:analyzed", "upsert", "save:indexed"]


async def test_ensure_ready_se_llama_una_vez_por_instancia() -> None:
    """Con Qdrant real, lo contrario seria un viaje de red por subida."""
    llamadas = 0

    class StoreContador(InMemoryVectorStore):
        async def ensure_ready(self) -> None:
            nonlocal llamadas
            llamadas += 1
            await super().ensure_ready()

    providers = montar(store=StoreContador(dimensions=DIMENSIONS))
    service = servicio(providers)
    otra = ImageBlob(data=b"otra foto distinta", media_type="image/png")
    providers.vision.register(otra, VisionScript(description="Otra cosa"))

    await subir(service)
    await subir(service, otra, "otra.png")

    assert llamadas == 1


# ---------------------------------------------------------------------------
# Duplicados (D10-A)
# ---------------------------------------------------------------------------
async def test_subir_dos_veces_la_misma_imagen_devuelve_el_registro_existente(
    service: IngestionService, providers: Providers
) -> None:
    primera = await subir(service)

    segunda = await subir(service, nombre="coche-renombrado.jpg")

    assert segunda.deduplicated is True
    assert segunda.record.id == primera.record.id
    assert len(await providers.repository.list()) == 1


async def test_el_duplicado_no_vuelve_a_pagar_vision(
    service: IngestionService, providers: Providers
) -> None:
    """El coste que la decision D10-A existe para evitar."""
    await subir(service)
    await subir(service)

    assert len(providers.vision.describe_calls) == 1


async def test_el_duplicado_no_duplica_puntos(
    service: IngestionService, providers: Providers
) -> None:
    """Si los duplicara, la misma foto saldria dos veces en una respuesta."""
    await subir(service)
    await subir(service)

    assert len(providers.store.snapshot()) == 3


async def test_un_registro_fallido_no_cuenta_como_duplicado(
    service: IngestionService, providers: Providers
) -> None:
    """Si contase, una imagen que fallo quedaria condenada: cada reintento
    devolveria el registro roto en vez de intentarlo otra vez."""
    roto = ImageRecord(
        content_hash=COCHE.content_hash,
        filename="coche.jpg",
        media_type="image/jpeg",
        size_bytes=len(BYTES_COCHE),
    ).failed(IngestionStage.INDEXING, "qdrant caido")
    await providers.repository.save(roto)

    resultado = await subir(service)

    assert resultado.deduplicated is False
    assert resultado.record.status is ImageStatus.INDEXED


# ---------------------------------------------------------------------------
# Fallos: fatales, pero visibles
# ---------------------------------------------------------------------------
async def test_un_fallo_de_vision_deja_el_registro_marcado_y_propaga() -> None:
    providers = montar(
        vision=FakeVisionProvider(
            {COCHE.content_hash: VisionScript(description="x", describe_error=VisionError("503"))}
        )
    )

    with pytest.raises(VisionError, match="503"):
        await subir(servicio(providers))

    (registro,) = await providers.repository.list()
    assert registro.status is ImageStatus.FAILED
    assert registro.failed_stage is IngestionStage.ANALYSIS
    assert registro.analysis is None
    assert "503" in (registro.failure_reason or "")


async def test_un_fallo_de_indexado_conserva_el_analisis() -> None:
    """El pago de la decision P2-B, de punta a punta.

    La llamada de vision ya esta hecha; un fallo del almacen no puede tirarla.
    """

    class StoreRoto(InMemoryVectorStore):
        async def upsert(self, points: Sequence[IndexedPoint]) -> None:
            raise VectorStoreError("connection refused")

    providers = montar(store=StoreRoto(dimensions=DIMENSIONS))

    with pytest.raises(VectorStoreError):
        await subir(servicio(providers))

    (registro,) = await providers.repository.list()
    assert registro.status is ImageStatus.FAILED
    assert registro.failed_stage is IngestionStage.INDEXING
    assert registro.analysis is not None
    assert registro.analysis.description == "Un coche rojo aparcado en la acera"


async def test_un_fallo_de_disco_marca_la_fase_store() -> None:
    """Y no llega a llamar a vision: no se paga por algo que ya ha fallado."""

    class DiscoRoto(InMemoryBlobStore):
        async def put(self, image_id: UUID, blob: ImageBlob) -> None:
            raise StorageError("disco lleno")

    providers = montar(blobs=DiscoRoto())

    with pytest.raises(StorageError, match="disco lleno"):
        await subir(servicio(providers))

    (registro,) = await providers.repository.list()
    assert registro.failed_stage is IngestionStage.STORE
    assert providers.vision.describe_calls == []


async def test_un_analisis_sin_texto_indexable_es_un_fallo() -> None:
    """Indexarlo seria un exito silencioso: la imagen quedaria registrada y no
    la encontraria ninguna consulta."""
    providers = montar(
        vision=FakeVisionProvider({COCHE.content_hash: VisionScript(description="   ")})
    )

    with pytest.raises(VisionError, match="ningun texto indexable"):
        await subir(servicio(providers))

    (registro,) = await providers.repository.list()
    assert registro.failed_stage is IngestionStage.ANALYSIS


async def test_un_timeout_de_vision_tambien_marca_el_registro() -> None:
    """Sin capturar TimeoutError en la fase, la excepcion escaparia sin dejar
    rastro en el registro."""

    class VisionLenta:
        @property
        def model_name(self) -> str:
            return "lenta"

        async def describe(self, image: ImageBlob) -> ImageAnalysis:
            await asyncio.sleep(10)
            raise AssertionError("no deberia llegar")

        async def answer_about(self, image: ImageBlob, question: str) -> str:
            raise AssertionError("no deberia llegar")

    providers = montar(vision=VisionLenta())

    with pytest.raises(TimeoutError):
        await subir(servicio(providers, vision_ingestion_seconds=0.01))

    (registro,) = await providers.repository.list()
    assert registro.status is ImageStatus.FAILED
    assert registro.failed_stage is IngestionStage.ANALYSIS


# ---------------------------------------------------------------------------
# Entrada invalida
# ---------------------------------------------------------------------------
async def test_una_imagen_vacia_se_rechaza(service: IngestionService) -> None:
    with pytest.raises(InvalidImageError, match="vacia"):
        await service.ingest(filename="x.jpg", media_type="image/jpeg", data=b"")


async def test_un_tipo_que_no_es_imagen_se_rechaza(service: IngestionService) -> None:
    with pytest.raises(InvalidImageError, match="no soportado"):
        await service.ingest(filename="x.pdf", media_type="application/pdf", data=b"algo")


async def test_una_entrada_invalida_no_deja_registro(
    service: IngestionService, providers: Providers
) -> None:
    """Se rechaza antes de tocar nada: no hay estado parcial que limpiar."""
    with pytest.raises(InvalidImageError):
        await service.ingest(filename="x.pdf", media_type="application/pdf", data=b"algo")

    assert await providers.repository.list() == []
