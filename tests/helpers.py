"""Utilidades compartidas por los tests.

Modulo normal y no `conftest.py` para que se pueda importar por su nombre desde
cualquier test y desde el editor. Las fixtures siguen en conftest.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from imagent.config import TimeoutSettings
from imagent.demo import DEMO_BLOBS, DEMO_FILENAMES, DEMO_SCRIPTS
from imagent.providers.fake.blobs import InMemoryBlobStore
from imagent.providers.fake.embeddings import FakeEmbeddingProvider
from imagent.providers.fake.repository import InMemoryImageRepository
from imagent.providers.fake.text import FakeTextProvider
from imagent.providers.fake.text_offline import HeuristicTextProvider
from imagent.providers.fake.vectorstore import InMemoryVectorStore
from imagent.providers.fake.vision import FakeVisionProvider
from imagent.providers.registry import Providers
from imagent.services.ingestion import IngestionService

DIMENSIONS = 256


def montar_providers(**sustituciones: Any) -> Providers:
    """Un juego de proveedores fake, con lo que se quiera sustituido."""
    base: dict[str, Any] = {
        "vision": FakeVisionProvider(DEMO_SCRIPTS),
        "text": FakeTextProvider(),
        "embeddings": FakeEmbeddingProvider(dimensions=DIMENSIONS),
        "store": InMemoryVectorStore(dimensions=DIMENSIONS),
        "repository": InMemoryImageRepository(),
        "blobs": InMemoryBlobStore(),
    }
    return Providers(**{**base, **sustituciones})


def montar_providers_de_app() -> Providers:
    """Lo mismo que monta el registry en modo fake, pero sin tocar el disco.

    El registry usa FilesystemBlobStore a proposito (el disco esta siempre y no
    depende de ningun extra). Para un test de la API eso significaria escribir
    ficheros de verdad, asi que aqui se sustituye por el de memoria y lo demas
    se deja igual: vision permisiva con los guiones de demostracion, y el
    planificador de reglas.
    """
    return montar_providers(
        vision=FakeVisionProvider(DEMO_SCRIPTS, strict=False),
        text=HeuristicTextProvider(),
    )


@dataclass
class Escena:
    """Las tres imagenes de demostracion ya ingeridas."""

    providers: Providers
    ids: dict[str, UUID]

    def id(self, nombre: str) -> UUID:
        return self.ids[nombre]

    @property
    def vision(self) -> Any:
        return self.providers.vision

    @property
    def store(self) -> Any:
        return self.providers.store

    @property
    def repository(self) -> Any:
        return self.providers.repository


async def ingerir_escena(providers: Providers) -> Escena:
    service = IngestionService(providers, timeouts=TimeoutSettings())

    ids: dict[str, UUID] = {}
    for nombre, blob in DEMO_BLOBS.items():
        resultado = await service.ingest(
            filename=DEMO_FILENAMES[nombre],
            media_type=blob.media_type,
            data=blob.data,
        )
        ids[nombre] = resultado.record.id

    return Escena(providers=providers, ids=ids)
