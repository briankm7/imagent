"""Conformidad de los fakes con los Protocol.

Por que existe este fichero: `Protocol` es tipado ESTRUCTURAL y lo verifica un
type checker estatico. En este proyecto no corremos mypy, asi que sin esto un
fake podria desviarse del contrato (un parametro renombrado, un metodo que deja
de ser async) y nada se quejaria hasta que fallara el proveedor real, que es
cuando el fake ya no puede ayudarte.

Se comparan nombres, tipo y valor por defecto de cada parametro, y si el metodo
es una corrutina. NO se comparan anotaciones: con `from __future__ import
annotations` son cadenas, y `Vector` y `list[float]` son lo mismo escrito de dos
maneras. Comparar texto daria falsos positivos sin aportar nada.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from imagent.providers.base import (
    BlobStore,
    EmbeddingProvider,
    ImageRepository,
    TextProvider,
    VectorStore,
    VisionProvider,
)
from imagent.providers.fake.blobs import InMemoryBlobStore
from imagent.providers.fake.embeddings import FakeEmbeddingProvider
from imagent.providers.fake.repository import InMemoryImageRepository
from imagent.providers.fake.text import FakeTextProvider
from imagent.providers.fake.vectorstore import InMemoryVectorStore
from imagent.providers.fake.vision import FakeVisionProvider
from imagent.providers.filesystem import FilesystemBlobStore
from imagent.providers.real.gemini import (
    GeminiEmbeddingProvider,
    GeminiTextProvider,
    GeminiVisionProvider,
)
from imagent.providers.real.qdrant import QdrantVectorStore
from imagent.providers.real.sqlite import SqliteImageRepository

PAREJAS = [
    # Fakes
    pytest.param(VisionProvider, FakeVisionProvider, id="vision-fake"),
    pytest.param(TextProvider, FakeTextProvider, id="text-fake"),
    pytest.param(EmbeddingProvider, FakeEmbeddingProvider, id="embeddings-fake"),
    pytest.param(VectorStore, InMemoryVectorStore, id="vectorstore-memoria"),
    pytest.param(ImageRepository, InMemoryImageRepository, id="repository-memoria"),
    pytest.param(BlobStore, InMemoryBlobStore, id="blobs-memoria"),
    # Adaptadores reales. Se comprueban AQUI, en CI, aunque en CI no esten
    # instalados ni `google-genai` ni `qdrant-client`: los modulos importan el
    # SDK dentro de los metodos, no arriba, asi que la CLASE se puede inspeccionar
    # sin el. Sin esto, la forma de los proveedores reales -que son justo los que
    # no cubre ningun otro test- solo se verificaria en produccion.
    pytest.param(VisionProvider, GeminiVisionProvider, id="vision-gemini"),
    pytest.param(TextProvider, GeminiTextProvider, id="text-gemini"),
    pytest.param(EmbeddingProvider, GeminiEmbeddingProvider, id="embeddings-gemini"),
    pytest.param(VectorStore, QdrantVectorStore, id="vectorstore-qdrant"),
    pytest.param(ImageRepository, SqliteImageRepository, id="repository-sqlite"),
    pytest.param(BlobStore, FilesystemBlobStore, id="blobs-disco"),
]


def _miembros(protocolo: type) -> list[str]:
    return sorted(name for name in vars(protocolo) if not name.startswith("_"))


def _parametros(funcion: Any) -> list[tuple[str, inspect._ParameterKind, Any]]:
    parametros = list(inspect.signature(funcion).parameters.values())
    if parametros and parametros[0].name == "self":
        parametros = parametros[1:]
    return [(p.name, p.kind, p.default) for p in parametros]


@pytest.mark.parametrize(("protocolo", "implementacion"), PAREJAS)
def test_la_implementacion_tiene_todos_los_miembros(protocolo: type, implementacion: type) -> None:
    faltan = [m for m in _miembros(protocolo) if not hasattr(implementacion, m)]

    assert faltan == [], f"{implementacion.__name__} no implementa {faltan}"


@pytest.mark.parametrize(("protocolo", "implementacion"), PAREJAS)
def test_las_firmas_coinciden(protocolo: type, implementacion: type) -> None:
    for nombre in _miembros(protocolo):
        esperado = inspect.getattr_static(protocolo, nombre)
        real = inspect.getattr_static(implementacion, nombre)

        if isinstance(esperado, property):
            assert isinstance(real, property), (
                f"{implementacion.__name__}.{nombre} deberia ser una propiedad"
            )
            continue

        assert _parametros(real) == _parametros(esperado), (
            f"firma distinta en {implementacion.__name__}.{nombre}"
        )


@pytest.mark.parametrize(("protocolo", "implementacion"), PAREJAS)
def test_lo_que_es_async_sigue_siendo_async(protocolo: type, implementacion: type) -> None:
    """Decision P7-A: si un metodo deja de ser corrutina, asyncio.timeout() deja
    de poder cortarlo y el timeout por agente se pierde en silencio."""
    for nombre in _miembros(protocolo):
        esperado = inspect.getattr_static(protocolo, nombre)
        if isinstance(esperado, property):
            continue

        real = inspect.getattr_static(implementacion, nombre)
        assert inspect.iscoroutinefunction(real) == inspect.iscoroutinefunction(esperado), (
            f"{implementacion.__name__}.{nombre} no coincide en ser corrutina"
        )


def test_el_test_detectaria_una_desviacion() -> None:
    """Un test que nunca puede fallar no protege de nada; se comprueba que falla."""

    class VisionDesviado:
        @property
        def model_name(self) -> str:
            return "x"

        async def describe(self, imagen: object) -> None: ...  # parametro renombrado

        async def answer_about(self, image: object, question: str) -> None: ...

    assert _parametros(VisionDesviado.describe) != _parametros(VisionProvider.describe)
