"""Tests del registry.

El ultimo es el que convierte "el CI no usa la red" de intencion a hecho.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from uuid import uuid4

import pytest

from imagent.config import RepositoryMode, Settings
from imagent.demo import LIBRETA
from imagent.domain.models import Aspect
from imagent.providers.base import ImageBlob, IndexedPoint
from imagent.providers.fake.embeddings import FakeEmbeddingProvider
from imagent.providers.fake.repository import InMemoryImageRepository
from imagent.providers.fake.text_offline import HeuristicTextProvider
from imagent.providers.fake.vectorstore import InMemoryVectorStore
from imagent.providers.fake.vision import FakeVisionProvider
from imagent.providers.filesystem import FilesystemBlobStore
from imagent.providers.real.sqlite import SqliteImageRepository
from imagent.providers.registry import build_providers


def test_por_defecto_construye_todo_fake(settings: Settings) -> None:
    providers = build_providers(settings)

    assert isinstance(providers.vision, FakeVisionProvider)
    assert isinstance(providers.text, HeuristicTextProvider)
    assert isinstance(providers.embeddings, FakeEmbeddingProvider)
    assert isinstance(providers.store, InMemoryVectorStore)
    assert isinstance(providers.repository, InMemoryImageRepository)
    # El de bytes es el de disco incluso en modo fake: no depende de extras ni
    # de red, asi que no tiene por que simularse.
    assert isinstance(providers.blobs, FilesystemBlobStore)


async def test_el_almacen_hereda_la_dimension_de_los_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Si almacen y embeddings se desincronizan, todo upsert falla.

    Que lo cablee el registry en un solo sitio es lo que lo evita. Se comprueba
    por comportamiento y no leyendo un atributo privado: lo que importa es que
    un vector del proveedor entre y uno de otra dimension no.
    """
    monkeypatch.setenv("IMAGENT_EMBEDDING_DIMENSIONS", "128")
    providers = build_providers(Settings())
    await providers.store.ensure_ready()

    vector = await providers.embeddings.embed_query("un coche rojo")
    punto = IndexedPoint.for_aspect(
        image_id=uuid4(), aspect=Aspect.DESCRIPTION, text="un coche rojo", vector=vector
    )
    await providers.store.upsert([punto])

    assert len(await providers.store.search(vector, limit=1)) == 1


async def test_la_vision_de_la_app_no_es_estricta(settings: Settings) -> None:
    """En la app real, subir una imagen sin guion no puede reventar; en los tests si."""
    providers = build_providers(settings)

    analisis = await providers.vision.describe(
        ImageBlob(data=b"cualquier cosa", media_type="image/png")
    )

    assert "sin guion registrado" in analisis.description


async def test_la_vision_de_la_app_trae_el_escenario_de_demostracion(
    settings: Settings,
) -> None:
    """Sin los guiones cargados, la demostracion no demuestra nada: la app
    aceptaria imagenes pero no sabria nada de ninguna."""
    providers = build_providers(settings)

    analisis = await providers.vision.describe(LIBRETA)

    assert "libreta" in analisis.description.lower()


def test_el_repositorio_sqlite_se_monta_sin_dependencias_extra(settings: Settings) -> None:
    """SQLite no necesita ningun extra: aiosqlite entra con el checkpointer.

    Por eso esta rama SI se puede comprobar en CI, a diferencia de las de Gemini
    y Qdrant, que necesitan sus SDK para construir el cliente.
    """
    con_sqlite = settings.model_copy(update={"repository_mode": RepositoryMode.SQLITE})

    providers = build_providers(con_sqlite)

    assert isinstance(providers.repository, SqliteImageRepository)


def test_el_modo_gemini_monta_los_adaptadores_de_gemini(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Se salta en CI, donde el SDK no esta instalado a proposito."""
    pytest.importorskip("google.genai", reason="el extra 'gemini' no esta instalado")
    monkeypatch.setenv("IMAGENT_PROVIDER_MODE", "gemini")
    monkeypatch.setenv("IMAGENT_GOOGLE_API_KEY", "clave-de-prueba")

    providers = build_providers(Settings())

    assert type(providers.vision).__name__ == "GeminiVisionProvider"
    assert type(providers.text).__name__ == "GeminiTextProvider"
    assert type(providers.embeddings).__name__ == "GeminiEmbeddingProvider"


def test_el_modo_qdrant_monta_el_almacen_de_qdrant(monkeypatch: pytest.MonkeyPatch) -> None:
    """Construir el cliente NO abre ninguna conexion: Qdrant conecta al usarse."""
    pytest.importorskip("qdrant_client", reason="el extra 'qdrant' no esta instalado")
    monkeypatch.setenv("IMAGENT_VECTOR_STORE_MODE", "qdrant")

    providers = build_providers(Settings())

    assert type(providers.store).__name__ == "QdrantVectorStore"


@pytest.mark.integration
def test_en_modo_fake_no_se_importa_ningun_sdk_de_red() -> None:
    """El requisito de "sin claves y sin base de datos", verificado.

    Se hace en un subproceso limpio porque otro test podria haber importado ya
    algo y contaminar sys.modules de este.
    """
    codigo = textwrap.dedent("""
        import sys
        from imagent.config import Settings
        from imagent.providers.registry import build_providers

        build_providers(Settings())

        prohibidos = sorted(
            m for m in sys.modules if m.split(".")[0] in {"google", "qdrant_client"}
        )
        assert not prohibidos, f"modulos de red importados: {prohibidos}"
    """)
    entorno = {k: v for k, v in os.environ.items() if not k.upper().startswith("IMAGENT_")}

    resultado = subprocess.run(
        [sys.executable, "-c", codigo],
        capture_output=True,
        text=True,
        env=entorno,
        check=False,
    )

    assert resultado.returncode == 0, resultado.stderr
