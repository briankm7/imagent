"""Construccion de proveedores. El unico modulo que sabe si estamos en modo real.

Los imports de `providers.real` son PEREZOSOS, dentro de cada rama. No es una
mania: las dependencias reales (`google-genai`, `qdrant-client`) estan en los
extras opcionales del pyproject y no se instalan en CI. Un import al principio
del fichero reventaria al arrancar en modo fake.

El efecto secundario es el que interesa: en CI los SDK ni siquiera existen en el
entorno, asi que el requisito de "testeable sin claves y sin base de datos" deja
de ser una intencion y pasa a ser una imposibilidad fisica de salir a la red.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from pydantic import SecretStr

from imagent.config import ProviderMode, RepositoryMode, Settings, VectorStoreMode
from imagent.demo import DEMO_SCRIPTS
from imagent.providers.base import (
    BlobStore,
    EmbeddingProvider,
    ImageRepository,
    TextProvider,
    VectorStore,
    VisionProvider,
)
from imagent.providers.fake.embeddings import FakeEmbeddingProvider
from imagent.providers.fake.repository import InMemoryImageRepository
from imagent.providers.fake.text_offline import HeuristicTextProvider
from imagent.providers.fake.vectorstore import InMemoryVectorStore
from imagent.providers.fake.vision import FakeVisionProvider
from imagent.providers.filesystem import FilesystemBlobStore

_PENDIENTE = (
    "los proveedores reales se implementan en un bloque posterior; usa IMAGENT_PROVIDER_MODE=fake"
)


@dataclass(frozen=True)
class Providers:
    """Todo lo que habla con el exterior, en un solo objeto.

    Se pasa entero a los servicios y a los nodos del grafo en vez de cinco
    parametros sueltos, y ninguno de ellos construye nada por su cuenta.
    """

    vision: VisionProvider
    text: TextProvider
    embeddings: EmbeddingProvider
    store: VectorStore
    repository: ImageRepository
    blobs: BlobStore


def build_providers(settings: Settings) -> Providers:
    embeddings = _build_embeddings(settings)
    return Providers(
        vision=_build_vision(settings),
        text=_build_text(settings),
        embeddings=embeddings,
        store=_build_store(settings, embeddings.dimensions),
        repository=_build_repository(settings),
        blobs=_build_blobs(settings),
    )


def _clave(settings: Settings) -> SecretStr:
    """La clave, con el fallo de configuracion ya descartado.

    El validador de Settings garantiza que en modo gemini existe, asi que este
    assert documenta la invariante en vez de comprobarla de verdad.
    """
    assert settings.google_api_key is not None
    return settings.google_api_key


def _build_vision(settings: Settings) -> VisionProvider:
    if settings.provider_mode is ProviderMode.GEMINI:
        from imagent.providers.real.gemini import GeminiVisionProvider

        return GeminiVisionProvider(api_key=_clave(settings), model=settings.vision_model)

    if settings.provider_mode is ProviderMode.FAKE:
        # Con los guiones de demostracion cargados, para que las tres imagenes
        # de ejemplo tengan contenido de verdad. strict=False para que subir
        # cualquier otra imagen no reviente: devuelve un relleno que se delata.
        # Los tests construyen el fake a mano y en modo estricto.
        return FakeVisionProvider(DEMO_SCRIPTS, strict=False)

    raise _modo_desconocido(settings.provider_mode)


def _build_text(settings: Settings) -> TextProvider:
    if settings.provider_mode is ProviderMode.GEMINI:
        from imagent.providers.real.gemini import GeminiTextProvider

        return GeminiTextProvider(api_key=_clave(settings), model=settings.text_model)

    if settings.provider_mode is ProviderMode.FAKE:
        # El de reglas, no el de cola: el de cola es un doble de test y aqui
        # degradaria en el primer turno.
        return HeuristicTextProvider()

    raise _modo_desconocido(settings.provider_mode)


def _build_embeddings(settings: Settings) -> EmbeddingProvider:
    if settings.provider_mode is ProviderMode.GEMINI:
        from imagent.providers.real.gemini import GeminiEmbeddingProvider

        return GeminiEmbeddingProvider(
            api_key=_clave(settings),
            model=settings.embedding_model,
            dimensions=settings.embedding_dimensions,
        )

    if settings.provider_mode is ProviderMode.FAKE:
        return FakeEmbeddingProvider(dimensions=settings.embedding_dimensions)

    raise _modo_desconocido(settings.provider_mode)


def _build_store(settings: Settings, dimensions: int) -> VectorStore:
    if settings.vector_store_mode is VectorStoreMode.QDRANT:
        from imagent.providers.real.qdrant import QdrantVectorStore

        return QdrantVectorStore(
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            # La dimension viene del proveedor de embeddings, no de la
            # configuracion: si se desincronizaran, todo upsert fallaria.
            dimensions=dimensions,
        )

    if settings.vector_store_mode is VectorStoreMode.MEMORY:
        return InMemoryVectorStore(dimensions=dimensions)

    raise _modo_desconocido(settings.vector_store_mode)


def _build_repository(settings: Settings) -> ImageRepository:
    if settings.repository_mode is RepositoryMode.SQLITE:
        from imagent.providers.real.sqlite import SqliteImageRepository

        return SqliteImageRepository(settings.sqlite_path)

    if settings.repository_mode is RepositoryMode.MEMORY:
        return InMemoryImageRepository()

    raise _modo_desconocido(settings.repository_mode)


def _build_blobs(settings: Settings) -> BlobStore:
    """El almacen de bytes NO tiene un modo de configuracion.

    El disco esta siempre disponible y no depende de ningun extra, asi que la
    app monta siempre el de disco. La eleccion del de memoria no es una opcion
    de despliegue: es un punto de inyeccion que usan los tests que no quieren
    tocar el sistema de ficheros. Un enum aqui seria un knob que nadie tocaria.
    """
    return FilesystemBlobStore(root=settings.storage_dir)


def _modo_desconocido(modo: StrEnum) -> NotImplementedError:
    """Un modo del enum sin rama aqui.

    Solo puede pasar si alguien añade un valor al enum y olvida este fichero.
    Que sea un error explicito y no un `return None` silencioso es la diferencia
    entre un fallo al arrancar y un None que revienta tres capas mas arriba.
    """
    return NotImplementedError(f"modo sin implementar: {type(modo).__name__}.{modo.name}")
