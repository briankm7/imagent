"""BlobStore en memoria."""

from __future__ import annotations

from uuid import UUID

from imagent.domain.errors import StorageError
from imagent.providers.base import ImageBlob


class InMemoryBlobStore:
    """Implementa BlobStore con un diccionario.

    No es solo para tests: hace que un test del grafo entero no toque el disco,
    que es la diferencia entre una suite de milisegundos y una de segundos.
    """

    def __init__(self) -> None:
        self._blobs: dict[UUID, bytes] = {}

    async def put(self, image_id: UUID, blob: ImageBlob) -> None:
        self._blobs[image_id] = blob.data

    async def get_bytes(self, image_id: UUID) -> bytes:
        datos = self._blobs.get(image_id)
        if datos is None:
            raise StorageError(f"no hay bytes guardados para la imagen {image_id}")
        return datos

    async def exists(self, image_id: UUID) -> bool:
        return image_id in self._blobs

    async def delete(self, image_id: UUID) -> None:
        self._blobs.pop(image_id, None)
