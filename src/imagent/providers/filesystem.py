"""BlobStore sobre el disco local.

No vive en `providers/real/` porque no depende de ningun extra opcional: el
disco esta siempre. La eleccion entre este y el de memoria no es un modo de
configuracion, es el punto de inyeccion: la app monta este, y un test que no
quiera tocar disco monta el de memoria.
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from uuid import UUID

from imagent.domain.errors import StorageError
from imagent.providers.base import ImageBlob


class FilesystemBlobStore:
    """Un fichero por imagen, con el UUID como nombre.

    El nombre no lleva extension a proposito: el media_type esta en el
    ImageRecord y duplicarlo en el nombre del fichero crearia dos fuentes que
    pueden contradecirse.
    """

    def __init__(self, root: Path) -> None:
        self._root = root

    def _path(self, image_id: UUID) -> Path:
        # image_id es un UUID, asi que no hay forma de que esto salga del
        # directorio: el tipo hace de validacion.
        return self._root / str(image_id)

    async def put(self, image_id: UUID, blob: ImageBlob) -> None:
        await asyncio.to_thread(self._put_sync, image_id, blob.data)

    def _put_sync(self, image_id: UUID, data: bytes) -> None:
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            # Escritura atomica: se escribe a un temporal en el MISMO directorio
            # y se renombra. Si el proceso muere a mitad, queda un temporal
            # suelto en vez de un fichero con el nombre bueno y el contenido
            # truncado, que pasaria por valido en cualquier lectura posterior.
            descriptor, temporal = tempfile.mkstemp(dir=self._root, suffix=".partial")
            try:
                with os.fdopen(descriptor, "wb") as fichero:
                    fichero.write(data)
                Path(temporal).replace(self._path(image_id))
            except BaseException:
                Path(temporal).unlink(missing_ok=True)
                raise
        except OSError as exc:
            raise StorageError(f"no se pudieron guardar los bytes de {image_id}: {exc}") from exc

    async def get_bytes(self, image_id: UUID) -> bytes:
        return await asyncio.to_thread(self._get_sync, image_id)

    def _get_sync(self, image_id: UUID) -> bytes:
        try:
            return self._path(image_id).read_bytes()
        except FileNotFoundError as exc:
            raise StorageError(f"no hay bytes guardados para la imagen {image_id}") from exc
        except OSError as exc:
            raise StorageError(f"no se pudieron leer los bytes de {image_id}: {exc}") from exc

    async def exists(self, image_id: UUID) -> bool:
        return await asyncio.to_thread(self._path(image_id).is_file)

    async def delete(self, image_id: UUID) -> None:
        await asyncio.to_thread(self._path(image_id).unlink, True)
