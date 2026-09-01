"""Tests del almacen de bytes.

Los mismos tests corren contra las DOS implementaciones. Ese es el rendimiento
real de tener una interfaz: el contrato se escribe una vez y se comprueba que
ambas lo cumplen igual. Un fake que se comporta distinto que la implementacion
real no sirve de nada.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

import pytest

from imagent.domain.errors import StorageError
from imagent.providers.base import BlobStore, ImageBlob
from imagent.providers.fake.blobs import InMemoryBlobStore
from imagent.providers.filesystem import FilesystemBlobStore

FOTO = ImageBlob(data=b"\x89PNG\r\n\x1a\n bytes de una foto", media_type="image/png")


@pytest.fixture(params=["memoria", "disco"])
def store(request: pytest.FixtureRequest, tmp_path: Path) -> BlobStore:
    if request.param == "memoria":
        return InMemoryBlobStore()
    return FilesystemBlobStore(root=tmp_path / "imagenes")


async def test_guardar_y_recuperar(store: BlobStore) -> None:
    image_id = uuid4()

    await store.put(image_id, FOTO)

    assert await store.get_bytes(image_id) == FOTO.data


async def test_pedir_algo_que_no_esta_lanza_storage_error(store: BlobStore) -> None:
    """No devuelve None: que falten los bytes de una imagen registrada es una
    inconsistencia, no un caso normal."""
    with pytest.raises(StorageError, match="no hay bytes"):
        await store.get_bytes(uuid4())


async def test_sobrescribir_es_idempotente(store: BlobStore) -> None:
    image_id = uuid4()
    await store.put(image_id, FOTO)

    await store.put(image_id, ImageBlob(data=b"otros bytes", media_type="image/png"))

    assert await store.get_bytes(image_id) == b"otros bytes"


async def test_exists(store: BlobStore) -> None:
    image_id = uuid4()
    assert await store.exists(image_id) is False

    await store.put(image_id, FOTO)

    assert await store.exists(image_id) is True


async def test_borrar(store: BlobStore) -> None:
    image_id = uuid4()
    await store.put(image_id, FOTO)

    await store.delete(image_id)

    assert await store.exists(image_id) is False


async def test_borrar_algo_que_no_existe_no_falla(store: BlobStore) -> None:
    """La limpieza tiene que poder reintentarse."""
    await store.delete(uuid4())


async def test_las_imagenes_no_se_pisan(store: BlobStore) -> None:
    una, otra = uuid4(), uuid4()

    await store.put(una, FOTO)
    await store.put(otra, ImageBlob(data=b"otra foto", media_type="image/jpeg"))

    assert await store.get_bytes(una) == FOTO.data
    assert await store.get_bytes(otra) == b"otra foto"


async def test_el_disco_crea_el_directorio_si_no_existe(tmp_path: Path) -> None:
    store = FilesystemBlobStore(root=tmp_path / "no" / "existe" / "todavia")

    await store.put(uuid4(), FOTO)


async def test_el_disco_no_deja_ficheros_a_medias(tmp_path: Path) -> None:
    """Escritura atomica: al terminar solo esta el fichero bueno.

    Un fichero truncado con el nombre correcto pasaria por valido en cualquier
    lectura posterior, que es la peor forma de fallar.
    """
    raiz = tmp_path / "imagenes"
    store = FilesystemBlobStore(root=raiz)
    image_id = uuid4()

    await store.put(image_id, FOTO)

    ficheros = sorted(p.name for p in raiz.iterdir())
    assert ficheros == [str(image_id)]


async def test_un_error_del_disco_se_convierte_en_storage_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Los errores del sistema operativo no se escapan a las capas de arriba."""
    store = FilesystemBlobStore(root=tmp_path)

    def revienta(*_: object, **__: object) -> None:
        raise OSError("disco lleno")

    monkeypatch.setattr("tempfile.mkstemp", revienta)

    with pytest.raises(StorageError, match="disco lleno"):
        await store.put(uuid4(), FOTO)


def test_el_nombre_del_fichero_es_el_uuid(tmp_path: Path) -> None:
    """El UUID hace de validacion: no hay forma de salir del directorio."""
    store = FilesystemBlobStore(root=tmp_path)
    image_id = uuid4()

    ruta: Callable[..., Path] = store._path

    assert ruta(image_id) == tmp_path / str(image_id)
