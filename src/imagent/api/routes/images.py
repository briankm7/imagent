"""Subida y consulta de imagenes."""

from __future__ import annotations

import logging
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, File, HTTPException, Response, UploadFile, status

from imagent.api.deps import IngestionDep, ProvidersDep, SettingsDep
from imagent.api.schemas import ImageOut, UploadOut
from imagent.domain.errors import InvalidImageError, StorageError
from imagent.observability.logging import log_extra

router = APIRouter(prefix="/api/images", tags=["images"])
logger = logging.getLogger(__name__)

TROZO = 64 * 1024


@router.post("", status_code=status.HTTP_201_CREATED)
async def subir(
    ingestion: IngestionDep,
    settings: SettingsDep,
    *,
    file: Annotated[UploadFile, File()],
) -> UploadOut:
    """Sube una imagen y la indexa.

    Sincrona (decision D7): el usuario sabe cuando esta lista. La maquina de
    estados de ImageRecord ya esta modelada, asi que pasar esto a segundo plano
    el dia que haga falta es cambiar donde se llama, no como funciona.
    """
    datos = await _leer_con_limite(file, settings.max_upload_bytes)

    resultado = await ingestion.ingest(
        filename=file.filename or "sin-nombre",
        media_type=file.content_type or "application/octet-stream",
        data=datos,
    )

    logger.info(
        "imagen ingerida",
        extra=log_extra(
            image_id=str(resultado.record.id),
            deduplicated=resultado.deduplicated,
            size_bytes=resultado.record.size_bytes,
        ),
    )
    return UploadOut.de(resultado)


async def _leer_con_limite(file: UploadFile, maximo: int) -> bytes:
    """Lee el fichero abortando en cuanto se pasa del limite.

    A trozos y no de golpe: `await file.read()` traeria a memoria un fichero de
    cualquier tamaño ANTES de poder rechazarlo, que es justo lo que el limite
    intenta evitar. Este es el sitio donde se comprueba; el servicio de ingesta
    ya no puede ahorrar nada.
    """
    trozos: list[bytes] = []
    total = 0

    while trozo := await file.read(TROZO):
        total += len(trozo)
        if total > maximo:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"la imagen supera el limite de {maximo} bytes",
            )
        trozos.append(trozo)

    return b"".join(trozos)


@router.get("")
async def listar(providers: ProvidersDep) -> list[ImageOut]:
    """Todas las imagenes, en el orden en que se subieron.

    Se incluyen las fallidas: una imagen que no se pudo indexar tiene que verse,
    y verse por que. Ocultarlas dejaria al usuario con un hueco invisible.
    """
    return [ImageOut.de(r) for r in await providers.repository.list()]


@router.get("/{image_id}")
async def obtener(image_id: UUID, providers: ProvidersDep) -> ImageOut:
    record = await providers.repository.get(image_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no existe esa imagen")
    return ImageOut.de(record)


@router.get("/{image_id}/bytes")
async def bytes_de(image_id: UUID, providers: ProvidersDep) -> Response:
    """Los bytes originales, para que el front pueda enseñar la miniatura."""
    record = await providers.repository.get(image_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="no existe esa imagen")

    try:
        datos = await providers.blobs.get_bytes(image_id)
    except StorageError as exc:
        # El registro existe pero los bytes no: es la divergencia entre los dos
        # almacenes, y se dice en vez de devolver un 404 que sugeriria que la
        # imagen nunca existio.
        logger.warning("bytes ausentes", extra=log_extra(image_id=str(image_id)))
        raise HTTPException(
            status_code=status.HTTP_410_GONE,
            detail="el registro existe pero sus bytes no estan disponibles",
        ) from exc

    return Response(content=datos, media_type=record.media_type)


__all__ = ["InvalidImageError", "router"]
