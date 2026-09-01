"""ImageRepository sobre SQLite.

A diferencia de los otros dos adaptadores reales, este SI se ejecuta en CI:
`aiosqlite` entra como dependencia del checkpointer, que es base. Asi que los
mismos tests de contrato que se pasan al repositorio en memoria se pasan a este,
que es la unica forma de saber que las dos implementaciones se comportan igual.

La tabla guarda el registro entero como JSON mas unas columnas indexadas para lo
que se filtra. Es un poco heterodoxo y es deliberado: el esquema de
`ImageRecord` va a cambiar mientras el proyecto crezca, y no quiero una
migracion por cada campo nuevo. Lo que se filtra son cuatro cosas y esas si
tienen columna.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Collection
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from imagent.domain.errors import RepositoryError
from imagent.domain.models import ImageRecord, ImageStatus

ESQUEMA = """
CREATE TABLE IF NOT EXISTS images (
    -- `seq` es el orden de subida. Es la pieza que hace que "la tercera imagen"
    -- signifique algo: created_at no tiene resolucion suficiente para
    -- distinguir dos subidas seguidas, y un UUID no tiene ninguna relacion con
    -- cuando llego la imagen. AUTOINCREMENT no reutiliza numeros de filas
    -- borradas, asi que el orden nunca se reordena hacia atras.
    seq          INTEGER PRIMARY KEY AUTOINCREMENT,
    id           TEXT    NOT NULL UNIQUE,
    content_hash TEXT    NOT NULL,
    status       TEXT    NOT NULL,
    created_at   TEXT    NOT NULL,
    record       TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_images_hash    ON images (content_hash);
CREATE INDEX IF NOT EXISTS idx_images_status  ON images (status);
CREATE INDEX IF NOT EXISTS idx_images_created ON images (created_at, seq);
"""


class SqliteImageRepository:
    """Implementa ImageRepository sobre un fichero sqlite."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._conexion: Any = None
        # La conexion se abre una vez, perezosamente. El cerrojo evita que dos
        # peticiones simultaneas creen dos conexiones y dos veces el esquema.
        self._cerrojo = asyncio.Lock()

    async def _conectar(self) -> Any:
        if self._conexion is not None:
            return self._conexion

        async with self._cerrojo:
            if self._conexion is not None:
                return self._conexion
            try:
                import aiosqlite

                self._path.parent.mkdir(parents=True, exist_ok=True)
                conexion = await aiosqlite.connect(self._path)
                # WAL: permite leer mientras se escribe. Sin esto, dos peticiones
                # concurrentes se bloquean entre si y con timeouts por agente
                # eso se convierte en fallos aparentemente aleatorios.
                await conexion.execute("PRAGMA journal_mode=WAL")
                await conexion.executescript(ESQUEMA)
                await conexion.commit()
                self._conexion = conexion
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                raise RepositoryError(f"no se pudo abrir {self._path}: {exc}") from exc

        return self._conexion

    async def close(self) -> None:
        if self._conexion is not None:
            await self._conexion.close()
            self._conexion = None

    async def save(self, record: ImageRecord) -> None:
        conexion = await self._conectar()
        try:
            await conexion.execute(
                """
                INSERT INTO images (id, content_hash, status, created_at, record)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    content_hash = excluded.content_hash,
                    status       = excluded.status,
                    created_at   = excluded.created_at,
                    record       = excluded.record
                """,
                # El ON CONFLICT no toca `seq`: una transicion de estado
                # actualiza el registro pero NO lo mueve de sitio en el orden de
                # subida. Es el mismo contrato que el repositorio en memoria.
                (
                    str(record.id),
                    record.content_hash,
                    record.status.value,
                    record.created_at.isoformat(),
                    record.model_dump_json(),
                ),
            )
            await conexion.commit()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise RepositoryError(f"no se pudo guardar {record.id}: {exc}") from exc

    async def get(self, image_id: UUID) -> ImageRecord | None:
        filas = await self._consultar("SELECT record FROM images WHERE id = ?", (str(image_id),))
        return _a_registro(filas[0][0]) if filas else None

    async def find_by_hash(self, content_hash: str) -> ImageRecord | None:
        filas = await self._consultar(
            """
            SELECT record FROM images
            WHERE content_hash = ?
            ORDER BY created_at, seq
            LIMIT 1
            """,
            (content_hash,),
        )
        return _a_registro(filas[0][0]) if filas else None

    async def list(
        self,
        *,
        statuses: Collection[ImageStatus] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> list[ImageRecord]:
        condiciones: list[str] = []
        parametros: list[Any] = []

        if statuses is not None:
            estados = list(statuses)
            if not estados:
                # Filtrar por "ninguno de estos estados" es una lista vacia, y
                # `IN ()` no es SQL valido. Se responde sin ir a la base.
                return []
            condiciones.append(f"status IN ({','.join('?' * len(estados))})")
            parametros.extend(e.value for e in estados)

        if created_after is not None:
            condiciones.append("created_at >= ?")
            parametros.append(created_after.isoformat())

        if created_before is not None:
            condiciones.append("created_at <= ?")
            parametros.append(created_before.isoformat())

        donde = f"WHERE {' AND '.join(condiciones)}" if condiciones else ""
        filas = await self._consultar(
            # El orden del contrato, resuelto por la base de datos: created_at y,
            # a igualdad, orden de insercion.
            f"SELECT record FROM images {donde} ORDER BY created_at, seq",
            tuple(parametros),
        )
        return [_a_registro(f[0]) for f in filas]

    async def _consultar(self, sql: str, parametros: tuple[Any, ...]) -> list[Any]:
        conexion = await self._conectar()
        try:
            async with conexion.execute(sql, parametros) as cursor:
                return list(await cursor.fetchall())
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise RepositoryError(f"consulta fallida: {exc}") from exc


def _a_registro(crudo: str) -> ImageRecord:
    """Reconstruye el registro validandolo.

    Se valida al LEER y no solo al escribir porque el fichero puede venir de una
    version anterior del esquema o de otra mano. Un registro que no cumple las
    invariantes es mejor que reviente aqui, con el JSON delante, que tres capas
    mas arriba.
    """
    try:
        return ImageRecord.model_validate(json.loads(crudo))
    except Exception as exc:
        raise RepositoryError(f"registro corrupto en la base de datos: {exc}") from exc
