"""ImageRepository en memoria."""

from __future__ import annotations

from collections.abc import Collection
from datetime import datetime
from uuid import UUID

from imagent.domain.models import ImageRecord, ImageStatus


class InMemoryImageRepository:
    """Implementa ImageRepository con un diccionario.

    Guardar los registros tal cual es seguro porque `ImageRecord` es inmutable:
    nadie de fuera puede modificar lo que hay dentro del repositorio sin pasar
    por `save`. Con un modelo mutable habria que copiar defensivamente en cada
    lectura, o aceptar que el repositorio y sus lectores compartan objeto.
    """

    def __init__(self) -> None:
        self._records: dict[UUID, ImageRecord] = {}
        self._orden: dict[UUID, int] = {}

    async def save(self, record: ImageRecord) -> None:
        # El numero de orden se asigna la PRIMERA vez que se ve el registro y no
        # cambia con las actualizaciones de estado: es el orden de subida, no el
        # de la ultima escritura.
        self._orden.setdefault(record.id, len(self._orden))
        self._records[record.id] = record

    def _clave(self, record: ImageRecord) -> tuple[datetime, int]:
        return (record.created_at, self._orden[record.id])

    async def get(self, image_id: UUID) -> ImageRecord | None:
        return self._records.get(image_id)

    async def find_by_hash(self, content_hash: str) -> ImageRecord | None:
        """El registro MAS ANTIGUO con ese contenido.

        El esquema permite varios (decision P3-C), asi que hay que decir cual se
        devuelve. Se elige el mas antiguo porque es el que ya tiene el analisis
        pagado: es el candidato util para deduplicar.
        """
        coincidencias = [r for r in self._records.values() if r.content_hash == content_hash]
        if not coincidencias:
            return None
        return min(coincidencias, key=self._clave)

    async def list(
        self,
        *,
        statuses: Collection[ImageStatus] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> list[ImageRecord]:
        estados = set(statuses) if statuses is not None else None

        seleccion = [
            record
            for record in self._records.values()
            if (estados is None or record.status in estados)
            and (created_after is None or record.created_at >= created_after)
            and (created_before is None or record.created_at <= created_before)
        ]

        # El orden es parte del contrato: "la tercera imagen" depende de el.
        #
        # El desempate NO va por id. Un UUID es aleatorio, asi que dos imagenes
        # subidas en el mismo instante saldrian en un orden distinto en cada
        # ejecucion, y "la tercera" dejaria de significar nada. Va por orden de
        # insercion, que es lo que realmente se quiere decir con "orden de
        # subida"; la marca de tiempo sola no tiene resolucion suficiente para
        # expresarlo.
        return sorted(seleccion, key=self._clave)
