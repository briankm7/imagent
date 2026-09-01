"""VectorStore en memoria.

Principio que guia este fichero: **un fake mas permisivo que la cosa real es
una trampa.** Si acepta lo que Qdrant rechaza, tus tests pasan en verde y el
fallo aparece la primera vez que arrancas con Qdrant de verdad, que es
exactamente cuando menos lo quieres. Por eso aqui se replican dos rechazos que
no cuestan nada y que Qdrant hace:

- buscar o escribir sin que la coleccion exista;
- escribir un vector con una dimension distinta a la de la coleccion.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence
from uuid import UUID

from imagent.domain.errors import VectorStoreError
from imagent.domain.models import Aspect
from imagent.providers.base import IndexedPoint, ScoredPoint, Vector


def cosine(a: Vector, b: Vector) -> float:
    """Coseno entre dos vectores.

    No se asume que lleguen normalizados aunque el fake de embeddings los
    normalice: el contrato del almacen no dice nada de eso, y un almacen que
    solo funciona con vectores unitarios es un almacen con una precondicion
    secreta.
    """
    if len(a) != len(b):
        raise VectorStoreError(f"dimensiones incompatibles: {len(a)} y {len(b)}")

    producto = sum(x * y for x, y in zip(a, b, strict=True))
    norma_a = math.sqrt(sum(x * x for x in a))
    norma_b = math.sqrt(sum(x * x for x in b))
    if norma_a == 0.0 or norma_b == 0.0:
        return 0.0
    return producto / (norma_a * norma_b)


class InMemoryVectorStore:
    """Implementa VectorStore sin salir del proceso."""

    def __init__(self, dimensions: int) -> None:
        self._dimensions = dimensions
        self._points: dict[UUID, IndexedPoint] = {}
        self._ready = False

    async def ensure_ready(self) -> None:
        self._ready = True

    def _require_ready(self) -> None:
        if not self._ready:
            raise VectorStoreError("la coleccion no existe; falta llamar a ensure_ready()")

    async def upsert(self, points: Sequence[IndexedPoint]) -> None:
        self._require_ready()
        for point in points:
            if len(point.vector) != self._dimensions:
                raise VectorStoreError(
                    f"vector de dimension {len(point.vector)}, "
                    f"la coleccion espera {self._dimensions}"
                )
        # Se valida TODO el lote antes de escribir nada: un upsert que deja la
        # mitad de los puntos dentro es peor que uno que falla entero.
        for point in points:
            self._points[point.point_id] = point

    async def search(
        self,
        vector: Vector,
        *,
        limit: int,
        aspects: Collection[Aspect] | None = None,
        image_ids: Collection[UUID] | None = None,
    ) -> list[ScoredPoint]:
        self._require_ready()
        if limit < 1:
            raise VectorStoreError("limit tiene que ser positivo")

        aspectos = set(aspects) if aspects is not None else None
        ids = set(image_ids) if image_ids is not None else None

        candidatos = [
            point
            for point in self._points.values()
            if (aspectos is None or point.aspect in aspectos)
            and (ids is None or point.image_id in ids)
        ]

        puntuados = [
            ScoredPoint(
                point_id=point.point_id,
                image_id=point.image_id,
                aspect=point.aspect,
                text=point.text,
                score=cosine(vector, point.vector),
            )
            for point in candidatos
        ]

        # Desempate por point_id para que el orden sea reproducible. Ojo: Qdrant
        # NO garantiza el orden de los empates, asi que un test que dependa del
        # desempate pasaria aqui y seria inestable en produccion.
        puntuados.sort(key=lambda p: (-p.score, str(p.point_id)))
        return puntuados[:limit]

    async def delete_image(self, image_id: UUID) -> None:
        self._require_ready()
        for point_id in [pid for pid, p in self._points.items() if p.image_id == image_id]:
            del self._points[point_id]

    def snapshot(self) -> list[IndexedPoint]:
        """Solo para tests. No forma parte del Protocol VectorStore."""
        return sorted(self._points.values(), key=lambda p: str(p.point_id))
