"""VectorStore real sobre Qdrant.

Importable sin `qdrant-client`, por el mismo motivo que el adaptador de Gemini:
el SDK entra dentro de los metodos y asi el test de conformidad puede comprobar
en CI que esta clase cumple el Protocol aunque el cliente no este instalado.
"""

from __future__ import annotations

import asyncio
from collections.abc import Collection, Sequence
from typing import Any
from uuid import UUID

from imagent.domain.errors import VectorStoreError
from imagent.domain.models import Aspect
from imagent.providers.base import IndexedPoint, ScoredPoint, Vector

CLAVE_IMAGEN = "image_id"
CLAVE_ASPECTO = "aspect"
CLAVE_TEXTO = "text"


class QdrantVectorStore:
    """Implementa VectorStore contra una coleccion de Qdrant.

    El payload que se guarda es MINIMO: image_id, aspecto y el texto indexado.
    Los metadatos (nombre de fichero, estado, fechas) viven en el repositorio
    (decision P8-B), asi que aqui no se duplican. Duplicarlos significaria tres
    copias por imagen -una por aspecto- que pueden contradecirse entre si y con
    el repositorio.
    """

    def __init__(self, *, url: str, collection: str, dimensions: int) -> None:
        self._collection = collection
        self._dimensions = dimensions
        self._cliente = self._construir(url)

    def _construir(self, url: str) -> Any:
        try:
            from qdrant_client import AsyncQdrantClient
        except ImportError as exc:  # pragma: no cover - depende de la instalacion
            raise VectorStoreError(
                "falta el extra 'qdrant'. Instala con: pip install '.[qdrant]'"
            ) from exc

        return AsyncQdrantClient(url=url)

    async def ensure_ready(self) -> None:
        from qdrant_client import models

        try:
            if await self._cliente.collection_exists(self._collection):
                return
            await self._cliente.create_collection(
                collection_name=self._collection,
                vectors_config=models.VectorParams(
                    size=self._dimensions,
                    # Coseno porque los vectores se normalizan al salir del
                    # proveedor de embeddings. Con DOT, un vector mal
                    # normalizado daria puntuaciones que parecen validas.
                    distance=models.Distance.COSINE,
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"{type(exc).__name__}: {exc}") from exc

    async def upsert(self, points: Sequence[IndexedPoint]) -> None:
        from qdrant_client import models

        if not points:
            return

        # Se valida el lote entero antes de mandar nada: un upsert a medias es
        # peor que uno que falla del todo. Qdrant tambien lo rechazaria, pero
        # entonces algunos puntos ya estarian dentro.
        for punto in points:
            if len(punto.vector) != self._dimensions:
                raise VectorStoreError(
                    f"vector de dimension {len(punto.vector)}, "
                    f"la coleccion espera {self._dimensions}"
                )

        cuerpo = [
            models.PointStruct(
                # El id es el uuid5 determinista de (imagen, aspecto): reindexar
                # sobrescribe en vez de duplicar, y por eso la ingesta puede
                # escribir aqui ANTES de marcar el registro como indexado.
                id=str(punto.point_id),
                vector=list(punto.vector),
                payload={
                    CLAVE_IMAGEN: str(punto.image_id),
                    CLAVE_ASPECTO: punto.aspect.value,
                    CLAVE_TEXTO: punto.text,
                },
            )
            for punto in points
        ]

        try:
            await self._cliente.upsert(collection_name=self._collection, points=cuerpo)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"{type(exc).__name__}: {exc}") from exc

    async def search(
        self,
        vector: Vector,
        *,
        limit: int,
        aspects: Collection[Aspect] | None = None,
        image_ids: Collection[UUID] | None = None,
    ) -> list[ScoredPoint]:
        if limit < 1:
            raise VectorStoreError("limit tiene que ser positivo")

        try:
            respuesta = await self._cliente.query_points(
                collection_name=self._collection,
                query=list(vector),
                limit=limit,
                query_filter=self._filtro(aspects, image_ids),
                with_payload=True,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"{type(exc).__name__}: {exc}") from exc

        return [self._a_scored(p) for p in respuesta.points]

    def _filtro(
        self, aspects: Collection[Aspect] | None, image_ids: Collection[UUID] | None
    ) -> Any:
        """Traduce los DOS filtros del contrato a la sintaxis de Qdrant.

        Solo dos, y por eso esta funcion cabe en diez lineas. Si la interfaz
        aceptara filtros arbitrarios, aqui habria un traductor entero y el fake
        en memoria tendria que implementar el mismo lenguaje para servir de algo.
        """
        from qdrant_client import models

        condiciones = []
        if aspects is not None:
            condiciones.append(
                models.FieldCondition(
                    key=CLAVE_ASPECTO,
                    match=models.MatchAny(any=[a.value for a in aspects]),
                )
            )
        if image_ids is not None:
            condiciones.append(
                models.FieldCondition(
                    key=CLAVE_IMAGEN,
                    match=models.MatchAny(any=[str(i) for i in image_ids]),
                )
            )

        return models.Filter(must=condiciones) if condiciones else None

    def _a_scored(self, punto: Any) -> ScoredPoint:
        payload = punto.payload or {}
        try:
            return ScoredPoint(
                point_id=UUID(str(punto.id)),
                image_id=UUID(payload[CLAVE_IMAGEN]),
                aspect=Aspect(payload[CLAVE_ASPECTO]),
                text=payload.get(CLAVE_TEXTO, ""),
                score=float(punto.score),
            )
        except (KeyError, ValueError) as exc:
            # Un punto sin el payload que este proyecto escribe es un punto que
            # escribio otra cosa en la misma coleccion. Fallar es correcto:
            # devolverlo a medias contaminaria la evidencia sin avisar.
            raise VectorStoreError(f"punto con payload inesperado: {exc}") from exc

    async def delete_image(self, image_id: UUID) -> None:
        from qdrant_client import models

        try:
            await self._cliente.delete(
                collection_name=self._collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(
                        must=[
                            models.FieldCondition(
                                key=CLAVE_IMAGEN,
                                match=models.MatchValue(value=str(image_id)),
                            )
                        ]
                    )
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"{type(exc).__name__}: {exc}") from exc
