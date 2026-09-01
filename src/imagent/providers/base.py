"""Contratos con el mundo exterior.

Todo lo que sale del proceso pasa por aqui. `agents/` y `services/` dependen de
estos Protocol, nunca de una implementacion concreta: es lo que permite que el
CI corra sin claves y sin base de datos.

Se usan `Protocol` y no clases base abstractas (decision D4). El tipado es
estructural: un fake cumple el contrato por tener los metodos, sin heredar de
nada. Con ABC, cada fake arrastraria una jerarquia que solo existe para
satisfacer al type checker.

Aviso honesto: `Protocol` lo verifica un type checker estatico, y en este
proyecto no corremos mypy. Por eso hay un test de conformidad que compara
firmas en tiempo de ejecucion; sin el, un fake podria desviarse del contrato
sin que nada se quejara hasta produccion.
"""

from __future__ import annotations

import hashlib
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import cached_property
from typing import Protocol, TypeVar, runtime_checkable
from uuid import UUID, uuid5

from pydantic import BaseModel, ConfigDict

from imagent.domain.models import Aspect, ImageAnalysis, ImageRecord, ImageStatus

type Vector = list[float]

StructuredT = TypeVar("StructuredT", bound=BaseModel)


@dataclass(frozen=True)
class ImageBlob:
    """Los bytes de una imagen y su tipo. Lo unico que un proveedor de vision necesita.

    No lleva id ni nombre de fichero a proposito: un proveedor de vision no
    tiene por que saber nada del ciclo de vida de tu registro.
    """

    data: bytes
    media_type: str

    @cached_property
    def content_hash(self) -> str:
        """SHA-256 en hexadecimal. Es la identidad del contenido (decision P3-C)."""
        return hashlib.sha256(self.data).hexdigest()


class VisionProvider(Protocol):
    """El unico que mira imagenes de verdad. Es el trabajo caro.

    Dos metodos porque son los dos modos del agente de vision y tienen costes y
    contratos distintos:

    - `describe` se ejecuta UNA vez por imagen, en la ingesta, y su salida es lo
      que se indexa;
    - `answer_about` se ejecuta bajo demanda, N veces por consulta, y es lo que
      el presupuesto acota.
    """

    @property
    def model_name(self) -> str:
        """Que modelo hay detras. Va a parar a ImageAnalysis como procedencia."""
        ...

    async def describe(self, image: ImageBlob) -> ImageAnalysis:
        """Analisis denso de ingesta: descripcion, objetos y texto visible."""
        ...

    async def answer_about(self, image: ImageBlob, question: str) -> str:
        """Vuelve a mirar UNA imagen con UNA pregunta concreta."""
        ...


class TextProvider(Protocol):
    """Razonamiento y redaccion. No ve imagenes."""

    @property
    def model_name(self) -> str: ...

    async def complete(self, *, system: str, user: str) -> str:
        """Prosa libre. Lo usa el nodo `respond`."""
        ...

    async def structured(self, *, system: str, user: str, schema: type[StructuredT]) -> StructuredT:
        """Salida validada contra un modelo pydantic.

        Es la mitad del hibrido de la decision D3-C: el coordinador solo puede
        devolver una forma conocida, y el codigo se encarga del resto (validar,
        aplicar presupuesto, hacer clamp). El limite no vive en el prompt.
        """
        ...


class EmbeddingProvider(Protocol):
    """Texto a vectores."""

    @property
    def dimensions(self) -> int:
        """Longitud de los vectores. El almacen tiene que estar de acuerdo."""
        ...

    @property
    def model_name(self) -> str: ...

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        """Vectores para lo que se indexa."""
        ...

    async def embed_query(self, text: str) -> Vector:
        """Vector para lo que se pregunta.

        Metodo aparte y no un flag porque no es lo mismo: Gemini distingue
        RETRIEVAL_DOCUMENT de RETRIEVAL_QUERY, y usar el mismo para ambos
        empeora la recuperacion de forma medible. Tenerlo en la interfaz obliga
        a que quien implemente el proveedor real se entere.
        """
        ...


# ---------------------------------------------------------------------------
# Almacenamiento
# ---------------------------------------------------------------------------
class IndexedPoint(BaseModel):
    """Un punto listo para escribir. Decision D2-B: uno por (imagen, aspecto)."""

    model_config = ConfigDict(frozen=True)

    point_id: UUID
    image_id: UUID
    aspect: Aspect
    text: str
    vector: Vector

    @classmethod
    def for_aspect(
        cls, *, image_id: UUID, aspect: Aspect, text: str, vector: Vector
    ) -> IndexedPoint:
        """Construye el punto con un id DETERMINISTA derivado de (imagen, aspecto).

        Dos consecuencias que valen mucho mas que las tres lineas que cuesta:

        1. reindexar una imagen sobrescribe sus puntos en vez de duplicarlos, asi
           que el upsert es idempotente y se puede reintentar sin miedo;
        2. eso permite ordenar la ingesta de la forma segura: escribir primero en
           el almacen y marcar el registro INDEXED despues. Si el proceso muere en
           medio, quedan puntos huerfanos que el reintento sobrescribe. Al reves
           tendrias un registro que dice estar indexado sin puntos detras, que es
           un hueco invisible.

        Es uuid5 y no una cadena "id:aspecto" porque Qdrant solo acepta enteros o
        UUID como id de punto. El fake podria tragarse cualquier cadena y luego
        Qdrant la rechazaria en el peor momento.
        """
        return cls(
            point_id=uuid5(image_id, aspect.value),
            image_id=image_id,
            aspect=aspect,
            text=text,
            vector=vector,
        )


class ScoredPoint(BaseModel):
    """Un punto recuperado, con su puntuacion."""

    model_config = ConfigDict(frozen=True)

    point_id: UUID
    image_id: UUID
    aspect: Aspect
    text: str
    score: float


class VectorStore(Protocol):
    """Solo vectores. Los metadatos viven en ImageRepository (decision P8-B).

    La superficie de filtrado es deliberadamente minima: `aspects` e `image_ids`.
    Cualquier otro filtro ("las de ayer") lo resuelve antes el repositorio, que
    devuelve una lista de ids. Traducir filtros arbitrarios a la sintaxis nativa
    de Qdrant ataria la interfaz al proveedor y haria el fake imposible.
    """

    async def ensure_ready(self) -> None:
        """Crea la coleccion si no existe. Idempotente."""
        ...

    async def upsert(self, points: Sequence[IndexedPoint]) -> None:
        """Escribe o sobrescribe puntos por su id."""
        ...

    async def search(
        self,
        vector: Vector,
        *,
        limit: int,
        aspects: Collection[Aspect] | None = None,
        image_ids: Collection[UUID] | None = None,
    ) -> list[ScoredPoint]:
        """Los `limit` puntos mas parecidos, de mayor a menor puntuacion."""
        ...

    async def delete_image(self, image_id: UUID) -> None:
        """Borra todos los puntos de una imagen. No falla si no hay ninguno."""
        ...


class ImageRepository(Protocol):
    """Los registros de imagen y su estado.

    Existe separado del almacen de vectores porque un registro PENDING o FAILED
    no tiene vectores y no cabria en Qdrant, y son precisamente los estados que
    la decision P2-B existe para conservar.
    """

    async def save(self, record: ImageRecord) -> None:
        """Guarda o sobrescribe por id."""
        ...

    async def get(self, image_id: UUID) -> ImageRecord | None: ...

    async def find_by_hash(self, content_hash: str) -> ImageRecord | None:
        """Soporta la politica de deduplicacion (decision P3-C).

        El esquema no impide subir dos veces la misma imagen; es el servicio de
        ingesta quien decide que hacer al encontrarla.
        """
        ...

    async def list(
        self,
        *,
        statuses: Collection[ImageStatus] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> list[ImageRecord]:
        """Registros filtrados, SIEMPRE en orden de subida ascendente.

        El orden es parte del contrato, no un detalle: "la tercera imagen" de la
        conversacion depende de el.

        Se ordena por `created_at` y, a igualdad, por orden de insercion en el
        repositorio. El desempate NO puede ir por id: un UUID es aleatorio, asi
        que dos imagenes subidas en el mismo instante saldrian en un orden
        distinto en cada ejecucion. La marca de tiempo sola tampoco basta,
        porque su resolucion no distingue dos subidas seguidas.
        """
        ...


class BlobStore(Protocol):
    """Los bytes de las imagenes.

    Existe separado del repositorio porque son dos cosas con vidas distintas:
    los metadatos se leen y reescriben constantemente (cada transicion de
    estado), y los bytes se escriben una vez y se leen cuando el agente de
    vision vuelve a mirar.

    `get_bytes` devuelve bytes y no un ImageBlob a proposito: el media_type vive
    en el ImageRecord, y quien pide los bytes ya lo tiene. Guardarlo tambien aqui
    seria una segunda copia que puede contradecir a la primera.
    """

    async def put(self, image_id: UUID, blob: ImageBlob) -> None:
        """Guarda los bytes. Idempotente: reescribir el mismo id sobrescribe."""
        ...

    async def get_bytes(self, image_id: UUID) -> bytes:
        """Los bytes guardados. Lanza StorageError si no estan.

        No devuelve None: que falten los bytes de una imagen registrada es una
        inconsistencia, no un caso normal. Quien pueda seguir sin ellos lo dice
        explicitamente con degrade_on.
        """
        ...

    async def exists(self, image_id: UUID) -> bool: ...

    async def delete(self, image_id: UUID) -> None:
        """Borra los bytes. No falla si no existen."""
        ...


@runtime_checkable
class Closeable(Protocol):
    """Un proveedor con recursos que hay que soltar al apagar.

    No todos lo son -el de memoria no tiene nada que cerrar- asi que no forma
    parte de ninguna de las interfaces de arriba: es una capacidad opcional que
    se comprueba con isinstance.

    Existe por un fallo concreto: la conexion de aiosqlite vive en un hilo
    propio, y si nadie la cierra ese hilo sobrevive al event loop y suelta
    "Event loop is closed" al terminar el proceso. Un fallo al apagar es facil
    de ignorar y significa que hay un recurso que no se esta soltando.
    """

    async def close(self) -> None: ...


async def close_all(*objetos: object) -> None:
    """Cierra lo que se pueda cerrar, sin importar el orden ni si sobra."""
    for objeto in objetos:
        if isinstance(objeto, Closeable):
            await objeto.close()
