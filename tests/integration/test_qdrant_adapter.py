"""Tests del adaptador de Qdrant, sin levantar Qdrant.

Mismo planteamiento que los de Gemini: se usan los tipos reales del SDK
-`PointStruct`, `Filter`, `VectorParams` los valida pydantic al construirlos- con
un cliente sustituido por un doble que anota lo que recibe.

Lo que no cubren: que un Qdrant de verdad acepte lo que se le manda. Para eso
esta `python -m imagent.check` con `IMAGENT_VECTOR_STORE_MODE=qdrant`.

Se saltan en CI, donde el extra `qdrant` no esta instalado a proposito.
"""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

pytest.importorskip("qdrant_client", reason="el extra 'qdrant' no esta instalado")

from imagent.domain.errors import VectorStoreError
from imagent.domain.models import Aspect
from imagent.providers.base import IndexedPoint
from imagent.providers.real.qdrant import (
    CLAVE_ASPECTO,
    CLAVE_IMAGEN,
    CLAVE_TEXTO,
    QdrantVectorStore,
)

DIMENSIONES = 4


class _PuntoFalso:
    def __init__(self, id_: Any, score: float, payload: dict[str, Any] | None) -> None:
        self.id = id_
        self.score = score
        self.payload = payload


class _ClienteFalso:
    """Anota las llamadas y devuelve lo que se le programe."""

    def __init__(self, *, existe: bool = True, puntos: list[_PuntoFalso] | None = None) -> None:
        self.existe = existe
        self.puntos = puntos or []
        self.error: Exception | None = None
        self.llamadas: list[tuple[str, dict[str, Any]]] = []

    def _anotar(self, nombre: str, kwargs: dict[str, Any]) -> None:
        self.llamadas.append((nombre, kwargs))
        if self.error is not None:
            raise self.error

    def de(self, nombre: str) -> dict[str, Any]:
        return next(k for n, k in self.llamadas if n == nombre)

    def hubo(self, nombre: str) -> bool:
        return any(n == nombre for n, _ in self.llamadas)

    async def collection_exists(self, collection_name: str, **_: Any) -> bool:
        self._anotar("collection_exists", {"collection_name": collection_name})
        return self.existe

    async def create_collection(self, **kwargs: Any) -> None:
        self._anotar("create_collection", kwargs)

    async def upsert(self, **kwargs: Any) -> None:
        self._anotar("upsert", kwargs)

    async def query_points(self, **kwargs: Any) -> Any:
        self._anotar("query_points", kwargs)
        return type("R", (), {"points": self.puntos})()

    async def delete(self, **kwargs: Any) -> None:
        self._anotar("delete", kwargs)


@pytest.fixture
def montar(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    def _montar(cliente: _ClienteFalso) -> QdrantVectorStore:
        monkeypatch.setattr(QdrantVectorStore, "_construir", lambda self, url: cliente)
        return QdrantVectorStore(
            url="http://irrelevante", collection="imagenes", dimensions=DIMENSIONES
        )

    return _montar


def punto(image_id: Any, aspecto: Aspect, texto: str = "un coche rojo") -> IndexedPoint:
    return IndexedPoint.for_aspect(
        image_id=image_id, aspect=aspecto, text=texto, vector=[1.0, 0.0, 0.0, 0.0]
    )


# ---------------------------------------------------------------------------
# Creacion de la coleccion
# ---------------------------------------------------------------------------
async def test_crea_la_coleccion_con_coseno_y_la_dimension(montar) -> None:  # type: ignore[no-untyped-def]
    """Coseno porque los vectores salen normalizados del proveedor de
    embeddings. Con producto escalar, un vector mal normalizado daria
    puntuaciones que parecen validas."""
    from qdrant_client import models

    cliente = _ClienteFalso(existe=False)

    await montar(cliente).ensure_ready()

    config = cliente.de("create_collection")["vectors_config"]
    assert config.size == DIMENSIONES
    assert config.distance == models.Distance.COSINE


async def test_no_recrea_una_coleccion_que_ya_existe(montar) -> None:  # type: ignore[no-untyped-def]
    """`ensure_ready` se llama una vez por instancia del servicio de ingesta,
    pero tiene que ser idempotente igualmente."""
    cliente = _ClienteFalso(existe=True)

    await montar(cliente).ensure_ready()

    assert not cliente.hubo("create_collection")


# ---------------------------------------------------------------------------
# Escritura
# ---------------------------------------------------------------------------
async def test_el_punto_lleva_el_id_determinista_y_el_payload_minimo(montar) -> None:  # type: ignore[no-untyped-def]
    """El payload NO duplica los metadatos: esos viven en el repositorio. Tres
    copias por imagen -una por aspecto- podrian contradecirse entre si."""
    cliente = _ClienteFalso()
    image_id = uuid4()
    p = punto(image_id, Aspect.OCR, "SE VENDE")

    await montar(cliente).upsert([p])

    (enviado,) = cliente.de("upsert")["points"]
    # Qdrant solo acepta enteros o UUID como id de punto: por eso el id es un
    # uuid5 de (imagen, aspecto) y no la cadena "id:aspecto".
    assert enviado.id == str(p.point_id)
    assert enviado.payload == {
        CLAVE_IMAGEN: str(image_id),
        CLAVE_ASPECTO: "ocr",
        CLAVE_TEXTO: "SE VENDE",
    }


async def test_una_dimension_equivocada_se_rechaza_antes_de_enviar(montar) -> None:  # type: ignore[no-untyped-def]
    """Qdrant tambien la rechazaria, pero entonces los puntos validos del mismo
    lote ya estarian dentro."""
    cliente = _ClienteFalso()
    malo = IndexedPoint.for_aspect(
        image_id=uuid4(), aspect=Aspect.DESCRIPTION, text="x", vector=[1.0]
    )

    with pytest.raises(VectorStoreError, match="dimension"):
        await montar(cliente).upsert([punto(uuid4(), Aspect.OCR), malo])

    assert not cliente.hubo("upsert")


async def test_un_lote_vacio_no_llama_al_cliente(montar) -> None:  # type: ignore[no-untyped-def]
    """Una imagen sin ningun aspecto indexable no tiene por que costar un viaje."""
    cliente = _ClienteFalso()

    await montar(cliente).upsert([])

    assert cliente.llamadas == []


# ---------------------------------------------------------------------------
# Busqueda
# ---------------------------------------------------------------------------
async def test_la_busqueda_traduce_los_dos_filtros(montar) -> None:  # type: ignore[no-untyped-def]
    """Solo dos filtros, y por eso la traduccion cabe en diez lineas. Con
    filtros arbitrarios haria falta un traductor entero, y el fake en memoria
    tendria que implementar el mismo lenguaje para servir de algo."""
    cliente = _ClienteFalso()
    image_id = uuid4()

    await montar(cliente).search(
        [1.0, 0.0, 0.0, 0.0], limit=5, aspects=[Aspect.OCR], image_ids=[image_id]
    )

    llamada = cliente.de("query_points")
    assert llamada["limit"] == 5
    condiciones = {c.key: c.match.any for c in llamada["query_filter"].must}
    assert condiciones == {CLAVE_ASPECTO: ["ocr"], CLAVE_IMAGEN: [str(image_id)]}


async def test_sin_filtros_no_se_manda_ninguno(montar) -> None:  # type: ignore[no-untyped-def]
    cliente = _ClienteFalso()

    await montar(cliente).search([1.0, 0.0, 0.0, 0.0], limit=3)

    assert cliente.de("query_points")["query_filter"] is None


async def test_la_respuesta_se_convierte_al_modelo_del_dominio(montar) -> None:  # type: ignore[no-untyped-def]
    image_id, point_id = uuid4(), uuid4()
    cliente = _ClienteFalso(
        puntos=[
            _PuntoFalso(
                str(point_id),
                0.87,
                {
                    CLAVE_IMAGEN: str(image_id),
                    CLAVE_ASPECTO: "description",
                    CLAVE_TEXTO: "un coche",
                },
            )
        ]
    )

    (encontrado,) = await montar(cliente).search([1.0, 0.0, 0.0, 0.0], limit=1)

    assert encontrado.point_id == point_id
    assert encontrado.image_id == image_id
    assert encontrado.aspect is Aspect.DESCRIPTION
    assert encontrado.text == "un coche"
    assert encontrado.score == pytest.approx(0.87)


async def test_un_punto_con_payload_ajeno_es_un_fallo(montar) -> None:  # type: ignore[no-untyped-def]
    """Un punto sin el payload que este proyecto escribe lo escribio otra cosa
    en la misma coleccion. Devolverlo a medias contaminaria la evidencia."""
    cliente = _ClienteFalso(puntos=[_PuntoFalso(str(uuid4()), 0.9, {"otra_cosa": "x"})])

    with pytest.raises(VectorStoreError, match="payload inesperado"):
        await montar(cliente).search([1.0, 0.0, 0.0, 0.0], limit=1)


async def test_un_limite_no_positivo_se_rechaza(montar) -> None:  # type: ignore[no-untyped-def]
    cliente = _ClienteFalso()

    with pytest.raises(VectorStoreError, match="limit"):
        await montar(cliente).search([1.0, 0.0, 0.0, 0.0], limit=0)

    assert cliente.llamadas == []


# ---------------------------------------------------------------------------
# Borrado y fallos
# ---------------------------------------------------------------------------
async def test_borrar_una_imagen_filtra_por_su_id(montar) -> None:  # type: ignore[no-untyped-def]
    cliente = _ClienteFalso()
    image_id = uuid4()

    await montar(cliente).delete_image(image_id)

    selector = cliente.de("delete")["points_selector"]
    (condicion,) = selector.filter.must
    assert condicion.key == CLAVE_IMAGEN
    assert condicion.match.value == str(image_id)


async def test_un_fallo_del_cliente_sale_como_vector_store_error(montar) -> None:  # type: ignore[no-untyped-def]
    cliente = _ClienteFalso()
    cliente.error = ConnectionRefusedError("connection refused")

    with pytest.raises(VectorStoreError, match="connection refused"):
        await montar(cliente).search([1.0, 0.0, 0.0, 0.0], limit=1)


async def test_una_cancelacion_no_se_convierte_en_vector_store_error(montar) -> None:  # type: ignore[no-untyped-def]
    import asyncio

    cliente = _ClienteFalso()
    cliente.error = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await montar(cliente).search([1.0, 0.0, 0.0, 0.0], limit=1)
