"""Tests del almacen de vectores en memoria.

Buena parte de estos tests defienden que el fake sea tan ESTRICTO como Qdrant.
Un fake permisivo deja pasar bugs hasta el primer arranque real.
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from imagent.domain.errors import VectorStoreError
from imagent.domain.models import Aspect
from imagent.providers.base import IndexedPoint
from imagent.providers.fake.vectorstore import InMemoryVectorStore, cosine

DIMENSIONS = 4


@pytest.fixture
async def store() -> InMemoryVectorStore:
    store = InMemoryVectorStore(dimensions=DIMENSIONS)
    await store.ensure_ready()
    return store


def punto(image_id, aspect: Aspect, vector: list[float], text: str = "") -> IndexedPoint:
    return IndexedPoint.for_aspect(image_id=image_id, aspect=aspect, text=text, vector=vector)


async def test_buscar_sin_coleccion_falla() -> None:
    """Qdrant rechaza buscar en una coleccion que no existe; el fake tambien."""
    virgen = InMemoryVectorStore(dimensions=DIMENSIONS)

    with pytest.raises(VectorStoreError, match="ensure_ready"):
        await virgen.search([1.0, 0.0, 0.0, 0.0], limit=3)


async def test_escribir_sin_coleccion_falla() -> None:
    virgen = InMemoryVectorStore(dimensions=DIMENSIONS)

    with pytest.raises(VectorStoreError, match="ensure_ready"):
        await virgen.upsert([punto(uuid4(), Aspect.DESCRIPTION, [1.0, 0.0, 0.0, 0.0])])


async def test_ensure_ready_es_idempotente(store: InMemoryVectorStore) -> None:
    await store.upsert([punto(uuid4(), Aspect.DESCRIPTION, [1.0, 0.0, 0.0, 0.0])])
    await store.ensure_ready()

    assert len(store.snapshot()) == 1


async def test_dimension_incorrecta_se_rechaza(store: InMemoryVectorStore) -> None:
    """Qdrant rechaza el vector de dimension equivocada. Si el fake lo aceptase,
    el fallo aparecerian en el primer arranque real y no en el CI."""
    with pytest.raises(VectorStoreError, match="dimension"):
        await store.upsert([punto(uuid4(), Aspect.DESCRIPTION, [1.0, 0.0])])


async def test_un_lote_invalido_no_escribe_nada(store: InMemoryVectorStore) -> None:
    """Todo o nada: un upsert a medias es peor que uno que falla entero."""
    bueno = punto(uuid4(), Aspect.DESCRIPTION, [1.0, 0.0, 0.0, 0.0])
    malo = punto(uuid4(), Aspect.OCR, [1.0])

    with pytest.raises(VectorStoreError):
        await store.upsert([bueno, malo])

    assert store.snapshot() == []


async def test_reindexar_sobrescribe_en_vez_de_duplicar(store: InMemoryVectorStore) -> None:
    """La consecuencia practica del id determinista de IndexedPoint.for_aspect."""
    image_id = uuid4()

    await store.upsert([punto(image_id, Aspect.OCR, [1.0, 0.0, 0.0, 0.0], "SE VENDE")])
    await store.upsert([punto(image_id, Aspect.OCR, [0.0, 1.0, 0.0, 0.0], "SE ALQUILA")])

    puntos = store.snapshot()
    assert len(puntos) == 1
    assert puntos[0].text == "SE ALQUILA"


async def test_los_aspectos_de_una_imagen_son_puntos_distintos(
    store: InMemoryVectorStore,
) -> None:
    """Decision D2-B: un punto por (imagen, aspecto)."""
    image_id = uuid4()

    await store.upsert(
        [
            punto(image_id, Aspect.DESCRIPTION, [1.0, 0.0, 0.0, 0.0]),
            punto(image_id, Aspect.OCR, [0.0, 1.0, 0.0, 0.0]),
            punto(image_id, Aspect.OBJECTS, [0.0, 0.0, 1.0, 0.0]),
        ]
    )

    assert len(store.snapshot()) == 3


async def test_ordena_por_puntuacion_y_respeta_el_limite(store: InMemoryVectorStore) -> None:
    cerca, medio, lejos = uuid4(), uuid4(), uuid4()
    await store.upsert(
        [
            punto(cerca, Aspect.DESCRIPTION, [1.0, 0.0, 0.0, 0.0]),
            punto(medio, Aspect.DESCRIPTION, [0.7, 0.7, 0.0, 0.0]),
            punto(lejos, Aspect.DESCRIPTION, [0.0, 1.0, 0.0, 0.0]),
        ]
    )

    resultados = await store.search([1.0, 0.0, 0.0, 0.0], limit=2)

    assert [r.image_id for r in resultados] == [cerca, medio]
    assert resultados[0].score > resultados[1].score


async def test_filtra_por_aspecto(store: InMemoryVectorStore) -> None:
    image_id = uuid4()
    await store.upsert(
        [
            punto(image_id, Aspect.DESCRIPTION, [1.0, 0.0, 0.0, 0.0], "descripcion"),
            punto(image_id, Aspect.OCR, [1.0, 0.0, 0.0, 0.0], "SE VENDE"),
        ]
    )

    resultados = await store.search([1.0, 0.0, 0.0, 0.0], limit=10, aspects=[Aspect.OCR])

    assert [r.text for r in resultados] == ["SE VENDE"]


async def test_filtra_por_imagen(store: InMemoryVectorStore) -> None:
    """El filtro que usa la recuperacion cuando el repositorio ya ha acotado."""
    quiero, no_quiero = uuid4(), uuid4()
    await store.upsert(
        [
            punto(quiero, Aspect.DESCRIPTION, [1.0, 0.0, 0.0, 0.0]),
            punto(no_quiero, Aspect.DESCRIPTION, [1.0, 0.0, 0.0, 0.0]),
        ]
    )

    resultados = await store.search([1.0, 0.0, 0.0, 0.0], limit=10, image_ids=[quiero])

    assert [r.image_id for r in resultados] == [quiero]


async def test_los_filtros_se_combinan(store: InMemoryVectorStore) -> None:
    quiero, otra = uuid4(), uuid4()
    await store.upsert(
        [
            punto(quiero, Aspect.DESCRIPTION, [1.0, 0.0, 0.0, 0.0], "no"),
            punto(quiero, Aspect.OCR, [1.0, 0.0, 0.0, 0.0], "si"),
            punto(otra, Aspect.OCR, [1.0, 0.0, 0.0, 0.0], "no"),
        ]
    )

    resultados = await store.search(
        [1.0, 0.0, 0.0, 0.0], limit=10, aspects=[Aspect.OCR], image_ids=[quiero]
    )

    assert [r.text for r in resultados] == ["si"]


async def test_un_almacen_vacio_devuelve_lista_vacia(store: InMemoryVectorStore) -> None:
    """Sin resultados NO es un error: es lo que dispara la escalada a vision."""
    assert await store.search([1.0, 0.0, 0.0, 0.0], limit=5) == []


async def test_limite_no_positivo_se_rechaza(store: InMemoryVectorStore) -> None:
    with pytest.raises(VectorStoreError, match="limit"):
        await store.search([1.0, 0.0, 0.0, 0.0], limit=0)


async def test_borrar_una_imagen_borra_todos_sus_aspectos(store: InMemoryVectorStore) -> None:
    borrar, mantener = uuid4(), uuid4()
    await store.upsert(
        [
            punto(borrar, Aspect.DESCRIPTION, [1.0, 0.0, 0.0, 0.0]),
            punto(borrar, Aspect.OCR, [1.0, 0.0, 0.0, 0.0]),
            punto(mantener, Aspect.DESCRIPTION, [1.0, 0.0, 0.0, 0.0]),
        ]
    )

    await store.delete_image(borrar)

    assert [p.image_id for p in store.snapshot()] == [mantener]


async def test_borrar_algo_que_no_existe_no_falla(store: InMemoryVectorStore) -> None:
    """Idempotente: el reintento de una limpieza no puede reventar."""
    await store.delete_image(uuid4())


def test_el_coseno_rechaza_dimensiones_incompatibles() -> None:
    with pytest.raises(VectorStoreError, match="dimensiones"):
        cosine([1.0, 0.0], [1.0, 0.0, 0.0])


def test_el_coseno_de_un_vector_nulo_es_cero_y_no_explota() -> None:
    assert cosine([0.0, 0.0], [1.0, 0.0]) == 0.0
