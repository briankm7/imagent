"""Tests del fake de embeddings.

Los dos primeros defienden el determinismo. Los del final defienden lo unico
que hace util a este fake: que tenga estructura suficiente para que un test de
recuperacion pueda afirmar algo.
"""

from __future__ import annotations

import math

import pytest

from imagent.providers.fake.embeddings import FakeEmbeddingProvider, tokenize


def coseno(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


@pytest.fixture
def embeddings() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(dimensions=256)


async def test_el_mismo_texto_da_siempre_el_mismo_vector(
    embeddings: FakeEmbeddingProvider,
) -> None:
    assert await embeddings.embed_query("un coche rojo") == await embeddings.embed_query(
        "un coche rojo"
    )


async def test_dos_instancias_coinciden(embeddings: FakeEmbeddingProvider) -> None:
    """El determinismo tiene que sobrevivir entre procesos, no solo dentro de uno.

    Si se usara la hash() de Python esto pasaria dentro de un proceso y fallaria
    entre ejecuciones distintas.
    """
    otra = FakeEmbeddingProvider(dimensions=256)

    assert await embeddings.embed_query("cartel") == await otra.embed_query("cartel")


async def test_los_vectores_estan_normalizados(embeddings: FakeEmbeddingProvider) -> None:
    vector = await embeddings.embed_query("un cartel azul en una calle estrecha")

    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


async def test_respeta_la_dimension_configurada() -> None:
    proveedor = FakeEmbeddingProvider(dimensions=32)

    assert len(await proveedor.embed_query("x")) == 32
    assert proveedor.dimensions == 32


async def test_el_texto_vacio_no_produce_un_vector_nulo(
    embeddings: FakeEmbeddingProvider,
) -> None:
    """Un vector de ceros haria que el coseno fuese 0/0 en el almacen."""
    vector = await embeddings.embed_query("   !!!   ")

    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0, rel_tol=1e-9)


async def test_embed_documents_conserva_el_orden(embeddings: FakeEmbeddingProvider) -> None:
    vectores = await embeddings.embed_documents(["coche", "cartel", "coche"])

    assert vectores[0] == vectores[2]
    assert vectores[0] != vectores[1]


async def test_lo_parecido_puntua_mas_que_lo_distinto(
    embeddings: FakeEmbeddingProvider,
) -> None:
    """Esta es la razon de ser de P6-B.

    Sin esta propiedad, ningun test de recuperacion probaria nada.
    """
    consulta = await embeddings.embed_query("coche")
    con_coche = await embeddings.embed_query("un coche rojo aparcado en la acera")
    sin_coche = await embeddings.embed_query("un cartel azul en una calle estrecha")

    assert coseno(consulta, con_coche) > coseno(consulta, sin_coche)


async def test_los_acentos_no_cambian_el_vector(embeddings: FakeEmbeddingProvider) -> None:
    """En español esto no es opcional."""
    assert await embeddings.embed_query("¿qué pone en el cartel?") == await embeddings.embed_query(
        "que pone en el cartel"
    )


def test_tokenize_parte_por_lo_no_alfanumerico() -> None:
    assert tokenize("¡Coche-rojo, aparcado!") == ["coche", "rojo", "aparcado"]


async def test_limitacion_conocida_los_sinonimos_no_casan(
    embeddings: FakeEmbeddingProvider,
) -> None:
    """Parecido lexico, no semantico. Documentado, no deseado.

    Este test existe para que la limitacion este escrita en el repo y no se
    confunda este fake con una recuperacion de verdad.
    """
    coche = await embeddings.embed_query("coche")
    automovil = await embeddings.embed_query("automovil")

    assert math.isclose(coseno(coche, automovil), 0.0, abs_tol=1e-9)
