"""Tests del agente de recuperacion.

Se prueba el nodo suelto, invocandolo como una funcion sobre un estado. No hace
falta grafo: un nodo de LangGraph es un callable que recibe estado y devuelve un
diccionario de cambios, y probarlo asi es mucho mas barato que montar el grafo.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Any
from uuid import UUID, uuid4

import pytest

from imagent.agents.budget import Budget
from imagent.agents.retrieval import EVENTO_FALLO, EVENTO_HUERFANO, RetrievalAgent
from imagent.agents.routing import CoordinatorDecision, Route
from imagent.agents.state import new_turn
from imagent.config import TimeoutSettings
from imagent.domain.errors import VectorStoreError
from imagent.domain.models import Aspect, RetrievedImage
from imagent.providers.base import ScoredPoint, Vector
from imagent.providers.fake.vectorstore import InMemoryVectorStore
from imagent.providers.registry import Providers
from tests.helpers import DIMENSIONS, Escena, montar_providers


def presupuesto() -> Budget:
    return Budget(iterations_left=6, vision_images_left=3, vision_seconds_left=45.0)


def estado(consulta: str, *, pregunta: str = "la pregunta cruda", **extra: Any) -> Any:
    base = new_turn(question=pregunta, budget=presupuesto())
    base["decision"] = CoordinatorDecision(
        action=Route.RETRIEVE, sufficient=False, standalone_query=consulta
    )
    base["focus_image_ids"] = []
    return {**base, **extra}


def agente(providers: Providers, *, top_k: int = 5, **timeouts: float) -> RetrievalAgent:
    return RetrievalAgent(providers, timeouts=TimeoutSettings(**timeouts), top_k=top_k)


async def test_encuentra_lo_que_se_le_pide(escena: Escena) -> None:
    salida = await agente(escena.providers)(estado("coche aparcado en la acera"))

    assert salida["retrieved"][0].image_id == escena.id("coche")


async def test_ordena_de_mas_a_menos_relevante(escena: Escena) -> None:
    salida = await agente(escena.providers)(estado("libreta sobre una mesa de madera"))

    assert salida["retrieved"][0].image_id == escena.id("libreta")


async def test_agrupa_los_aspectos_de_una_imagen_en_un_solo_resultado(
    escena: Escena,
) -> None:
    """Decision D2-B: tres puntos por imagen entran, una imagen sale."""
    salida = await agente(escena.providers)(estado("coche cartel calle SE VENDE"))

    del_coche = [r for r in salida["retrieved"] if r.image_id == escena.id("coche")]
    assert len(del_coche) == 1
    assert len(del_coche[0].matched_aspects) > 1


async def test_conserva_que_aspecto_caso(escena: Escena) -> None:
    """No es lo mismo casar por descripcion que por OCR, y el coordinador
    necesita saberlo para juzgar si la evidencia le basta."""
    salida = await agente(escena.providers)(estado("SE VENDE"))

    del_coche = next(r for r in salida["retrieved"] if r.image_id == escena.id("coche"))
    assert Aspect.OCR in del_coche.matched_aspects


async def test_completa_el_resultado_con_el_registro(escena: Escena) -> None:
    """El punto que caso solo trae SU texto; la descripcion entera viene del
    repositorio para que el coordinador vea la imagen completa."""
    salida = await agente(escena.providers)(estado("SE VENDE"))

    del_coche = next(r for r in salida["retrieved"] if r.image_id == escena.id("coche"))
    assert del_coche.filename == "coche.jpg"
    assert "coche rojo" in del_coche.description
    assert "coche" in del_coche.objects


async def test_respeta_el_top_k(escena: Escena) -> None:
    salida = await agente(escena.providers, top_k=1)(estado("coche libreta playa"))

    assert len(salida["retrieved"]) == 1


async def test_usa_la_consulta_resuelta_y_no_la_pregunta_cruda(escena: Escena) -> None:
    """El agente de recuperacion no ve el historial: solo la standalone_query.

    Aqui la pregunta cruda habla de una playa y la consulta resuelta de un
    coche. Gana la consulta resuelta.
    """
    salida = await agente(escena.providers)(
        estado("coche rojo aparcado", pregunta="¿y en esa hay una playa?")
    )

    assert salida["retrieved"][0].image_id == escena.id("coche")


async def test_apunta_la_consulta_para_el_corte_de_repeticiones(escena: Escena) -> None:
    salida = await agente(escena.providers)(estado("coches"))

    assert salida["queries_done"] == ["coches"]


async def test_acumula_las_consultas_del_turno(escena: Escena) -> None:
    salida = await agente(escena.providers)(estado("coches", queries_done=["playas"]))

    assert salida["queries_done"] == ["playas", "coches"]


async def test_mezcla_con_la_evidencia_anterior(escena: Escena) -> None:
    """D13-B permite reformular: la segunda busqueda completa a la primera."""
    previa = RetrievedImage(image_id=uuid4(), filename="previa.jpg", score=0.99)

    salida = await agente(escena.providers)(estado("coche", retrieved=[previa]))

    assert previa in salida["retrieved"]
    assert len(salida["retrieved"]) > 1


# ---------------------------------------------------------------------------
# Degradacion
# ---------------------------------------------------------------------------
async def test_un_fallo_del_almacen_degrada_en_vez_de_tumbar_el_turno(
    escena: Escena,
) -> None:
    """Responder "no he podido consultar el indice" es mejor que no responder."""

    class StoreRoto(InMemoryVectorStore):
        async def search(
            self,
            vector: Vector,
            *,
            limit: int,
            aspects: Collection[Aspect] | None = None,
            image_ids: Collection[UUID] | None = None,
        ) -> list[ScoredPoint]:
            raise VectorStoreError("connection refused")

    providers = montar_providers(store=StoreRoto(dimensions=DIMENSIONS))

    salida = await agente(providers)(estado("coches"))

    assert salida["retrieved"] == []
    assert [d.event for d in salida["degradations"]] == [EVENTO_FALLO]
    assert "connection refused" in salida["degradations"][0].detail


async def test_una_consulta_fallida_tambien_se_apunta(escena: Escena) -> None:
    """Repetirla en el mismo turno fallaria igual y solo quemaria otra iteracion."""

    class StoreRoto(InMemoryVectorStore):
        async def search(self, *args: Any, **kwargs: Any) -> list[ScoredPoint]:
            raise VectorStoreError("caido")

    providers = montar_providers(store=StoreRoto(dimensions=DIMENSIONS))

    salida = await agente(providers)(estado("coches"))

    assert salida["queries_done"] == ["coches"]


async def test_un_timeout_degrada(escena: Escena) -> None:
    class StoreLento(InMemoryVectorStore):
        async def search(self, *args: Any, **kwargs: Any) -> list[ScoredPoint]:
            import asyncio

            await asyncio.sleep(10)
            raise AssertionError("no deberia llegar")

    providers = montar_providers(store=StoreLento(dimensions=DIMENSIONS))

    salida = await agente(providers, retrieval_seconds=0.01)(estado("coches"))

    assert salida["retrieved"] == []
    assert salida["degradations"][0].error_type == "TimeoutError"


async def test_puntos_sin_registro_se_saltan_y_dejan_constancia(escena: Escena) -> None:
    """La divergencia entre los dos almacenes, hecha visible.

    Se borra el registro pero no sus puntos: devolver un resultado que no se
    puede describir seria peor que saltarselo, y saltarselo en silencio seria
    peor que decirlo.
    """
    escena.repository._records.pop(escena.id("coche"))

    salida = await agente(escena.providers)(estado("coche rojo aparcado en la acera"))

    assert all(r.image_id != escena.id("coche") for r in salida["retrieved"])
    assert EVENTO_HUERFANO in [d.event for d in salida["degradations"]]


async def test_un_almacen_vacio_no_es_un_fallo() -> None:
    """No encontrar nada es justo la condicion que dispara la escalada."""
    providers = montar_providers()
    await providers.store.ensure_ready()

    salida = await agente(providers)(estado("lo que sea"))

    assert salida["retrieved"] == []
    assert salida["degradations"] == []


@pytest.mark.parametrize("nodo_devuelve", ["retrieved", "queries_done", "degradations"])
async def test_el_nodo_devuelve_las_listas_enteras(escena: Escena, nodo_devuelve: str) -> None:
    """Decision D12-A: sin reducers, el nodo mezcla y devuelve el estado final
    del campo, no solo lo nuevo."""
    salida = await agente(escena.providers)(estado("coche"))

    assert isinstance(salida[nodo_devuelve], Sequence)
