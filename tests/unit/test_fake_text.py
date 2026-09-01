"""Tests del fake de texto."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from imagent.domain.errors import TextError
from imagent.providers.fake.text import FakeTextProvider


class Plan(BaseModel):
    action: str


class Otra(BaseModel):
    valor: int


async def test_consume_las_respuestas_en_orden() -> None:
    """Con esto un test describe el recorrido del grafo de forma literal."""
    texto = FakeTextProvider(
        structured=[Plan(action="retrieve"), Plan(action="vision"), Plan(action="respond")]
    )

    acciones = [
        (await texto.structured(system="s", user="u", schema=Plan)).action for _ in range(3)
    ]

    assert acciones == ["retrieve", "vision", "respond"]


async def test_la_cola_agotada_falla_con_un_mensaje_util() -> None:
    """El mensaje ES la informacion: el grafo dio mas vueltas de las previstas."""
    texto = FakeTextProvider(structured=[Plan(action="respond")])
    await texto.structured(system="s", user="u", schema=Plan)

    with pytest.raises(TextError, match="mas llamadas de las que el guion preveia"):
        await texto.structured(system="s", user="u", schema=Plan)


async def test_las_dos_colas_son_independientes() -> None:
    texto = FakeTextProvider(completions=["la respuesta final"], structured=[Plan(action="x")])

    await texto.structured(system="s", user="u", schema=Plan)

    assert await texto.complete(system="s", user="u") == "la respuesta final"


async def test_una_excepcion_en_la_cola_se_lanza() -> None:
    """Inyeccion de fallo sin añadir otro parametro al constructor."""
    texto = FakeTextProvider(completions=[TextError("el modelo devolvio 500")])

    with pytest.raises(TextError, match="500"):
        await texto.complete(system="s", user="u")


async def test_un_esquema_equivocado_es_un_bug_del_test() -> None:
    texto = FakeTextProvider(structured=[Otra(valor=1)])

    with pytest.raises(TypeError, match="Otra"):
        await texto.structured(system="s", user="u", schema=Plan)


async def test_registra_las_llamadas() -> None:
    texto = FakeTextProvider(completions=["ok"])

    await texto.complete(system="eres un asistente", user="hola")

    assert texto.complete_calls == [("eres un asistente", "hola")]


async def test_pending_permite_exigir_que_no_sobren_respuestas() -> None:
    """Si sobran, el grafo hizo MENOS llamadas de las previstas: tambien es un aviso."""
    texto = FakeTextProvider(completions=["a", "b"])
    await texto.complete(system="s", user="u")

    assert texto.pending == (1, 0)
