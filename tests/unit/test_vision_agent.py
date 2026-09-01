"""Tests del agente de vision bajo demanda.

El test central del proyecto esta aqui: `test_descubre_lo_que_la_ingesta_no_indexo`.
Es el que demuestra que la escalada sirve para algo.
"""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from imagent.agents.budget import Budget
from imagent.agents.routing import CoordinatorDecision, Route
from imagent.agents.state import new_turn
from imagent.agents.vision import (
    EVENTO_FALLO,
    EVENTO_SIN_REGISTRO,
    EVENTO_SIN_TIEMPO,
    VisionAgent,
)
from imagent.config import TimeoutSettings
from imagent.demo import DEMO_SCRIPTS, LIBRETA
from imagent.domain.errors import VisionError
from imagent.providers.base import ImageBlob
from imagent.providers.fake.vision import FakeVisionProvider, VisionScript
from imagent.providers.registry import Providers
from tests.helpers import Escena, montar_providers


def presupuesto(*, imagenes: int = 3, segundos: float = 45.0) -> Budget:
    return Budget(iterations_left=6, vision_images_left=imagenes, vision_seconds_left=segundos)


def estado(
    candidatas: list[Any],
    *,
    autorizadas: int | None = None,
    pregunta: str = "¿hay algo escrito a mano?",
    gap: str = "",
    budget: Budget | None = None,
    **extra: Any,
) -> Any:
    base = new_turn(question=pregunta, budget=budget or presupuesto())
    base["decision"] = CoordinatorDecision(
        action=Route.VISION,
        sufficient=False,
        candidate_image_ids=candidatas,
        gap=gap,
    )
    base["vision_allowance"] = autorizadas if autorizadas is not None else len(candidatas)
    base["focus_image_ids"] = []
    return {**base, **extra}


def agente(providers: Providers, **timeouts: float) -> VisionAgent:
    return VisionAgent(providers, timeouts=TimeoutSettings(**timeouts))


# ---------------------------------------------------------------------------
# El caso que justifica todo el proyecto
# ---------------------------------------------------------------------------
async def test_descubre_lo_que_la_ingesta_no_indexo(escena: Escena) -> None:
    """ "¿En alguna hay algo escrito a mano?", de punta a punta.

    La descripcion indexada de la libreta NO menciona la caligrafia: la
    recuperacion no puede encontrarla por mucho que busque. Solo aparece al
    volver a mirar la imagen con la pregunta concreta delante.
    """
    indexado = (await escena.repository.get(escena.id("libreta"))).analysis
    assert "manuscrit" not in indexado.description.lower()

    salida = await agente(escena.providers)(
        estado([escena.id("libreta")], gap="¿hay algo manuscrito en la imagen?")
    )

    (hallazgo,) = salida["vision_findings"]
    assert "manuscrita" in hallazgo.answer
    assert hallazgo.image_id == escena.id("libreta")


async def test_usa_el_hueco_que_identifico_el_coordinador(escena: Escena) -> None:
    """El agente de vision no recibe la conversacion: recibe UNA pregunta."""
    salida = await agente(escena.providers)(
        estado(
            [escena.id("playa")],
            pregunta="oye, y de la tercera, ¿que tal se ve?",
            gap="¿esta bien iluminada?",
        )
    )

    (hallazgo,) = salida["vision_findings"]
    assert hallazgo.question == "¿esta bien iluminada?"
    assert "atardecer" in hallazgo.answer


async def test_sin_hueco_se_usa_la_pregunta_original(escena: Escena) -> None:
    salida = await agente(escena.providers)(
        estado([escena.id("playa")], pregunta="¿esta bien iluminada?")
    )

    assert salida["vision_findings"][0].question == "¿esta bien iluminada?"


async def test_mira_varias_imagenes_en_lote(escena: Escena) -> None:
    """Decision D5: sin parada temprana. Para "¿en cuales aparece un coche?" hay
    que verlas todas."""
    salida = await agente(escena.providers)(
        estado([escena.id("coche"), escena.id("libreta"), escena.id("playa")])
    )

    assert len(salida["vision_findings"]) == 3


async def test_una_imagen_sin_respuesta_util_tambien_deja_hallazgo(escena: Escena) -> None:
    """El sistema tiene que saber decir "no lo se" DESPUES de haber pagado."""
    salida = await agente(escena.providers)(
        estado([escena.id("coche")], gap="¿hay algo manuscrito?")
    )

    assert "No puedo determinarlo" in salida["vision_findings"][0].answer


# ---------------------------------------------------------------------------
# El presupuesto manda
# ---------------------------------------------------------------------------
async def test_mira_solo_las_autorizadas_por_el_router(escena: Escena) -> None:
    """El recorte lo hizo el router; este nodo obedece y no vuelve a decidir."""
    tres = [escena.id("coche"), escena.id("libreta"), escena.id("playa")]

    salida = await agente(escena.providers)(estado(tres, autorizadas=2))

    assert len(salida["vision_findings"]) == 2
    assert len(escena.vision.on_demand_calls) == 2


async def test_sin_autorizacion_no_se_mira_nada(escena: Escena) -> None:
    """La configuracion del test de presupuesto agotado: cero llamadas caras."""
    salida = await agente(escena.providers)(estado([escena.id("libreta")], autorizadas=0))

    assert salida["vision_findings"] == []
    assert escena.vision.on_demand_calls == []


async def test_cada_mirada_gasta_una_imagen_del_presupuesto(escena: Escena) -> None:
    salida = await agente(escena.providers)(
        estado([escena.id("coche"), escena.id("libreta")], budget=presupuesto(imagenes=3))
    )

    assert salida["budget"].vision_images_left == 1


async def test_el_tiempo_de_las_miradas_tambien_se_descuenta(escena: Escena) -> None:
    """Se comprueba con un proveedor que tarda algo medible.

    Con el fake normal el reloj puede no avanzar entre dos llamadas, y un test
    que asegurase "ha bajado" seria inestable por motivos que no tienen nada que
    ver con lo que se quiere probar.
    """

    class VisionConLatencia(FakeVisionProvider):
        async def answer_about(self, image: ImageBlob, question: str) -> str:
            await asyncio.sleep(0.05)
            return await super().answer_about(image, question)

    providers = montar_providers(
        vision=VisionConLatencia(DEMO_SCRIPTS),
        repository=escena.repository,
        blobs=escena.providers.blobs,
    )

    salida = await agente(providers)(
        estado([escena.id("coche"), escena.id("libreta")], budget=presupuesto(segundos=45.0))
    )

    assert salida["budget"].vision_seconds_left <= 45.0 - 0.1


async def test_el_tiempo_agotado_a_mitad_del_lote_para_las_restantes(
    escena: Escena,
) -> None:
    """El presupuesto de tiempo es independiente del de imagenes y puede
    agotarse antes, con imagenes todavia autorizadas."""

    class VisionLenta(FakeVisionProvider):
        async def answer_about(self, image: ImageBlob, question: str) -> str:
            await asyncio.sleep(0.3)
            return await super().answer_about(image, question)

    # Se reutilizan los registros y los bytes ya ingeridos de la escena; solo
    # se sustituye el proveedor de vision por uno lento.
    providers = montar_providers(
        vision=VisionLenta(DEMO_SCRIPTS),
        repository=escena.repository,
        blobs=escena.providers.blobs,
    )

    salida = await agente(providers)(
        estado(
            [escena.id("coche"), escena.id("libreta"), escena.id("playa")],
            budget=presupuesto(imagenes=3, segundos=0.5),
        )
    )

    eventos = [d.event for d in salida["degradations"]]
    assert EVENTO_SIN_TIEMPO in eventos
    assert len(salida["vision_findings"]) < 3
    assert salida["budget"].can_look is False


# ---------------------------------------------------------------------------
# Degradacion, imagen a imagen
# ---------------------------------------------------------------------------
async def test_el_fallo_de_una_imagen_no_tira_las_demas(escena: Escena) -> None:
    """Un lote que fuera todo o nada seria pagar tres llamadas para tirar el
    resultado por una."""
    rota = ImageBlob(data=b"bytes-de-la-rota", media_type="image/png")
    escena.vision.register(
        rota, VisionScript(description="Rota", on_demand_error=VisionError("503"))
    )
    from imagent.services.ingestion import IngestionService

    resultado = await IngestionService(escena.providers, timeouts=TimeoutSettings()).ingest(
        filename="rota.png", media_type="image/png", data=rota.data
    )

    salida = await agente(escena.providers)(
        estado([escena.id("libreta"), resultado.record.id, escena.id("playa")])
    )

    assert len(salida["vision_findings"]) == 2
    assert [d.event for d in salida["degradations"]] == [EVENTO_FALLO]


async def test_una_imagen_que_falla_igualmente_gasta_presupuesto(escena: Escena) -> None:
    """La llamada al proveedor ya se ha hecho: no cobrarla convertiria un fallo
    en presupuesto gratis."""
    rota = ImageBlob(data=b"bytes-de-la-rota", media_type="image/png")
    escena.vision.register(
        rota, VisionScript(description="Rota", on_demand_error=VisionError("503"))
    )
    from imagent.services.ingestion import IngestionService

    resultado = await IngestionService(escena.providers, timeouts=TimeoutSettings()).ingest(
        filename="rota.png", media_type="image/png", data=rota.data
    )

    salida = await agente(escena.providers)(
        estado([resultado.record.id], budget=presupuesto(imagenes=3))
    )

    assert salida["vision_findings"] == []
    assert salida["budget"].vision_images_left == 2


async def test_una_candidata_inexistente_no_gasta_presupuesto(escena: Escena) -> None:
    """No llega a costar una llamada al modelo, asi que no tiene por que cobrarse."""
    salida = await agente(escena.providers)(
        estado([uuid4(), escena.id("libreta")], budget=presupuesto(imagenes=3))
    )

    assert len(salida["vision_findings"]) == 1
    assert salida["budget"].vision_images_left == 2
    assert EVENTO_SIN_REGISTRO in [d.event for d in salida["degradations"]]


async def test_un_timeout_de_una_mirada_degrada(escena: Escena) -> None:
    class VisionLentisima(FakeVisionProvider):
        async def answer_about(self, image: ImageBlob, question: str) -> str:
            await asyncio.sleep(10)
            raise AssertionError("no deberia llegar")

    providers = montar_providers(
        vision=VisionLentisima(DEMO_SCRIPTS),
        repository=escena.repository,
        blobs=escena.providers.blobs,
    )

    salida = await agente(providers, vision_on_demand_seconds=0.01)(estado([escena.id("libreta")]))

    assert salida["vision_findings"] == []
    assert salida["degradations"][0].event == EVENTO_FALLO
    assert salida["degradations"][0].error_type == "TimeoutError"


async def test_sin_candidatas_no_hace_nada(escena: Escena) -> None:
    salida = await agente(escena.providers)(estado([]))

    assert salida["vision_findings"] == []
    assert salida["degradations"] == []
    assert escena.vision.on_demand_calls == []


async def test_acumula_sobre_los_hallazgos_previos(escena: Escena) -> None:
    """Decision D12-A: el nodo devuelve la lista entera, no solo lo nuevo."""
    previo = (await agente(escena.providers)(estado([escena.id("libreta")], gap="¿manuscrito?")))[
        "vision_findings"
    ]

    salida = await agente(escena.providers)(
        estado([escena.id("playa")], gap="¿iluminada?", vision_findings=previo)
    )

    assert len(salida["vision_findings"]) == 2


def test_libreta_es_el_caso_asimetrico() -> None:
    """Guarda del escenario: si alguien "arregla" el guion añadiendo la
    caligrafia a la descripcion, el test de escalada dejaria de probar nada y
    seguiria pasando. Esto lo impide."""
    guion = DEMO_SCRIPTS[LIBRETA.content_hash]

    assert "manuscrit" not in guion.description.lower()
    assert "manuscrit" not in guion.ocr_text.lower()
    assert any("manuscrit" in v.lower() for v in guion.on_demand.values())
