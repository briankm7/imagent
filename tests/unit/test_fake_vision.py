"""Tests del fake de vision.

El test que importa es el ultimo: es el que hace posible probar la escalada.
"""

from __future__ import annotations

import pytest

from imagent.domain.errors import VisionError
from imagent.providers.base import ImageBlob
from imagent.providers.fake.vision import FakeVisionProvider, VisionScript

LIBRETA = ImageBlob(data=b"bytes de la libreta", media_type="image/png")
COCHE = ImageBlob(data=b"bytes del coche", media_type="image/jpeg")

GUION_LIBRETA = VisionScript(
    description="Una libreta abierta sobre una mesa de madera",
    objects=("libreta", "mesa"),
    on_demand={"manuscrito": "Si, hay una nota manuscrita en el margen derecho"},
)


@pytest.fixture
def vision() -> FakeVisionProvider:
    return FakeVisionProvider({LIBRETA.content_hash: GUION_LIBRETA})


async def test_la_ingesta_devuelve_lo_que_dice_el_guion(vision: FakeVisionProvider) -> None:
    analisis = await vision.describe(LIBRETA)

    assert analisis.description == "Una libreta abierta sobre una mesa de madera"
    assert analisis.objects == ["libreta", "mesa"]
    assert analisis.model_name == "fake-vision-1"


async def test_es_determinista(vision: FakeVisionProvider) -> None:
    assert await vision.describe(LIBRETA) == await vision.describe(LIBRETA)


async def test_sin_guion_falla_ruidosamente(vision: FakeVisionProvider) -> None:
    """En los tests, una imagen sin guion es un bug del test. Silenciarlo seria peor."""
    with pytest.raises(VisionError, match="sin guion"):
        await vision.describe(COCHE)


async def test_en_modo_no_estricto_devuelve_relleno_reconocible() -> None:
    """Modo de la app real: subir una imagen cualquiera no puede reventar.

    El texto dice que es relleno; una descripcion inventada que PARECIERA real
    seria mucho peor que una que se delata.
    """
    permisivo = FakeVisionProvider(strict=False)

    analisis = await permisivo.describe(COCHE)

    assert "sin guion registrado" in analisis.description


async def test_register_acepta_el_blob_o_el_hash() -> None:
    vision = FakeVisionProvider()
    vision.register(LIBRETA, GUION_LIBRETA)
    vision.register(COCHE.content_hash, VisionScript(description="Un coche rojo"))

    assert (await vision.describe(COCHE)).description == "Un coche rojo"


async def test_la_pregunta_bajo_demanda_casa_por_fragmento(vision: FakeVisionProvider) -> None:
    respuesta = await vision.answer_about(LIBRETA, "¿Hay algo manuscrito en esta imagen?")

    assert respuesta == "Si, hay una nota manuscrita en el margen derecho"


async def test_los_acentos_y_mayusculas_no_importan(vision: FakeVisionProvider) -> None:
    assert await vision.answer_about(LIBRETA, "MANUSCRITO?") == await vision.answer_about(
        LIBRETA, "manuscrito"
    )


async def test_sin_coincidencia_responde_que_no_lo_sabe(vision: FakeVisionProvider) -> None:
    """Que exista esta respuesta importa: el sistema tiene que saber decir "no lo se"
    DESPUES de haber pagado la mirada."""
    respuesta = await vision.answer_about(LIBRETA, "¿de que color es el coche?")

    assert respuesta == "No puedo determinarlo mirando esta imagen."


async def test_cuenta_las_llamadas(vision: FakeVisionProvider) -> None:
    """El test de presupuesto necesita afirmar que se miraron N imagenes exactas."""
    await vision.describe(LIBRETA)
    await vision.answer_about(LIBRETA, "¿manuscrito?")
    await vision.answer_about(LIBRETA, "¿que pone?")

    assert vision.describe_calls == [LIBRETA.content_hash]
    assert len(vision.on_demand_calls) == 2


async def test_el_fallo_se_inyecta_por_imagen() -> None:
    """Permite el caso realista: de tres re-miradas, falla la segunda."""
    vision = FakeVisionProvider(
        {
            LIBRETA.content_hash: GUION_LIBRETA,
            COCHE.content_hash: VisionScript(
                description="Un coche", on_demand_error=VisionError("503 del proveedor")
            ),
        }
    )

    assert await vision.answer_about(LIBRETA, "manuscrito")

    with pytest.raises(VisionError, match="503"):
        await vision.answer_about(COCHE, "manuscrito")


async def test_lo_indexado_y_lo_que_se_ve_al_volver_a_mirar_son_distintos(
    vision: FakeVisionProvider,
) -> None:
    """EL test que hace posible probar la escalada.

    Lo que la ingesta indexa no menciona la caligrafia; solo aparece al volver a
    mirar con la pregunta concreta. Si el fake devolviera lo mismo en los dos
    modos, la recuperacion ya habria encontrado la respuesta y la escalada a
    vision nunca llegaria a ocurrir: el test central del proyecto no probaria nada.
    """
    indexado = await vision.describe(LIBRETA)
    al_volver_a_mirar = await vision.answer_about(LIBRETA, "¿hay algo manuscrito?")

    assert "manuscrit" not in indexado.description.lower()
    assert "manuscrit" not in indexado.ocr_text.lower()
    assert "manuscrita" in al_volver_a_mirar.lower()
