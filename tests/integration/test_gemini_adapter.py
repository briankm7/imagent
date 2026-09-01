"""Tests del adaptador de Gemini, sin salir a la red.

Lo que no cubren: que Google acepte lo que se le manda. Para eso hace falta una
clave y esta `python -m imagent.check`.

Lo que SI cubren, que es casi todo lo demas: que el adaptador construya
exactamente la llamada que pretende y parsee la respuesta como dice su contrato.
Se usa el SDK de verdad -sus tipos, sus validaciones- con un cliente sustituido
por un doble que anota lo que recibe. Un nombre de parametro mal escrito o un
campo de configuracion inexistente se caen aqui, no en la primera llamada de
pago.

Se saltan en CI, donde el extra `gemini` no esta instalado a proposito.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("google.genai", reason="el extra 'gemini' no esta instalado")

from pydantic import BaseModel, SecretStr

from imagent.domain.errors import EmbeddingError, TextError, VisionError
from imagent.providers import real
from imagent.providers.base import ImageBlob
from imagent.providers.real import gemini as adaptador
from imagent.providers.real.gemini import (
    GeminiEmbeddingProvider,
    GeminiTextProvider,
    GeminiVisionProvider,
    _AnalisisCrudo,
)

FOTO = ImageBlob(data=b"\x89PNG\r\n\x1a\n bytes", media_type="image/png")


class Plan(BaseModel):
    accion: str


class _RespuestaFalsa:
    def __init__(self, *, text: str = "", parsed: Any = None) -> None:
        self.text = text
        self.parsed = parsed


class _EmbeddingFalso:
    def __init__(self, values: list[float]) -> None:
        self.values = values


class _ModelosFalsos:
    """Anota lo que recibe y devuelve lo que se le programe."""

    def __init__(self, respuesta: Any = None, error: Exception | None = None) -> None:
        self.respuesta = respuesta
        self.error = error
        self.llamadas: list[dict[str, Any]] = []

    async def generate_content(self, **kwargs: Any) -> Any:
        self.llamadas.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.respuesta

    async def embed_content(self, **kwargs: Any) -> Any:
        self.llamadas.append(kwargs)
        if self.error is not None:
            raise self.error
        return self.respuesta


class _ClienteFalso:
    def __init__(self, modelos: _ModelosFalsos) -> None:
        self.aio = type("Aio", (), {"models": modelos})()


@pytest.fixture
def sustituir_cliente(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    def _sustituir(modelos: _ModelosFalsos) -> None:
        monkeypatch.setattr(adaptador, "_construir_cliente", lambda api_key: _ClienteFalso(modelos))

    return _sustituir


CLAVE = SecretStr("clave-de-prueba")


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------
async def test_la_ingesta_manda_la_imagen_y_pide_json(sustituir_cliente) -> None:  # type: ignore[no-untyped-def]
    modelos = _ModelosFalsos(
        respuesta=_RespuestaFalsa(
            parsed=_AnalisisCrudo(
                description="Un coche rojo", ocr_text="SE VENDE", objects=["coche"]
            )
        )
    )
    sustituir_cliente(modelos)

    analisis = await GeminiVisionProvider(api_key=CLAVE, model="modelo-v").describe(FOTO)

    (llamada,) = modelos.llamadas
    assert llamada["model"] == "modelo-v"
    # Los bytes de la imagen viajan como Part, no como base64 a mano.
    parte = llamada["contents"][0]
    assert parte.inline_data.data == FOTO.data
    assert parte.inline_data.mime_type == "image/png"
    # Salida estructurada, no texto libre que luego haya que parsear.
    assert llamada["config"].response_mime_type == "application/json"
    assert llamada["config"].response_schema is _AnalisisCrudo

    assert analisis.description == "Un coche rojo"
    assert analisis.ocr_text == "SE VENDE"
    # El modelo no dice como se llama: lo sabe el adaptador.
    assert analisis.model_name == "modelo-v"


async def test_la_ingesta_va_a_temperatura_cero(sustituir_cliente) -> None:  # type: ignore[no-untyped-def]
    """Se paga una vez y su salida se indexa: dos ejecuciones con descripciones
    distintas harian irreproducible todo lo que viene despues."""
    modelos = _ModelosFalsos(respuesta=_RespuestaFalsa(parsed=_AnalisisCrudo(description="x")))
    sustituir_cliente(modelos)

    await GeminiVisionProvider(api_key=CLAVE, model="m").describe(FOTO)

    assert modelos.llamadas[0]["config"].temperature == 0.0


async def test_una_respuesta_sin_la_forma_pedida_es_un_fallo(sustituir_cliente) -> None:  # type: ignore[no-untyped-def]
    """`parsed` puede venir a None si el modelo no respeta el esquema. Construir
    un ImageAnalysis a partir de eso indexaria basura."""
    sustituir_cliente(_ModelosFalsos(respuesta=_RespuestaFalsa(parsed=None)))

    with pytest.raises(VisionError, match="forma pedida"):
        await GeminiVisionProvider(api_key=CLAVE, model="m").describe(FOTO)


async def test_la_mirada_bajo_demanda_devuelve_texto(sustituir_cliente) -> None:  # type: ignore[no-untyped-def]
    modelos = _ModelosFalsos(respuesta=_RespuestaFalsa(text="  Si, hay una nota manuscrita.  "))
    sustituir_cliente(modelos)

    respuesta = await GeminiVisionProvider(api_key=CLAVE, model="m").answer_about(
        FOTO, "¿hay algo manuscrito?"
    )

    assert respuesta == "Si, hay una nota manuscrita."
    assert "¿hay algo manuscrito?" in modelos.llamadas[0]["contents"][1]


async def test_un_fallo_del_sdk_sale_como_vision_error(sustituir_cliente) -> None:  # type: ignore[no-untyped-def]
    """La frontera traduce. Lo que este proyecto promete es que un fallo de
    vision se ve como VisionError, y eso no puede depender de como llame el SDK
    a sus excepciones."""
    sustituir_cliente(_ModelosFalsos(error=RuntimeError("429 quota exceeded")))

    with pytest.raises(VisionError, match="429"):
        await GeminiVisionProvider(api_key=CLAVE, model="m").describe(FOTO)


async def test_una_cancelacion_no_se_convierte_en_vision_error(sustituir_cliente) -> None:  # type: ignore[no-untyped-def]
    """Convertirla le robaria al llamador su propio corte: el timeout por agente
    dejaria de funcionar y se veria como un fallo del proveedor."""
    import asyncio

    sustituir_cliente(_ModelosFalsos(error=asyncio.CancelledError()))

    with pytest.raises(asyncio.CancelledError):
        await GeminiVisionProvider(api_key=CLAVE, model="m").describe(FOTO)


# ---------------------------------------------------------------------------
# Texto
# ---------------------------------------------------------------------------
async def test_el_plan_va_como_salida_estructurada(sustituir_cliente) -> None:  # type: ignore[no-untyped-def]
    modelos = _ModelosFalsos(respuesta=_RespuestaFalsa(parsed=Plan(accion="vision")))
    sustituir_cliente(modelos)

    plan = await GeminiTextProvider(api_key=CLAVE, model="modelo-t").structured(
        system="eres el coordinador", user="¿que hago?", schema=Plan
    )

    llamada = modelos.llamadas[0]
    # El system va como system_instruction, no concatenado al mensaje.
    assert llamada["config"].system_instruction == "eres el coordinador"
    assert llamada["config"].response_schema is Plan
    assert llamada["config"].temperature == 0.0
    assert plan.accion == "vision"


async def test_un_plan_con_otra_forma_se_rechaza(sustituir_cliente) -> None:  # type: ignore[no-untyped-def]
    sustituir_cliente(_ModelosFalsos(respuesta=_RespuestaFalsa(parsed=None)))

    with pytest.raises(TextError, match="Plan"):
        await GeminiTextProvider(api_key=CLAVE, model="m").structured(
            system="s", user="u", schema=Plan
        )


async def test_redactar_permite_mas_variacion_que_planificar(sustituir_cliente) -> None:  # type: ignore[no-untyped-def]
    """La decision de que hacer tiene que ser lo mas estable posible; la prosa
    puede permitirse variar."""
    modelos = _ModelosFalsos(respuesta=_RespuestaFalsa(text="En coche.jpg hay un coche."))
    sustituir_cliente(modelos)

    await GeminiTextProvider(api_key=CLAVE, model="m").complete(system="s", user="u")

    assert modelos.llamadas[0]["config"].temperature > 0.0


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
async def test_indexar_y_buscar_usan_task_types_distintos(sustituir_cliente) -> None:  # type: ignore[no-untyped-def]
    """La razon de que la interfaz tenga dos metodos y no un flag.

    Gemini entrena los dos modos de forma asimetrica: usar el mismo para indexar
    y para buscar empeora la recuperacion de forma medible.
    """
    modelos = _ModelosFalsos(
        respuesta=type("R", (), {"embeddings": [_EmbeddingFalso([3.0, 4.0])]})()
    )
    sustituir_cliente(modelos)
    proveedor = GeminiEmbeddingProvider(api_key=CLAVE, model="modelo-e", dimensions=2)

    await proveedor.embed_documents(["un coche"])
    await proveedor.embed_query("coche")

    assert modelos.llamadas[0]["config"].task_type == "RETRIEVAL_DOCUMENT"
    assert modelos.llamadas[1]["config"].task_type == "RETRIEVAL_QUERY"


async def test_se_pide_la_dimension_configurada(sustituir_cliente) -> None:  # type: ignore[no-untyped-def]
    modelos = _ModelosFalsos(
        respuesta=type("R", (), {"embeddings": [_EmbeddingFalso([1.0, 0.0])]})()
    )
    sustituir_cliente(modelos)

    await GeminiEmbeddingProvider(api_key=CLAVE, model="m", dimensions=2).embed_query("x")

    assert modelos.llamadas[0]["config"].output_dimensionality == 2


async def test_los_vectores_se_normalizan(sustituir_cliente) -> None:  # type: ignore[no-untyped-def]
    """No es precaucion: al pedir una dimension menor que la nativa, Gemini
    TRUNCA el vector, y un vector truncado ya no tiene norma 1. Buscar por
    coseno con normas distintas da resultados que parecen razonables y estan mal.
    """
    import math

    modelos = _ModelosFalsos(
        respuesta=type("R", (), {"embeddings": [_EmbeddingFalso([3.0, 4.0])]})()
    )
    sustituir_cliente(modelos)

    vector = await GeminiEmbeddingProvider(api_key=CLAVE, model="m", dimensions=2).embed_query("x")

    assert vector == [0.6, 0.8]
    assert math.isclose(math.sqrt(sum(v * v for v in vector)), 1.0)


async def test_si_faltan_vectores_es_un_fallo(sustituir_cliente) -> None:  # type: ignore[no-untyped-def]
    """Emparejar textos con vectores por posicion exige que haya los mismos.
    Si no, se indexaria el vector de un texto bajo el id de otro."""
    modelos = _ModelosFalsos(
        respuesta=type("R", (), {"embeddings": [_EmbeddingFalso([1.0, 0.0])]})()
    )
    sustituir_cliente(modelos)

    with pytest.raises(EmbeddingError, match="menos vectores"):
        await GeminiEmbeddingProvider(api_key=CLAVE, model="m", dimensions=2).embed_documents(
            ["uno", "dos"]
        )


async def test_un_vector_nulo_es_un_fallo(sustituir_cliente) -> None:  # type: ignore[no-untyped-def]
    modelos = _ModelosFalsos(
        respuesta=type("R", (), {"embeddings": [_EmbeddingFalso([0.0, 0.0])]})()
    )
    sustituir_cliente(modelos)

    with pytest.raises(EmbeddingError, match="nulo"):
        await GeminiEmbeddingProvider(api_key=CLAVE, model="m", dimensions=2).embed_query("x")


def test_el_modulo_no_importa_el_sdk_al_cargarse() -> None:
    """Es lo que permite que el test de conformidad verifique estas clases en CI,
    donde `google-genai` no esta instalado."""
    import inspect

    codigo = inspect.getsource(real.gemini)
    cabecera = codigo[: codigo.index("class _AnalisisCrudo")]

    assert "from google" not in cabecera
