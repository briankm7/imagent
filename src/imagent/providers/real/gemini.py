"""Proveedores reales sobre Gemini.

Este modulo **se puede importar sin tener `google-genai` instalado**: el SDK se
importa dentro de los metodos, no arriba. No es casualidad ni pereza, es lo que
permite que el test de conformidad compruebe en CI que estas clases cumplen los
Protocol, aunque en CI el SDK no exista. Sin eso, la forma de los proveedores
reales solo se verificaria en produccion.

Los prompts de vision son constantes aqui y no ficheros en `agents/prompts/`
porque pertenecen al ADAPTADOR, no al agente: son la forma concreta de pedirle a
Gemini lo que el dominio llama "un analisis". Otro proveedor los tendria
distintos.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel, SecretStr

from imagent.domain.errors import EmbeddingError, ProviderError, TextError, VisionError
from imagent.domain.models import ImageAnalysis
from imagent.providers.base import ImageBlob, StructuredT, Vector

PROMPT_INGESTA = """\
Analiza esta imagen para poder buscarla despues por texto.

- description: una descripcion densa y concreta de lo que se ve, en español.
  Menciona escena, objetos, personas, colores y contexto. Nada de suposiciones.
- ocr_text: transcribe LITERALMENTE todo el texto visible. Cadena vacia si no hay.
- objects: lista de objetos identificables, en singular y minusculas.
"""

PROMPT_BAJO_DEMANDA = """\
Responde a esta pregunta mirando solo esta imagen.

Si la imagen no permite responder, dilo claramente en vez de suponer.
Se breve: una o dos frases.

Pregunta: {pregunta}
"""


class _AnalisisCrudo(BaseModel):
    """El esquema que se le pide a Gemini.

    Separado de `ImageAnalysis` a proposito: aquel es dominio y este es lo que
    cabe pedirle a un modelo. `model_name` no se le pregunta al modelo, lo sabe
    el adaptador.
    """

    description: str
    ocr_text: str = ""
    objects: list[str] = []


def _construir_cliente(api_key: SecretStr) -> Any:
    try:
        from google import genai
    except ImportError as exc:  # pragma: no cover - depende de como se instalo
        # Sin esto, quien arranque en modo gemini con la instalacion base ve
        # "No module named 'google'" y tiene que deducir el resto. El mensaje
        # dice exactamente que ejecutar.
        raise ProviderError(
            "falta el extra 'gemini'. Instala con: pip install '.[gemini]'"
        ) from exc

    return genai.Client(api_key=api_key.get_secret_value())


class GeminiVisionProvider:
    """Implementa VisionProvider contra un modelo multimodal de Gemini."""

    def __init__(self, *, api_key: SecretStr, model: str) -> None:
        self._model = model
        self._cliente = _construir_cliente(api_key)

    @property
    def model_name(self) -> str:
        return self._model

    async def describe(self, image: ImageBlob) -> ImageAnalysis:
        from google.genai import types

        parte = types.Part.from_bytes(data=image.data, mime_type=image.media_type)
        crudo = await self._llamar(
            contents=[parte, PROMPT_INGESTA],
            config=types.GenerateContentConfig(
                # temperature=0: la ingesta se paga una vez y su salida se
                # indexa. Que dos ejecuciones den descripciones distintas de la
                # misma imagen haria irreproducible todo lo que viene despues.
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=_AnalisisCrudo,
            ),
        )

        analisis = crudo.parsed
        if not isinstance(analisis, _AnalisisCrudo):
            raise VisionError("Gemini no devolvio un analisis con la forma pedida")

        return ImageAnalysis(
            description=analisis.description,
            ocr_text=analisis.ocr_text,
            objects=analisis.objects,
            model_name=self._model,
        )

    async def answer_about(self, image: ImageBlob, question: str) -> str:
        from google.genai import types

        parte = types.Part.from_bytes(data=image.data, mime_type=image.media_type)
        respuesta = await self._llamar(
            contents=[parte, PROMPT_BAJO_DEMANDA.format(pregunta=question)],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        return (respuesta.text or "").strip()

    async def _llamar(self, *, contents: list[Any], config: Any) -> Any:
        """Frontera con el SDK: todo lo que salga mal sale como VisionError.

        Se captura `Exception` y no una lista de excepciones del SDK porque esa
        lista es del SDK y cambia con sus versiones. Lo que este proyecto
        promete es que un fallo de vision se ve como `VisionError`, y eso no
        puede depender de que el proveedor no renombre una clase.

        `CancelledError` y `TimeoutError` NO se capturan: heredan de
        BaseException o son la señal de que quien llama corto por su cuenta, y
        convertirlas en VisionError le robaria al llamador su propio corte.
        """
        try:
            return await self._cliente.aio.models.generate_content(
                model=self._model, contents=contents, config=config
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise VisionError(f"{type(exc).__name__}: {exc}") from exc


class GeminiTextProvider:
    """Implementa TextProvider contra un modelo de texto de Gemini."""

    def __init__(self, *, api_key: SecretStr, model: str) -> None:
        self._model = model
        self._cliente = _construir_cliente(api_key)

    @property
    def model_name(self) -> str:
        return self._model

    async def complete(self, *, system: str, user: str) -> str:
        from google.genai import types

        respuesta = await self._llamar(
            user,
            types.GenerateContentConfig(system_instruction=system, temperature=0.3),
        )
        return (respuesta.text or "").strip()

    async def structured(self, *, system: str, user: str, schema: type[StructuredT]) -> StructuredT:
        from google.genai import types

        respuesta = await self._llamar(
            user,
            types.GenerateContentConfig(
                system_instruction=system,
                # temperature=0 para planificar y 0.3 para redactar: la decision
                # de que hacer tiene que ser lo mas estable posible; la prosa
                # puede permitirse variar.
                temperature=0.0,
                response_mime_type="application/json",
                response_schema=schema,
            ),
        )

        plan = respuesta.parsed
        if not isinstance(plan, schema):
            raise TextError(f"Gemini no devolvio un {schema.__name__} valido")
        return plan

    async def _llamar(self, user: str, config: Any) -> Any:
        try:
            return await self._cliente.aio.models.generate_content(
                model=self._model, contents=user, config=config
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise TextError(f"{type(exc).__name__}: {exc}") from exc


class GeminiEmbeddingProvider:
    """Implementa EmbeddingProvider contra el modelo de embeddings de Gemini."""

    def __init__(self, *, api_key: SecretStr, model: str, dimensions: int) -> None:
        self._model = model
        self._dimensions = dimensions
        self._cliente = _construir_cliente(api_key)

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return self._model

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        return await self._embeber(list(texts), "RETRIEVAL_DOCUMENT")

    async def embed_query(self, text: str) -> Vector:
        vectores = await self._embeber([text], "RETRIEVAL_QUERY")
        return vectores[0]

    async def _embeber(self, textos: list[str], task_type: str) -> list[Vector]:
        """La distincion entre documento y consulta no es cosmetica.

        Gemini entrena esos dos modos de forma asimetrica: usar el mismo para
        indexar y para buscar empeora la recuperacion de forma medible. Tener
        dos metodos en la interfaz es lo que obliga a que quien escriba este
        adaptador se entere.
        """
        from google.genai import types

        try:
            respuesta = await self._cliente.aio.models.embed_content(
                model=self._model,
                contents=textos,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self._dimensions,
                ),
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            raise EmbeddingError(f"{type(exc).__name__}: {exc}") from exc

        if not respuesta.embeddings or len(respuesta.embeddings) != len(textos):
            raise EmbeddingError("Gemini devolvio menos vectores de los pedidos")

        return [_normalizar(list(e.values or [])) for e in respuesta.embeddings]


def _normalizar(vector: Vector) -> Vector:
    """Vector unitario.

    Hace falta de verdad, no es una precaucion: cuando se pide una dimension
    menor que la nativa del modelo, Gemini TRUNCA el vector, y un vector
    truncado ya no tiene norma 1. Buscar por coseno con vectores de normas
    distintas da resultados que parecen razonables y estan mal, que es la peor
    clase de bug.
    """
    norma = math.sqrt(sum(v * v for v in vector))
    if norma == 0.0:
        raise EmbeddingError("Gemini devolvio un vector nulo")
    return [v / norma for v in vector]
