"""Comprobacion de los proveedores reales.

    python -m imagent.check

Gasta TRES llamadas al proveedor y dice si tus claves, tus modelos y tu Qdrant
funcionan. Existe porque hay un hueco que ningun test cubre: los adaptadores
reales se verifican en forma (el test de conformidad comprueba sus firmas en CI)
pero nadie los ha visto hablar con la API de verdad, y no se pueden meter en la
suite sin convertir el CI en algo que necesita claves.

Tres llamadas es deliberado: lo justo para saber que cada proveedor responde y
devuelve lo que dice su contrato, y lo bastante poco para no gastar cuota. En
modo fake no llama a nada y solo confirma que el sistema arranca.

Lo que comprueba no es "responde 200", es el contrato:

- que el embedding tiene la dimension configurada y norma 1;
- que la salida estructurada del modelo de texto se valida contra el esquema;
- que el modelo de vision ve de verdad, preguntandole el color de una imagen
  cuyo color sabemos.
"""

from __future__ import annotations

import asyncio
import math
import struct
import sys
import zlib
from time import perf_counter
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from imagent.config import (
    ProviderMode,
    RepositoryMode,
    Settings,
    VectorStoreMode,
    get_settings,
)
from imagent.domain.models import Aspect
from imagent.providers.base import ImageBlob, IndexedPoint, close_all
from imagent.providers.registry import Providers, build_providers

ROJO = (220, 30, 30)


class _ColorDetectado(BaseModel):
    """Esquema minimo para comprobar la salida estructurada."""

    color: str


def png_solido(ancho: int, alto: int, rgb: tuple[int, int, int]) -> bytes:
    """Un PNG de un color plano, sin dependencias.

    Se genera en vez de incluir un fichero para que la comprobacion sea
    autocontenida y para que la respuesta esperada sea conocida: si el modelo
    dice que ve rojo, esta viendo la imagen y no inventandose una respuesta.
    """

    def trozo(tipo: bytes, datos: bytes) -> bytes:
        cuerpo = tipo + datos
        return struct.pack(">I", len(datos)) + cuerpo + struct.pack(">I", zlib.crc32(cuerpo))

    cabecera = struct.pack(">IIBBBBB", ancho, alto, 8, 2, 0, 0, 0)  # 8 bits, RGB
    filas = b"".join(b"\x00" + bytes(rgb) * ancho for _ in range(alto))

    return (
        b"\x89PNG\r\n\x1a\n"
        + trozo(b"IHDR", cabecera)
        + trozo(b"IDAT", zlib.compress(filas))
        + trozo(b"IEND", b"")
    )


def _linea(etiqueta: str, valor: Any) -> None:
    print(f"  {etiqueta:<22} {valor}")


async def _paso(nombre: str, corrutina: Any) -> tuple[bool, Any]:
    """Ejecuta una comprobacion y la reporta sin tumbar las demas.

    Se captura todo a proposito: el valor de este comando es decirte CUALES de
    los tres proveedores funcionan, no pararse en el primero que falle.
    """
    print(f"\n[{nombre}]")
    inicio = perf_counter()
    try:
        resultado = await corrutina
    except Exception as exc:
        _linea("ERROR", f"{type(exc).__name__}: {exc}")
        return False, None
    _linea("tiempo", f"{perf_counter() - inicio:.2f}s")
    return True, resultado


async def _comprobar_embeddings(providers: Providers, settings: Settings) -> bool:
    ok, vector = await _paso(
        "embeddings", providers.embeddings.embed_query("un coche rojo aparcado")
    )
    if not ok:
        return False

    norma = math.sqrt(sum(v * v for v in vector))
    _linea("dimension", f"{len(vector)} (configurada: {settings.embedding_dimensions})")
    _linea("norma", f"{norma:.6f}")

    if len(vector) != settings.embedding_dimensions:
        _linea("FALLO", "la dimension no coincide con la configurada")
        return False
    if not math.isclose(norma, 1.0, abs_tol=1e-6):
        # Si esto salta, el coseno de Qdrant dara puntuaciones que parecen
        # razonables y estan mal.
        _linea("FALLO", "el vector no esta normalizado")
        return False
    return True


async def _comprobar_texto(providers: Providers) -> bool:
    ok, plan = await _paso(
        "texto (salida estructurada)",
        providers.text.structured(
            system="Devuelve el color que se te pide, en español, en una palabra.",
            user="El color del cielo despejado a mediodia.",
            schema=_ColorDetectado,
        ),
    )
    if not ok:
        return False

    _linea("respuesta", plan.color)
    _linea("esquema", "validado")
    return True


async def _comprobar_vision(providers: Providers) -> bool:
    imagen = ImageBlob(data=png_solido(64, 64, ROJO), media_type="image/png")

    ok, respuesta = await _paso(
        "vision (mira una imagen)",
        providers.vision.answer_about(imagen, "¿De que color es esta imagen? Una palabra."),
    )
    if not ok:
        return False

    _linea("respuesta", respuesta)
    if "roj" not in respuesta.lower():
        # No es un fallo duro: el modelo puede decir "carmesi" o describirlo de
        # otra forma. Pero merece la pena que lo mires.
        _linea("AVISO", "no menciona el rojo; comprueba que esta viendo la imagen")
    return True


async def _comprobar_almacen(providers: Providers, settings: Settings) -> bool:
    """Escribe un punto de prueba, lo busca y lo borra.

    No usa una coleccion aparte: comprueba la que se usa de verdad, porque el
    fallo tipico -dimension distinta a la de la coleccion existente- solo
    aparece contra ella. El punto se borra al terminar.
    """
    image_id = uuid4()
    vector = [0.0] * settings.embedding_dimensions
    vector[0] = 1.0

    punto = IndexedPoint.for_aspect(
        image_id=image_id,
        aspect=Aspect.DESCRIPTION,
        text="punto de comprobacion de imagent",
        vector=vector,
    )

    print("\n[almacen de vectores]")
    inicio = perf_counter()
    try:
        await providers.store.ensure_ready()
        await providers.store.upsert([punto])
        encontrados = await providers.store.search(vector, limit=1, image_ids=[image_id])
        await providers.store.delete_image(image_id)
    except Exception as exc:
        _linea("ERROR", f"{type(exc).__name__}: {exc}")
        return False

    _linea("tiempo", f"{perf_counter() - inicio:.2f}s")
    _linea("ciclo", "crear, escribir, buscar y borrar")
    if not encontrados:
        _linea("FALLO", "el punto escrito no se recupero")
        return False
    _linea("puntuacion", f"{encontrados[0].score:.4f}")
    return True


async def _comprobar_repositorio(providers: Providers) -> bool:
    """Escribe un registro, lo lee y comprueba que el fichero es utilizable."""
    from imagent.domain.models import ImageRecord

    prueba = ImageRecord(
        content_hash="0" * 64,
        filename="comprobacion-imagent.png",
        media_type="image/png",
        size_bytes=1,
    )

    print("\n[repositorio]")
    inicio = perf_counter()
    try:
        await providers.repository.save(prueba)
        leido = await providers.repository.get(prueba.id)
    except Exception as exc:
        _linea("ERROR", f"{type(exc).__name__}: {exc}")
        return False

    _linea("tiempo", f"{perf_counter() - inicio:.2f}s")
    if leido is None or leido.id != prueba.id:
        _linea("FALLO", "lo guardado no se recupero")
        return False
    _linea("ciclo", "escribir y leer")
    _linea("aviso", f"queda un registro de prueba con id {prueba.id}")
    return True


async def main() -> int:
    settings = get_settings()

    print("Configuracion")
    _linea("provider_mode", settings.provider_mode.value)
    _linea("vector_store_mode", settings.vector_store_mode.value)
    _linea("repository_mode", settings.repository_mode.value)
    if settings.provider_mode is ProviderMode.GEMINI:
        _linea("modelo de vision", settings.vision_model)
        _linea("modelo de texto", settings.text_model)
        _linea("modelo de embeddings", settings.embedding_model)
        _linea("dimensiones", settings.embedding_dimensions)
    if settings.vector_store_mode is VectorStoreMode.QDRANT:
        _linea("qdrant", settings.qdrant_url)

    providers = build_providers(settings)

    # Cada proveedor se comprueba SOLO si es el real. Los modos son ejes
    # independientes, asi que "Gemini de verdad contra un almacen en memoria" o
    # "modelos fake contra Qdrant de verdad" son configuraciones legitimas y
    # este comando tiene que decir algo util en las dos.
    modelos_reales = settings.provider_mode is ProviderMode.GEMINI
    qdrant_real = settings.vector_store_mode is VectorStoreMode.QDRANT
    sqlite_real = settings.repository_mode is RepositoryMode.SQLITE

    resultados: dict[str, bool | None] = {
        "embeddings": await _comprobar_embeddings(providers, settings) if modelos_reales else None,
        "texto": await _comprobar_texto(providers) if modelos_reales else None,
        "vision": await _comprobar_vision(providers) if modelos_reales else None,
        "almacen de vectores": await _comprobar_almacen(providers, settings)
        if qdrant_real
        else None,
        "repositorio": await _comprobar_repositorio(providers) if sqlite_real else None,
    }

    print("\nResumen")
    for nombre, estado in resultados.items():
        _linea(nombre, "ok" if estado else ("FALLO" if estado is False else "omitido (modo fake)"))

    if all(e is None for e in resultados.values()):
        await close_all(providers.repository, providers.store, providers.blobs)
        print("\nNo habia nada real que comprobar.")
        print("Pon IMAGENT_PROVIDER_MODE=gemini con una clave, o VECTOR_STORE_MODE=qdrant.")
        return 0

    await close_all(providers.repository, providers.store, providers.blobs)

    fallos = [n for n, e in resultados.items() if e is False]
    if fallos:
        print(f"\nHan fallado: {', '.join(fallos)}")
        return 1

    print("\nTodo lo comprobado funciona.")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
