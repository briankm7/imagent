"""Como se le enseña el estado a un modelo.

Modulo propio y no funciones privadas dentro del coordinador porque lo usan dos
nodos, y porque es la capa donde se decide **cuanto contexto ve cada agente**.
Esa decision merece un sitio con nombre: es la mitad de lo que hace que un
sistema de agentes cueste poco o cueste mucho.
"""

from __future__ import annotations

from uuid import UUID

from langchain_core.messages import AnyMessage

from imagent.domain.errors import DegradationEvent
from imagent.domain.models import ImageRecord, RetrievedImage, VisionFinding

MENSAJES_DE_HISTORIAL = 8
"""Cuantos mensajes recientes ve el coordinador.

No es el historial entero a proposito: resolver "y de esa" necesita los ultimos
turnos, no la conversacion de hace media hora. Meterla entera encarece la
llamada y diluye lo que importa.
"""


def historial(messages: list[AnyMessage]) -> str:
    if not messages:
        return "(vacio)"
    return "\n".join(f"{m.type}: {m.content}" for m in messages[-MENSAJES_DE_HISTORIAL:])


def identificadores(ids: list[UUID]) -> str:
    return ", ".join(str(i) for i in ids) or "ninguna"


def evidencia(recuperadas: list[RetrievedImage]) -> str:
    """La evidencia barata, con el id delante.

    El id va primero porque el coordinador tiene que poder devolverlo en
    `candidate_image_ids`: si no lo ve, no puede señalar nada y la escalada a
    vision no llega a ocurrir nunca.

    Tambien van los aspectos que casaron. No es lo mismo que una imagen salga
    por su descripcion que por su OCR, y es justo lo que hay que mirar para
    juzgar si la evidencia basta o hay que volver a mirar la imagen.
    """
    if not recuperadas:
        return "(no se ha recuperado nada)"

    lineas = []
    for imagen in recuperadas:
        aspectos = ", ".join(sorted(a.value for a in imagen.matched_aspects))
        lineas.append(
            f"- id={imagen.image_id} fichero={imagen.filename} "
            f"puntuacion={imagen.score:.3f} caso_por=[{aspectos}]\n"
            f"  descripcion: {imagen.description}\n"
            f"  texto visible: {imagen.ocr_text or '(ninguno)'}\n"
            f"  objetos: {', '.join(imagen.objects) or '(ninguno)'}"
        )
    return "\n".join(lineas)


def hallazgos(encontrados: list[VisionFinding]) -> str:
    if not encontrados:
        return "(no se ha vuelto a mirar ninguna imagen)"
    return "\n".join(
        f"- fichero={h.filename or h.image_id} id={h.image_id} pregunta: {h.question}\n"
        f"  respuesta: {h.answer}"
        for h in encontrados
    )


def degradaciones(eventos: list[DegradationEvent]) -> str:
    if not eventos:
        return "(ninguna)"
    return "\n".join(f"- {e.event}: {e.detail}" for e in eventos)


def catalogo(registros: list[ImageRecord], limite: int) -> str:
    """Que imagenes existen, numeradas y con su id.

    Sin esto, el coordinador solo conoce las imagenes que la recuperacion le ha
    devuelto, y entonces la escalada a vision **no puede ocurrir nunca en el
    caso que mas importa**: si nadie indexo la caligrafia, buscar "escrito a
    mano" no devuelve nada, y sin ids que señalar no hay nada que mirar.

    La numeracion sale del orden por fecha de subida que garantiza el
    repositorio, y es lo que hace que "la tercera" signifique algo.

    LIMITACION: esto crece con la coleccion. Con miles de imagenes habria que
    cambiarlo por una preseleccion. Va al README.
    """
    if not registros:
        return "(no hay ninguna imagen indexada)"

    lineas = [
        f"{i}. id={r.id} fichero={r.filename}" for i, r in enumerate(registros[:limite], start=1)
    ]
    if len(registros) > limite:
        lineas.append(f"... y {len(registros) - limite} mas no mostradas")
    return "\n".join(lineas)
