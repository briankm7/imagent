"""Que se indexa de una imagen y como.

Politica de indexado, separada del pipeline de ingesta que la usa. Aqui vive la
decision D2-B (un punto por aspecto) y la D11-A (texto crudo, sin prefijos).

Esta en `services/` y no en `domain/` a proposito: el dominio describe QUE es un
analisis, y esto decide QUE HACER con el para poder buscarlo. Si mañana quieres
recortar descripciones largas o partirlas en trozos, el cambio es de aqui y el
dominio no se entera.
"""

from __future__ import annotations

from uuid import UUID

from imagent.domain.models import Aspect, ImageAnalysis
from imagent.providers.base import EmbeddingProvider, IndexedPoint


def aspect_texts(analysis: ImageAnalysis) -> list[tuple[Aspect, str]]:
    """Los pares (aspecto, texto) indexables de un analisis.

    Decision D11-A: texto crudo, sin prefijos del tipo "Texto visible en la
    imagen:". El prefijo ayudaria a casar preguntas con palabras como "pone",
    pero diluiria el contenido corto, que es exactamente el problema que la
    separacion por aspectos venia a resolver: un OCR de cuatro palabras no puede
    competir con veinte de relleno.

    Un aspecto vacio NO genera punto. Ademas de ahorrar un vector inutil, evita
    un bug fino: el texto vacio produce el vector unitario de reserva del
    proveedor de embeddings, que casaria con cualquier consulta de forma
    aparentemente aleatoria.
    """
    pares: list[tuple[Aspect, str]] = []

    if descripcion := analysis.description.strip():
        pares.append((Aspect.DESCRIPTION, descripcion))

    if ocr := analysis.ocr_text.strip():
        pares.append((Aspect.OCR, ocr))

    if objetos := ", ".join(o.strip() for o in analysis.objects if o.strip()):
        pares.append((Aspect.OBJECTS, objetos))

    return pares


async def build_points(
    *, image_id: UUID, analysis: ImageAnalysis, embeddings: EmbeddingProvider
) -> list[IndexedPoint]:
    """Los puntos listos para escribir en el almacen.

    Los aspectos se embeben en UNA sola llamada a `embed_documents` y no en una
    por aspecto. Con un proveedor real eso son tres viajes de red menos por
    imagen, y el proveedor puede batchear por dentro.
    """
    pares = aspect_texts(analysis)
    if not pares:
        return []

    vectores = await embeddings.embed_documents([texto for _, texto in pares])

    return [
        IndexedPoint.for_aspect(image_id=image_id, aspect=aspecto, text=texto, vector=vector)
        for (aspecto, texto), vector in zip(pares, vectores, strict=True)
    ]
