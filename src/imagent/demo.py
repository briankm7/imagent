"""Escenario de demostracion.

Tres imagenes con sus guiones de vision. Vive en el paquete y no en los tests
porque lo usan los dos:

- los tests de los agentes, como escenario compartido;
- el modo fake de la aplicacion, para que quien clone el repo pueda arrancar sin
  claves y hacerle preguntas de verdad a algo. Sin esto, el modo fake acepta
  cualquier imagen pero no sabe nada de ninguna, y la demo no demuestra nada.

El guion de la libreta es el importante y esta construido a proposito: **la
descripcion que se indexa no menciona la caligrafia, y la respuesta bajo demanda
si**. Esa asimetria es la que hace que "¿en alguna hay algo escrito a mano?"
tenga que escalar al agente de vision en vez de resolverse con una busqueda.
"""

from __future__ import annotations

from imagent.providers.base import ImageBlob
from imagent.providers.fake.vision import VisionScript

COCHE = ImageBlob(data=b"bytes-de-la-foto-del-coche", media_type="image/jpeg")
LIBRETA = ImageBlob(data=b"bytes-de-la-foto-de-la-libreta", media_type="image/png")
PLAYA = ImageBlob(data=b"bytes-de-la-foto-de-la-playa", media_type="image/jpeg")

DEMO_BLOBS = {"coche": COCHE, "libreta": LIBRETA, "playa": PLAYA}
DEMO_FILENAMES = {"coche": "coche.jpg", "libreta": "libreta.png", "playa": "playa.jpg"}

DEMO_SCRIPTS = {
    COCHE.content_hash: VisionScript(
        description="Un coche rojo aparcado en la acera de una calle estrecha",
        ocr_text="SE VENDE",
        objects=("coche", "cartel", "calle"),
    ),
    LIBRETA.content_hash: VisionScript(
        # OJO: la descripcion NO menciona la caligrafia. Solo aparece al volver
        # a mirar. Si alguien "arregla" esto añadiendola aqui, el test de
        # escalada seguiria pasando y dejaria de probar nada.
        description="Una libreta abierta sobre una mesa de madera",
        objects=("libreta", "mesa"),
        on_demand={
            # Varias formas de preguntar lo mismo. El fake casa por subcadena,
            # asi que aqui van las que un usuario usaria de verdad.
            "escrito a mano": "Si, hay una nota manuscrita en el margen derecho",
            "manuscrit": "Si, hay una nota manuscrita en el margen derecho",
            "caligrafi": "Si, hay una nota manuscrita en el margen derecho",
        },
    ),
    PLAYA.content_hash: VisionScript(
        description="Una playa al atardecer con dos personas paseando",
        objects=("playa", "personas", "mar"),
        on_demand={
            "iluminad": "Bien iluminada: luz calida y suave de atardecer, sin zonas quemadas",
            "luz": "Bien iluminada: luz calida y suave de atardecer, sin zonas quemadas",
        },
    ),
}
