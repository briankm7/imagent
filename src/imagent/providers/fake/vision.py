"""VisionProvider determinista y offline (decision P5-A).

El fake no ve nada: cada imagen lleva un GUION, indexado por el hash de su
contenido, que dice que se ve en ella. El guion separa dos cosas a proposito:

- `description`, `ocr_text` y `objects` son lo que la INGESTA extrae y se indexa;
- `on_demand` es lo que solo se descubre al VOLVER A MIRAR con una pregunta.

Esa separacion es la que hace posible el test central del proyecto. Para el caso
de "¿en alguna hay algo escrito a mano?" montas una imagen cuya descripcion
indexada no menciona la caligrafia y cuyo `on_demand` si:

    VisionScript(
        description="Una libreta abierta sobre una mesa de madera",
        objects=("libreta", "mesa"),
        on_demand={"manuscrito": "Si, hay una nota manuscrita en el margen derecho"},
    )

Si el fake devolviera lo mismo en los dos modos, ese test no probaria nada: la
recuperacion ya habria encontrado la respuesta y la escalada nunca ocurriria.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from imagent.domain.errors import VisionError
from imagent.domain.models import ImageAnalysis
from imagent.domain.text import normalize
from imagent.providers.base import ImageBlob


@dataclass(frozen=True)
class VisionScript:
    """Lo que el fake 've' en una imagen concreta."""

    description: str
    ocr_text: str = ""
    objects: tuple[str, ...] = ()

    on_demand: Mapping[str, str] = field(default_factory=dict)
    """Fragmento de pregunta -> respuesta. Se recorre en orden de insercion y
    gana la primera clave contenida en la pregunta normalizada."""

    on_demand_default: str = "No puedo determinarlo mirando esta imagen."
    """Respuesta cuando ninguna clave casa. Que exista importa: el sistema tiene
    que saber responder "no lo se" despues de haber pagado la mirada."""

    describe_error: Exception | None = None
    on_demand_error: Exception | None = None
    """Inyeccion de fallo por imagen. Va en el guion y no en el proveedor para
    poder montar el caso realista: de tres imagenes re-miradas, falla la segunda
    y el sistema responde con las otras dos, marcado como degradado."""


class FakeVisionProvider:
    """Implementa VisionProvider a partir de guiones."""

    def __init__(
        self,
        scripts: Mapping[str, VisionScript] | None = None,
        *,
        strict: bool = True,
        model_name: str = "fake-vision-1",
    ) -> None:
        self._scripts: dict[str, VisionScript] = dict(scripts or {})
        self._strict = strict
        self._model_name = model_name

        # Contadores: el test de presupuesto necesita afirmar que se miraron
        # exactamente N imagenes, no "unas cuantas".
        self.describe_calls: list[str] = []
        self.on_demand_calls: list[tuple[str, str]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    def register(self, image: ImageBlob | str, script: VisionScript) -> None:
        content_hash = image if isinstance(image, str) else image.content_hash
        self._scripts[content_hash] = script

    def _script(self, image: ImageBlob) -> VisionScript:
        script = self._scripts.get(image.content_hash)
        if script is not None:
            return script

        if self._strict:
            conocidos = ", ".join(sorted(h[:8] for h in self._scripts)) or "ninguno"
            raise VisionError(
                f"sin guion para la imagen {image.content_hash[:8]} "
                f"(registrados: {conocidos}). Registralo con FakeVisionProvider.register()."
            )

        # Modo no estricto: para arrancar la app en modo fake y que subir una
        # imagen cualquiera no reviente. El texto dice claramente que es relleno,
        # porque una descripcion inventada que PAREZCA real seria peor.
        return VisionScript(
            description=f"Imagen de demostracion sin guion registrado ({image.content_hash[:8]})",
        )

    async def describe(self, image: ImageBlob) -> ImageAnalysis:
        self.describe_calls.append(image.content_hash)
        script = self._script(image)
        if script.describe_error is not None:
            raise script.describe_error

        return ImageAnalysis(
            description=script.description,
            ocr_text=script.ocr_text,
            objects=list(script.objects),
            model_name=self._model_name,
        )

    async def answer_about(self, image: ImageBlob, question: str) -> str:
        self.on_demand_calls.append((image.content_hash, question))
        script = self._script(image)
        if script.on_demand_error is not None:
            raise script.on_demand_error

        pregunta = normalize(question)
        for clave, respuesta in script.on_demand.items():
            if normalize(clave) in pregunta:
                return respuesta
        return script.on_demand_default
