"""TextProvider determinista y offline.

Es una cola de respuestas programadas. Cada llamada consume la siguiente.

Por que una cola y no una funcion que "razone" sobre la entrada: la mitad de la
decision D3-C es que el enrutado sea reproducible en los tests. Con una cola,
un test describe el recorrido completo del grafo de forma literal:

    FakeTextProvider(structured=[
        Plan(action="retrieve", ...),   # primer paso del coordinador
        Plan(action="vision", ...),     # no basta: escala
        Plan(action="respond", ...),    # ya basta
    ])

Se lee como un guion y no depende de que ningun modelo se porte bien.

Si la cola se agota, se lanza TextError. Ese fallo es informacion: significa que
el grafo hizo mas llamadas de las que el test preveia, que suele ser justo el
bug que buscabas.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence

from pydantic import BaseModel

from imagent.domain.errors import TextError
from imagent.providers.base import StructuredT

type Scripted[T] = T | Exception


class FakeTextProvider:
    """Implementa TextProvider consumiendo respuestas de dos colas."""

    def __init__(
        self,
        *,
        completions: Sequence[Scripted[str]] = (),
        structured: Sequence[Scripted[BaseModel]] = (),
        model_name: str = "fake-text-1",
    ) -> None:
        self._completions: deque[Scripted[str]] = deque(completions)
        self._structured: deque[Scripted[BaseModel]] = deque(structured)
        self._model_name = model_name

        self.complete_calls: list[tuple[str, str]] = []
        self.structured_calls: list[tuple[str, str, type[BaseModel]]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def pending(self) -> tuple[int, int]:
        """Respuestas sin consumir. Un test puede exigir que no sobre ninguna."""
        return len(self._completions), len(self._structured)

    def queue_structured(self, *respuestas: Scripted[BaseModel]) -> None:
        """Añade planes al guion despues de construir el proveedor.

        Hace falta porque un test que programa la escalada necesita los ids de
        las imagenes, y esos no existen hasta que la escena esta ingerida.
        """
        self._structured.extend(respuestas)

    def queue_completions(self, *respuestas: Scripted[str]) -> None:
        self._completions.extend(respuestas)

    async def complete(self, *, system: str, user: str) -> str:
        self.complete_calls.append((system, user))
        if not self._completions:
            raise TextError(
                "se han agotado las respuestas de texto programadas: el sistema "
                "hizo mas llamadas de las que el guion preveia"
            )

        siguiente = self._completions.popleft()
        # Una excepcion en la cola es inyeccion de fallo: no hace falta otro
        # parametro en el constructor para probar la degradacion.
        if isinstance(siguiente, Exception):
            raise siguiente
        return siguiente

    async def structured(self, *, system: str, user: str, schema: type[StructuredT]) -> StructuredT:
        self.structured_calls.append((system, user, schema))
        if not self._structured:
            raise TextError(
                "se han agotado las respuestas estructuradas programadas: el "
                "sistema hizo mas llamadas de las que el guion preveia"
            )

        siguiente = self._structured.popleft()
        if isinstance(siguiente, Exception):
            raise siguiente
        if not isinstance(siguiente, schema):
            # Bug del test, no del sistema: falla ruidosamente y con nombres.
            raise TypeError(
                f"el guion devuelve {type(siguiente).__name__} pero se pidio {schema.__name__}"
            )
        return siguiente
