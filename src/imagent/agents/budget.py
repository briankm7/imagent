"""El presupuesto de una consulta.

Esta es la pieza que separa un sistema multi-agente de una cadena de llamadas
disfrazada: el coordinador propone, y esto dispone. Vive en codigo y no en el
prompt porque un limite que el modelo puede entender es un limite que el modelo
puede incumplir.

Hay DOS niveles de freno en el proyecto y hacen falta los dos:

- este, el presupuesto blando: es una regla de negocio. Cuando se agota, el
  sistema responde con lo que tenga, marcado como incompleto. Es un camino
  NORMAL, con sus tests;
- el `recursion_limit` de LangGraph, que lanza una excepcion. Es el cinturon de
  seguridad. Si salta, es un bug de esta contabilidad.

Con solo el segundo, la unica forma de parar seria una excepcion, y una
excepcion no puede devolver "esto es lo que encontre, pero incompleto".
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from imagent.config import BudgetSettings


class Budget(BaseModel):
    """Lo que queda por gastar EN ESTE TURNO.

    Inmutable, como ImageRecord: gastar devuelve un presupuesto nuevo. Asi un
    nodo que revienta a mitad no puede dejar el contador descuadrado, y el valor
    anterior sigue siendo valido.

    Los tres campos son `ge=0`. Si alguna resta intenta bajar de cero salta un
    ValidationError, y eso NO es una condicion del entorno: significa que algo
    gasto sin preguntar antes si podia. Es un bug del enrutado y tiene que doler.
    """

    model_config = ConfigDict(frozen=True)

    iterations_left: int = Field(ge=0)
    """Super-steps del grafo que quedan antes de cortar y responder."""

    vision_images_left: int = Field(ge=0)
    """Imagenes que el agente de vision puede volver a mirar."""

    vision_seconds_left: float = Field(ge=0)
    """Techo de tiempo agregado para toda la vision bajo demanda del turno.

    Hace falta ademas del contador de imagenes: tres imagenes a 25 s son 75 s, y
    quien pregunto sigue mirando la pantalla.
    """

    @classmethod
    def for_turn(cls, settings: BudgetSettings) -> Budget:
        """Presupuesto nuevo para un turno.

        Se construye en CADA turno, no una vez por conversacion. Si se arrastrase,
        la tercera pregunta de una conversacion no tendria derecho a mirar
        ninguna imagen porque las dos primeras se lo habrian gastado.
        """
        return cls(
            iterations_left=settings.max_graph_iterations,
            vision_images_left=settings.max_vision_images,
            vision_seconds_left=settings.max_vision_seconds,
        )

    # -- consultas -----------------------------------------------------------
    @property
    def can_iterate(self) -> bool:
        return self.iterations_left > 0

    @property
    def can_look(self) -> bool:
        """Si queda margen para volver a mirar aunque sea una imagen.

        Las dos condiciones, no una: con imagenes de sobra pero sin segundos, o
        al reves, la escalada no puede ocurrir.
        """
        return self.vision_images_left > 0 and self.vision_seconds_left > 0

    def allowance(self, wanted: int) -> int:
        """Cuantas imagenes se autorizan de las `wanted` que se piden.

        Este es el clamp de la decision D3-C, y el sitio exacto donde la
        propuesta del modelo deja de ser una orden. Si el coordinador senala
        veinte candidatas y el presupuesto son tres, salen tres. No se avisa al
        modelo ni se le pide que lo reconsidere: se recorta.
        """
        if wanted < 0:
            raise ValueError("no se pueden pedir imagenes negativas")
        return min(wanted, self.vision_images_left)

    def time_allowance(self, per_image_seconds: float) -> float:
        """El timeout efectivo para una mirada.

        Es el minimo entre el timeout por imagen y lo que quede de presupuesto
        de tiempo: la ultima imagen de un lote no puede gastar mas de lo que
        sobra aunque su timeout individual sea mayor.
        """
        return min(per_image_seconds, self.vision_seconds_left)

    # -- gasto ---------------------------------------------------------------
    def spend_iteration(self) -> Budget:
        """Descuenta un super-step. Lanza si no quedaba: seria un bug del router."""
        return self.model_copy_validated(iterations_left=self.iterations_left - 1)

    def spend_vision(self, *, images: int, seconds: float) -> Budget:
        """Descuenta lo gastado en volver a mirar.

        El tiempo se descuenta con lo REALMENTE transcurrido, no con el timeout:
        si tres miradas tardaron 2 s en total, el turno sigue teniendo margen
        para mas, y castigarlo con el timeout maximo seria un presupuesto que
        miente.
        """
        if images < 0 or seconds < 0:
            raise ValueError("no se puede gastar una cantidad negativa")
        return self.model_copy_validated(
            vision_images_left=self.vision_images_left - images,
            # El tiempo puede pasarse de lo previsto por decimas sin que sea un
            # bug (medir no es gratis), asi que se corta en cero en vez de
            # reventar. Las imagenes no: esas se cuentan de una en una.
            vision_seconds_left=max(0.0, self.vision_seconds_left - seconds),
        )

    def model_copy_validated(self, **changes: object) -> Budget:
        """Copia que SI pasa por los validadores.

        `model_copy(update=...)` se los salta, y aqui los validadores son el
        unico sitio donde se detecta un descuadre del presupuesto.
        """
        return Budget(**{**self.model_dump(), **changes})
