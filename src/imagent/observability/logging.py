"""Logging estructurado y correlacion de peticiones.

El proyecto tiene una regla: **el camino degradado tiene que ser audible**. Un
`logger.warning("algo fallo")` suelto no la cumple, porque en cuanto hay trafico
no se puede juntar lo que le paso a UNA peticion.

Aqui hay dos piezas para eso:

- un identificador de peticion en un ContextVar, que se propaga solo por todo el
  async de esa peticion sin tener que pasarlo de funcion en funcion;
- un formateador que saca los campos estructurados al final de la linea.
"""

from __future__ import annotations

import logging
from contextvars import ContextVar
from typing import Any

CLAVE = "imagent"
"""Bajo que atributo del LogRecord viajan los campos estructurados.

Van todos dentro de UN diccionario y no sueltos por una razon concreta:
`logging` reserva un monton de nombres en el LogRecord (`filename`, `module`,
`message`, `args`, `name`...) y pasar uno de ellos por `extra` no se ignora,
lanza un KeyError **en la llamada al logger**. Es decir: intentar loguear algo
rompe la peticion que intentabas observar.

Metiendolo todo bajo una sola clave, la colision es imposible por construccion
en vez de por acordarse.
"""

request_id: ContextVar[str] = ContextVar("request_id", default="-")
"""Identificador de la peticion en curso.

ContextVar y no un parametro: los nodos del grafo estan a cinco llamadas de la
capa HTTP y hacer que todos lo acepten solo para loguearlo contaminaria todas
las firmas del proyecto.
"""


class _FormateadorConExtras(logging.Formatter):
    """Formato legible que ademas saca los campos estructurados."""

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        campos: dict[str, Any] = dict(getattr(record, CLAVE, {}) or {})
        campos.setdefault("request_id", getattr(record, "request_id", "-"))
        return base + " | " + " ".join(f"{k}={v}" for k, v in sorted(campos.items()))


class _FiltroDeCorrelacion(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id.get()
        return True


def configure_logging(level: str = "INFO") -> None:
    """Deja el logging de la aplicacion listo. Idempotente."""
    manejador = logging.StreamHandler()
    manejador.setFormatter(
        _FormateadorConExtras("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
    )
    manejador.addFilter(_FiltroDeCorrelacion())

    raiz = logging.getLogger()
    # Se reemplazan los manejadores en vez de añadir: llamar dos veces a esto
    # (por ejemplo en un test) no puede acabar duplicando cada linea.
    raiz.handlers = [manejador]
    raiz.setLevel(level.upper())


def log_extra(**campos: Any) -> dict[str, Any]:
    """Campos estructurados para `logger.info(..., extra=log_extra(...))`.

    Cualquier nombre vale, incluidos los que `logging` tiene reservados: van
    dentro de `CLAVE` y no tocan el LogRecord directamente.
    """
    return {CLAVE: campos}
