"""Normalizacion de texto.

Vive en el dominio porque lo usan tanto los agentes (para comparar consultas)
como los proveedores fake (para casar guiones y para tokenizar). Tenerlo tres
veces copiado seria tres sitios donde el tratamiento de acentos puede divergir.
"""

from __future__ import annotations

import re
import unicodedata

_ESPACIOS = re.compile(r"\s+")
_PUNTUACION = re.compile(r"[^\w\s]")


def strip_accents(text: str) -> str:
    """Quita las tildes conservando las letras.

    En español no es opcional: "que" y "que" con tilde tienen que compararse
    igual, o "¿que pone en el cartel?" y "¿qué pone en el cartel?" serian dos
    consultas distintas.
    """
    descompuesto = unicodedata.normalize("NFKD", text)
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


def normalize(text: str) -> str:
    """Minusculas, sin acentos y con los espacios colapsados."""
    return _ESPACIOS.sub(" ", strip_accents(text.lower())).strip()


def fingerprint(text: str) -> str:
    """Forma canonica para comparar dos textos POR IGUALDAD.

    Va mas lejos que `normalize` y ademas quita la puntuacion, porque
    "¿que imagenes tienen coches?" y "que imagenes tienen coches" son la misma
    consulta y tratarlas como distintas dejaria un agujero en el corte de
    busquedas repetidas: bastaria una interrogacion para saltarselo.

    Se mantiene separada de `normalize` porque son dos usos distintos: normalize
    conserva el texto para buscar subcadenas dentro de el, y esto lo destruye
    hasta dejar solo lo que sirve para decir "es la misma consulta".
    """
    return _ESPACIOS.sub(" ", _PUNTUACION.sub(" ", normalize(text))).strip()
