"""Carga de plantillas de prompt.

Los prompts viven en ficheros y no en constantes dentro del codigo por dos
motivos: un prompt de cuarenta lineas dentro de un f-string es ilegible, y un
cambio de prompt en un diff se lee mucho mejor cuando es un fichero de texto.
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

_DIRECTORIO = Path(__file__).parent


@cache
def load(nombre: str) -> str:
    """El contenido de una plantilla, cacheado.

    Se cachea porque un nodo del grafo puede ejecutarse varias veces por turno y
    leer del disco en cada una seria gratuito de evitar.
    """
    return (_DIRECTORIO / f"{nombre}.md").read_text(encoding="utf-8")
