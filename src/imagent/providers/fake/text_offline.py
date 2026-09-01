"""TextProvider offline con reglas, para el modo demostracion.

Hay dos fakes de texto y hacen dos trabajos distintos:

- `FakeTextProvider` es un **doble de test**: consume un guion escrito a mano y
  falla ruidosamente si se agota. Sirve para describir el recorrido exacto que
  un test espera del grafo;
- este es un **sustituto offline**: no tiene guion, decide con reglas fijas, y
  existe para que quien clone el repo y arranque sin API keys tenga un sistema
  que de verdad hace algo. Con el fake de cola, la demostracion degradaria en el
  primer turno y no demostraria nada.

Detalle importante: este proveedor **lee el prompt renderizado**, igual que
leeria un modelo de verdad. La interfaz `TextProvider` solo pasa dos cadenas, y
darle acceso al estado del grafo para que decidiera mejor romperia justo la
frontera que hace que los proveedores sean intercambiables. Asi que analiza el
mismo texto que veria Gemini. Es tosco a proposito.

Todo lo que produce va marcado con un prefijo: una respuesta generada sin modelo
que pareciera generada por un modelo seria peor que una que se delata.
"""

from __future__ import annotations

import re
from uuid import UUID

from imagent.agents.routing import CoordinatorDecision, Route
from imagent.domain.errors import TextError
from imagent.domain.text import normalize
from imagent.providers.base import StructuredT

MARCA = "(modo sin modelo)"

_PREGUNTA = re.compile(r"^Pregunta del usuario:\s*(.+)$", re.MULTILINE)
_CATALOGO = re.compile(r"^\d+\.\s+id=([0-9a-f-]{36})", re.MULTILINE)
_HALLAZGO = re.compile(r"^- fichero=(\S+).*\n\s+respuesta:\s*(.+)$", re.MULTILINE)
_FICHEROS = re.compile(r"fichero=(\S+)")

SIN_EVIDENCIA = "(no se ha recuperado nada)"
SIN_HALLAZGOS = "(no se ha vuelto a mirar ninguna imagen)"

PISTAS_VISUALES = (
    "escrito a mano",
    "manuscrit",
    "caligrafi",
    "iluminad",
    "luz",
    "brillo",
    "nitid",
    "borros",
    "color",
    "se ve mejor",
)
"""Que hace pensar que la respuesta no esta en lo indexado.

Es una lista de palabras, no un modelo. Un modelo de verdad decide esto mucho
mejor; esto solo tiene que ser suficiente para que la escalada se pueda ver
funcionar sin claves.
"""


class HeuristicTextProvider:
    """Implementa TextProvider con reglas fijas."""

    @property
    def model_name(self) -> str:
        return "offline-heuristic"

    async def structured(
        self,
        *,
        system: str,  # noqa: ARG002 -- lo impone el Protocol; aqui no hay modelo
        user: str,
        schema: type[StructuredT],
    ) -> StructuredT:
        if schema is not CoordinatorDecision:
            raise TextError(
                f"el proveedor offline solo sabe planificar, no producir {schema.__name__}"
            )
        return self._planificar(user)  # type: ignore[return-value]

    def _planificar(self, user: str) -> CoordinatorDecision:
        pregunta = self._pregunta(user)
        hay_evidencia = SIN_EVIDENCIA not in user
        hay_hallazgos = SIN_HALLAZGOS not in user

        # 1. Sin nada recuperado todavia, siempre se busca primero: es lo barato.
        if not hay_evidencia and not hay_hallazgos:
            return CoordinatorDecision(
                action=Route.RETRIEVE, sufficient=False, standalone_query=pregunta
            )

        # 2. La pregunta va de algo que la ingesta no suele recoger y aun no se
        #    ha vuelto a mirar: se escala. El presupuesto recortara la lista.
        candidatas = self._catalogo(user)
        if not hay_hallazgos and candidatas and self._parece_visual(pregunta):
            return CoordinatorDecision(
                action=Route.VISION,
                sufficient=False,
                candidate_image_ids=candidatas,
                gap=pregunta,
            )

        # 3. Con lo que haya.
        return CoordinatorDecision(action=Route.RESPOND, sufficient=True)

    async def complete(
        self,
        *,
        system: str,  # noqa: ARG002 -- lo impone el Protocol
        user: str,
    ) -> str:
        """Redacta a partir de lo que hay en el prompt. No interpreta nada.

        `system` no se usa: son instrucciones para un modelo y aqui no hay
        ninguno. La firma la impone el Protocol y no puede cambiar, que es justo
        lo que garantiza que este fake y Gemini sean intercambiables.
        """
        hallazgos = _HALLAZGO.findall(user)
        if hallazgos:
            partes = [f"{fichero}: {respuesta.strip()}" for fichero, respuesta in hallazgos]
            return f"{MARCA} Al volver a mirar las imagenes -- " + " | ".join(partes)

        ficheros = list(dict.fromkeys(_FICHEROS.findall(user)))
        if ficheros:
            return f"{MARCA} Imagenes relevantes: " + ", ".join(ficheros) + "."

        return f"{MARCA} No he encontrado nada relevante."

    # -- lectura del prompt ---------------------------------------------------
    def _pregunta(self, user: str) -> str:
        encontrada = _PREGUNTA.search(user)
        return encontrada.group(1).strip() if encontrada else ""

    def _catalogo(self, user: str) -> list[UUID]:
        return [UUID(i) for i in _CATALOGO.findall(user)]

    def _parece_visual(self, pregunta: str) -> bool:
        normalizada = normalize(pregunta)
        return any(pista in normalizada for pista in PISTAS_VISUALES)
