"""Lo que la aplicacion tiene montado, y como llega a las rutas.

Todo se construye UNA vez en el lifespan y se guarda en `app.state`. Las rutas
lo reciben por `Depends`. Ninguna ruta construye nada: ni proveedores, ni
servicios, ni grafo. Es la misma regla que en el resto del proyecto, aplicada a
la capa HTTP.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from imagent.agents.graph import Conversation
from imagent.config import Settings
from imagent.providers.registry import Providers
from imagent.services.ingestion import IngestionService


@dataclass(frozen=True)
class AppState:
    """Todo lo vivo de la aplicacion, en un solo objeto."""

    settings: Settings
    providers: Providers
    ingestion: IngestionService
    conversation: Conversation


def _estado(request: Request) -> AppState:
    return request.app.state.imagent  # type: ignore[no-any-return]


def get_settings_dep(estado: Annotated[AppState, Depends(_estado)]) -> Settings:
    return estado.settings


def get_providers(estado: Annotated[AppState, Depends(_estado)]) -> Providers:
    return estado.providers


def get_ingestion(estado: Annotated[AppState, Depends(_estado)]) -> IngestionService:
    return estado.ingestion


def get_conversation(estado: Annotated[AppState, Depends(_estado)]) -> Conversation:
    return estado.conversation


SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
ProvidersDep = Annotated[Providers, Depends(get_providers)]
IngestionDep = Annotated[IngestionService, Depends(get_ingestion)]
ConversationDep = Annotated[Conversation, Depends(get_conversation)]
