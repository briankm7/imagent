"""Salud del proceso."""

from __future__ import annotations

from fastapi import APIRouter

from imagent.api.deps import ProvidersDep, SettingsDep
from imagent.api.schemas import HealthOut
from imagent.domain.models import ImageStatus

router = APIRouter(tags=["health"])


@router.get("/health")
async def health(settings: SettingsDep, providers: ProvidersDep) -> HealthOut:
    """Estado y modo de arranque.

    Devuelve en que modo esta corriendo porque en un despliegue real la pregunta
    "¿esto esta hablando con Gemini o con los fakes?" tiene que poder
    responderse sin entrar en la maquina.
    """
    indexadas = await providers.repository.list(statuses=[ImageStatus.INDEXED])
    return HealthOut(
        status="ok",
        provider_mode=settings.provider_mode.value,
        vector_store_mode=settings.vector_store_mode.value,
        images_indexed=len(indexadas),
    )
