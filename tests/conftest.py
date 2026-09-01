"""Fixtures compartidas.

El autouse de `_clean_env` es la pieza importante: garantiza que ningun test
herede variables IMAGENT_* del entorno de la maquina que lo ejecuta. Sin esto,
la suite pasa en CI y falla en tu portatil (o al reves) y tardas media hora
en entender por que.

Las utilidades de construccion viven en `tests/helpers.py`, no aqui, para que se
puedan importar por su nombre desde cualquier test.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from imagent.config import Settings, get_settings
from imagent.providers.registry import Providers
from tests.helpers import Escena, ingerir_escena, montar_providers


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    for name in list(os.environ):
        if name.startswith("IMAGENT_"):
            monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings() -> Settings:
    """Configuracion por defecto: todo fake, sin claves, sin red."""
    return Settings()


@pytest.fixture
def providers() -> Providers:
    return montar_providers()


@pytest.fixture
async def escena(providers: Providers) -> Escena:
    return await ingerir_escena(providers)
