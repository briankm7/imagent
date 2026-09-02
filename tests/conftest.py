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
from pathlib import Path

import pytest

from imagent.config import Settings, get_settings
from imagent.providers.registry import Providers
from tests.helpers import Escena, ingerir_escena, montar_providers


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Aisla cada test del entorno de la maquina que lo ejecuta.

    Dos cosas, y las dos hacen falta:

    1. se borran las variables IMAGENT_*, o la suite pasa en CI y falla en tu
       portatil segun lo que tengas exportado;
    2. se cambia el directorio de trabajo a uno temporal, porque `Settings` lee
       un fichero `.env` del directorio actual. Sin esto, el dia que alguien
       crea un `.env` para hablar con Gemini, la mitad de los tests empiezan a
       ver `provider_mode=gemini` y fallan por un motivo que no tiene nada que
       ver con lo que estaban probando. Paso exactamente eso.

    De regalo, las rutas relativas por defecto (`./data/images`) escriben en el
    temporal en vez de ensuciar el repositorio.
    """
    for name in list(os.environ):
        if name.startswith("IMAGENT_"):
            monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)

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
