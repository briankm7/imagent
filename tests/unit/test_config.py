"""Tests de configuracion.

Cada test aqui defiende una decision concreta de `config.py`, no la libreria
de pydantic. Si alguno falla, es que alguien ha cambiado una decision.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from imagent.config import (
    BudgetSettings,
    ProviderMode,
    Settings,
    VectorStoreMode,
    get_settings,
)


def test_por_defecto_no_hace_falta_nada() -> None:
    """Clonar el repo y arrancar tiene que funcionar sin claves ni base de datos."""
    settings = Settings()

    assert settings.provider_mode is ProviderMode.FAKE
    assert settings.vector_store_mode is VectorStoreMode.MEMORY
    assert settings.google_api_key is None


def test_modo_gemini_sin_clave_falla_al_construir(monkeypatch: pytest.MonkeyPatch) -> None:
    """Configuracion incompleta = fallo fatal al arrancar, no en la primera peticion."""
    monkeypatch.setenv("IMAGENT_PROVIDER_MODE", "gemini")

    with pytest.raises(ValidationError, match="IMAGENT_GOOGLE_API_KEY"):
        Settings()


def test_modo_gemini_con_clave_construye(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IMAGENT_PROVIDER_MODE", "gemini")
    monkeypatch.setenv("IMAGENT_GOOGLE_API_KEY", "clave-de-prueba")

    settings = Settings()

    assert settings.provider_mode is ProviderMode.GEMINI
    assert settings.google_api_key is not None
    assert settings.google_api_key.get_secret_value() == "clave-de-prueba"


def test_la_clave_no_aparece_en_el_repr(monkeypatch: pytest.MonkeyPatch) -> None:
    """SecretStr evita filtrar la clave en un log o un traceback."""
    monkeypatch.setenv("IMAGENT_PROVIDER_MODE", "gemini")
    monkeypatch.setenv("IMAGENT_GOOGLE_API_KEY", "clave-secretisima")

    assert "clave-secretisima" not in repr(Settings())


def test_presupuesto_configurable_por_entorno_anidado(monkeypatch: pytest.MonkeyPatch) -> None:
    """El doble guion bajo llega a la seccion anidada."""
    monkeypatch.setenv("IMAGENT_BUDGET__MAX_VISION_IMAGES", "0")

    assert Settings().budget.max_vision_images == 0


def test_presupuesto_cero_es_valido() -> None:
    """0 imagenes desactiva la escalada: es la config del test de presupuesto agotado."""
    assert BudgetSettings(max_vision_images=0).max_vision_images == 0

    with pytest.raises(ValidationError):
        BudgetSettings(max_graph_iterations=0)


def test_clave_mal_escrita_en_dotenv_revienta(tmp_path: Path) -> None:
    """extra='forbid' si protege del typo en el fichero .env, que se lee entero."""
    dotenv = tmp_path / ".env"
    dotenv.write_text("IMAGENT_PROVIDR_MODE=gemini\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="extra_forbidden"):
        Settings(_env_file=dotenv)  # type: ignore[call-arg]


def test_typo_en_variable_de_entorno_revienta(monkeypatch: pytest.MonkeyPatch) -> None:
    """Decision P4: cerramos el fallo silencioso que extra='forbid' no cubre.

    pydantic-settings no enumera os.environ (busca cada campo por su nombre),
    asi que sin la guardia explicita un IMAGENT_PROVIDR_MODE arrancaria con la
    configuracion por defecto sin decir nada.
    """
    monkeypatch.setenv("IMAGENT_PROVIDR_MODE", "gemini")

    with pytest.raises(ValidationError, match="IMAGENT_PROVIDR_MODE"):
        Settings()


def test_la_guardia_acepta_los_nombres_anidados(monkeypatch: pytest.MonkeyPatch) -> None:
    """Los nombres con doble guion bajo son legitimos y no deben rechazarse."""
    monkeypatch.setenv("IMAGENT_BUDGET__MAX_GRAPH_ITERATIONS", "2")
    monkeypatch.setenv("IMAGENT_TIMEOUTS__RETRIEVAL_SECONDS", "1.5")

    settings = Settings()

    assert settings.budget.max_graph_iterations == 2
    assert settings.timeouts.retrieval_seconds == 1.5


def test_la_guardia_ignora_variables_de_otros(monkeypatch: pytest.MonkeyPatch) -> None:
    """Solo mira las que llevan nuestro prefijo; el resto del entorno no es asunto suyo."""
    monkeypatch.setenv("OTRA_APP_PROVIDR_MODE", "lo que sea")

    assert Settings().provider_mode is ProviderMode.FAKE


def test_get_settings_cachea() -> None:
    assert get_settings() is get_settings()
