"""Configuracion central del proyecto.

Este modulo es el UNICO sitio que lee variables de entorno. Todo lo demas
recibe un `Settings` (o una de sus secciones) por inyeccion. Esto es lo que
permite que un test construya su propia configuracion sin tocar el entorno
del proceso ni monkeypatchear nada.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import cache, lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


@cache
def _expected_env_names(model: type[BaseModel], prefix: str) -> frozenset[str]:
    """Todos los nombres de variable de entorno que esta configuracion acepta.

    Recorre los campos y baja a las secciones anidadas generando los nombres con
    doble guion bajo. Se acepta tambien el nombre de la seccion a secas, porque
    pydantic-settings permite darle un JSON completo (IMAGENT_BUDGET='{...}').
    """
    names: set[str] = set()
    for field_name, field in model.model_fields.items():
        env_name = f"{prefix}{field_name.upper()}"
        names.add(env_name)
        annotation = field.annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            names |= _expected_env_names(annotation, f"{env_name}__")
    return frozenset(names)


class ProviderMode(StrEnum):
    """Familia de proveedores de modelo que construye el registry.

    FAKE es el valor por defecto a proposito: clonar el repo y ejecutar
    `pytest` tiene que funcionar sin ninguna clave. Gastar dinero es algo
    a lo que te apuntas explicitamente, nunca lo que pasa por omision.
    """

    FAKE = "fake"
    GEMINI = "gemini"


class RepositoryMode(StrEnum):
    """Donde viven los ImageRecord (decision P8-B).

    Eje independiente de los otros dos por la misma razon: querras el
    repositorio en memoria mientras desarrollas contra Qdrant de verdad, y al
    reves. Un unico flag "offline" no expresa esas combinaciones.
    """

    MEMORY = "memory"
    SQLITE = "sqlite"


class VectorStoreMode(StrEnum):
    """Donde viven los vectores.

    Va separado de ProviderMode porque son ejes independientes: querras
    poder usar Gemini de verdad contra un almacen en memoria mientras
    desarrollas, sin levantar Qdrant.
    """

    MEMORY = "memory"
    QDRANT = "qdrant"


class BudgetSettings(BaseModel):
    """Limites duros de una consulta. Se reinician en cada turno.

    Son el freno del sistema: el coordinador propone, esto dispone.
    """

    # Numero maximo de super-steps del grafo antes de cortar y responder
    # con lo que haya, marcado como incompleto.
    max_graph_iterations: int = Field(default=6, ge=1)

    # Cuantas imagenes puede volver a mirar el agente de vision en un turno.
    # ge=0 es deliberado: 0 desactiva la escalada por completo, que es
    # justo la configuracion del test de "presupuesto agotado".
    max_vision_images: int = Field(default=3, ge=0)

    # Techo de tiempo agregado para TODA la vision bajo demanda de un turno.
    # Hace falta ademas del contador de imagenes: 3 imagenes x 25 s son 75 s,
    # y eso ya es demasiado para alguien mirando la pantalla.
    max_vision_seconds: float = Field(default=45.0, gt=0)


class TimeoutSettings(BaseModel):
    """Timeout por agente. Distintos porque es distinto quien espera."""

    coordinator_seconds: float = Field(default=15.0, gt=0)
    # El usuario esta mirando la pantalla: la recuperacion tiene que ser barata.
    retrieval_seconds: float = Field(default=5.0, gt=0)
    vision_on_demand_seconds: float = Field(default=25.0, gt=0)
    # Nadie espera esto en una peticion interactiva larga; puede permitirse mas.
    vision_ingestion_seconds: float = Field(default=60.0, gt=0)
    responder_seconds: float = Field(default=20.0, gt=0)


class Settings(BaseSettings):
    """Configuracion de la aplicacion.

    Variables de entorno con prefijo `IMAGENT_`. Las secciones anidadas usan
    doble guion bajo:

        IMAGENT_PROVIDER_MODE=gemini
        IMAGENT_BUDGET__MAX_VISION_IMAGES=1
    """

    model_config = SettingsConfigDict(
        env_prefix="IMAGENT_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        # OJO con el alcance real de esto: pydantic-settings solo valida
        # "extras" en las fuentes que puede enumerar. El fichero .env se lee
        # entero, asi que una clave mal escrita ahi revienta al arrancar. Las
        # variables de entorno del proceso NO se enumeran (se busca cada campo
        # declarado por su nombre), asi que un IMAGENT_PROVIDR_MODE en el
        # entorno se ignora en silencio. Ver test_config.py.
        extra="forbid",
    )

    provider_mode: ProviderMode = ProviderMode.FAKE
    vector_store_mode: VectorStoreMode = VectorStoreMode.MEMORY
    repository_mode: RepositoryMode = RepositoryMode.MEMORY

    # --- Gemini (solo se usa si provider_mode=gemini) ---
    # SecretStr para que la clave no aparezca en un repr, un log ni un traceback.
    google_api_key: SecretStr | None = None
    vision_model: str = "gemini-2.5-flash"
    text_model: str = "gemini-2.5-flash"
    embedding_model: str = "gemini-embedding-001"
    # El fake genera vectores de esta misma dimension, para que el almacen en
    # memoria y Qdrant sean intercambiables sin tocar nada mas.
    embedding_dimensions: int = Field(default=768, ge=1)

    # --- Qdrant (solo se usa si vector_store_mode=qdrant) ---
    # Cuantas imagenes devuelve como mucho el agente de recuperacion. No es un
    # presupuesto (no acota coste, la busqueda ya se ha hecho): es cuanta
    # evidencia se le pone delante al coordinador para que juzgue.
    retrieval_top_k: int = Field(default=5, ge=1)

    # Cuantas imagenes de la coleccion se le enumeran al coordinador. Es lo que
    # le permite señalar una imagen que la recuperacion no ha devuelto, y lo que
    # hace que "la tercera" signifique algo.
    coordinator_catalogue_limit: int = Field(default=50, ge=1)

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "imagent_images"

    # --- Almacenamiento local ---
    storage_dir: Path = Path("./data/images")
    sqlite_path: Path = Path("./data/imagent.sqlite")
    max_upload_bytes: int = Field(default=10 * 1024 * 1024, gt=0)

    log_level: str = "INFO"

    # default_factory y no BudgetSettings(): explicito y sin instancias
    # compartidas entre objetos Settings distintos.
    budget: BudgetSettings = Field(default_factory=BudgetSettings)
    timeouts: TimeoutSettings = Field(default_factory=TimeoutSettings)

    @model_validator(mode="after")
    def _require_credentials_for_real_providers(self) -> Settings:
        """Falta de configuracion = fallo fatal, y al arrancar.

        Primera aplicacion concreta del contrato de fallo: esto no degrada.
        Arrancar sin clave en modo gemini solo consigue que el error salte
        en la primera peticion de un usuario en vez de en el despliegue.
        """
        if self.provider_mode is ProviderMode.GEMINI and self.google_api_key is None:
            raise ValueError(
                "provider_mode=gemini requiere IMAGENT_GOOGLE_API_KEY. "
                "Usa provider_mode=fake para ejecutar sin claves."
            )
        return self

    @model_validator(mode="after")
    def _reject_unknown_env_vars(self) -> Settings:
        """Cierra el fallo silencioso que `extra='forbid'` no cubre (decision P4).

        pydantic-settings no recorre os.environ: busca cada campo declarado por
        su nombre, asi que un IMAGENT_PROVIDR_MODE mal escrito seria invisible y
        el proceso arrancaria con la configuracion por defecto sin decir nada.
        Un typo asi puede costarte media tarde. Aqui se enumera el entorno y se
        rechaza lo que no corresponda a ningun campo.
        """
        prefix: str = type(self).model_config.get("env_prefix") or ""
        if not prefix:
            return self

        expected = _expected_env_names(type(self), prefix)
        unknown = sorted(
            name
            for name in os.environ
            if name.upper().startswith(prefix) and name.upper() not in expected
        )
        if unknown:
            raise ValueError(
                f"variables de entorno {prefix}* desconocidas: {', '.join(unknown)}. "
                "Suele ser un typo; comprueba .env.example."
            )
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Punto de acceso a la configuracion, cacheado.

    Es una funcion y no un `settings = Settings()` a nivel de modulo para que
    los tests puedan invalidarla con `get_settings.cache_clear()`. Un singleton
    de modulo se construye en el import y ya no hay forma de aislarlo.
    """
    return Settings()
