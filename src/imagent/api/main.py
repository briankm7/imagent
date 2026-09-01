"""La aplicacion.

Todo lo caro se construye una vez, en el lifespan, y se guarda en `app.state`.
La factoria acepta proveedores y checkpointer ya montados para que un test pueda
levantar la aplicacion entera sin tocar disco ni abrir sqlite.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, Request, status
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from langgraph.checkpoint.base import BaseCheckpointSaver

from imagent.agents.graph import Conversation, build_graph, open_checkpointer
from imagent.api.deps import AppState
from imagent.api.routes import chat, health, images
from imagent.api.schemas import ErrorOut
from imagent.config import ProviderMode, Settings, get_settings
from imagent.demo import DEMO_BLOBS, DEMO_FILENAMES
from imagent.domain.errors import ImagentError, InvalidImageError
from imagent.observability.logging import configure_logging, log_extra, request_id
from imagent.providers.base import close_all
from imagent.providers.registry import Providers, build_providers
from imagent.services.ingestion import IngestionService

logger = logging.getLogger(__name__)
WEB = Path(__file__).parent.parent / "web"


async def _cerrar(estado: AppState) -> None:
    """Suelta los recursos de los proveedores al apagar.

    Solo importa para los que tienen alguno: el repositorio SQLite mantiene una
    conexion en un hilo propio y, sin cerrarla, el proceso termina con un
    "Event loop is closed" desde ese hilo. Un fallo al apagar es facil de
    ignorar y siempre significa que algo no se esta soltando.
    """
    await close_all(
        estado.providers.repository,
        estado.providers.store,
        estado.providers.blobs,
    )


def _montar_estado(
    settings: Settings, providers: Providers, checkpointer: BaseCheckpointSaver
) -> AppState:
    grafo = build_graph(providers, settings=settings, checkpointer=checkpointer)
    return AppState(
        settings=settings,
        providers=providers,
        ingestion=IngestionService(providers, timeouts=settings.timeouts),
        conversation=Conversation(grafo, settings=settings),
    )


async def sembrar_demo(estado: AppState) -> None:
    """Ingiere el escenario de ejemplo si estamos en modo fake.

    Sin esto, quien clone el repo y arranque sin claves tiene una aplicacion que
    acepta imagenes pero no sabe nada de ninguna, y la demostracion no demuestra
    nada. Con esto puede preguntar "¿en alguna hay algo escrito a mano?" y ver
    la escalada a vision ocurrir de verdad.
    """
    if estado.settings.provider_mode is not ProviderMode.FAKE:
        return

    for nombre, blob in DEMO_BLOBS.items():
        resultado = await estado.ingestion.ingest(
            filename=DEMO_FILENAMES[nombre],
            media_type=blob.media_type,
            data=blob.data,
        )
        logger.info(
            "imagen de demostracion",
            extra=log_extra(filename=DEMO_FILENAMES[nombre], id=str(resultado.record.id)),
        )


def create_app(
    *,
    settings: Settings | None = None,
    providers: Providers | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
    seed_demo: bool = True,
) -> FastAPI:
    """Factoria de la aplicacion.

    Los tres parametros opcionales son el punto de inyeccion: en produccion no
    se pasa ninguno y se construye todo desde la configuracion; en un test se
    pasan los fakes y no se abre ni un fichero.
    """
    ajustes = settings or get_settings()
    configure_logging(ajustes.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if providers is not None and checkpointer is not None:
            estado = _montar_estado(ajustes, providers, checkpointer)
            app.state.imagent = estado
            if seed_demo:
                await sembrar_demo(estado)
            try:
                yield
            finally:
                await _cerrar(estado)
            return

        # Camino de produccion: el checkpointer es un context manager porque su
        # conexion hay que cerrarla, y el lifespan es justo quien sabe cuando.
        async with open_checkpointer(ajustes) as saver:
            estado = _montar_estado(ajustes, providers or build_providers(ajustes), saver)
            app.state.imagent = estado
            await estado.providers.store.ensure_ready()
            if seed_demo:
                await sembrar_demo(estado)
            try:
                yield
            finally:
                await _cerrar(estado)

    app = FastAPI(title="imagent", version="0.1.0", lifespan=lifespan)

    @app.middleware("http")
    async def correlacionar(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Un identificador por peticion, propagado por ContextVar.

        Es lo que permite juntar en los logs todo lo que le paso a una sola
        pregunta, incluidos los eventos de degradacion que emiten los nodos del
        grafo cinco llamadas mas abajo.
        """
        token = request_id.set(request.headers.get("x-request-id") or uuid4().hex[:12])
        try:
            respuesta = await call_next(request)
            respuesta.headers["x-request-id"] = request_id.get()
            return respuesta
        finally:
            request_id.reset(token)

    @app.exception_handler(InvalidImageError)
    async def _imagen_invalida(_: Request, exc: InvalidImageError) -> JSONResponse:
        """Culpa de quien llama: 400, y se dice por que."""
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorOut(error="invalid_image", detail=str(exc)).model_dump(),
        )

    @app.exception_handler(ImagentError)
    async def _fallo_de_proveedor(_: Request, exc: ImagentError) -> JSONResponse:
        """Un fallo que llega hasta aqui es fatal por contrato.

        Todo lo que podia degradar ya degrado mas abajo; si una excepcion del
        dominio llega a la capa HTTP es porque alguien decidio que ese fallo
        tenia que salir a la luz. 502 y no 500: el problema es de algo de lo que
        dependemos, no del proceso.
        """
        logger.exception("fallo no degradable", extra=log_extra(error_type=type(exc).__name__))
        return JSONResponse(
            status_code=status.HTTP_502_BAD_GATEWAY,
            content=ErrorOut(error=type(exc).__name__, detail=str(exc)).model_dump(),
        )

    app.include_router(health.router)
    app.include_router(images.router)
    app.include_router(chat.router)

    if WEB.is_dir():
        app.mount("/static", StaticFiles(directory=WEB), name="static")

        @app.get("/", include_in_schema=False)
        async def indice() -> FileResponse:
            return FileResponse(WEB / "index.html")

    return app


app = create_app()
