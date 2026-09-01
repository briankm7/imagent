"""Tests de la capa HTTP.

Levantan la aplicacion entera con los proveedores fake inyectados: sin disco,
sin sqlite y sin red. Es el mismo `create_app` que usa produccion; lo unico que
cambia es lo que se le pasa.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from imagent.agents.graph import memory_checkpointer
from imagent.api.main import create_app
from imagent.config import Settings
from imagent.domain.errors import VectorStoreError
from imagent.providers.fake.repository import InMemoryImageRepository
from imagent.providers.fake.vectorstore import InMemoryVectorStore
from imagent.providers.registry import Providers
from tests.helpers import DIMENSIONS, montar_providers, montar_providers_de_app

FOTO = b"\x89PNG\r\n\x1a\n bytes de una foto nueva"


def cliente(
    *, providers: Providers | None = None, seed: bool = True, **ajustes: Any
) -> Iterator[TestClient]:
    settings = Settings(**ajustes)
    app = create_app(
        settings=settings,
        providers=providers or montar_providers_de_app(),
        checkpointer=memory_checkpointer(),
        seed_demo=seed,
    )
    with TestClient(app) as c:
        yield c


@pytest.fixture
def api() -> Iterator[TestClient]:
    yield from cliente()


@pytest.fixture
def api_vacia() -> Iterator[TestClient]:
    yield from cliente(seed=False)


# ---------------------------------------------------------------------------
# Salud y demostracion
# ---------------------------------------------------------------------------
def test_health_dice_en_que_modo_arranco(api: TestClient) -> None:
    """En un despliegue real, "¿esto habla con Gemini o con los fakes?" tiene que
    poder responderse sin entrar en la maquina."""
    cuerpo = api.get("/health").json()

    assert cuerpo["status"] == "ok"
    assert cuerpo["provider_mode"] == "fake"
    assert cuerpo["images_indexed"] == 3


def test_arranca_con_el_escenario_de_demostracion(api: TestClient) -> None:
    """Quien clone el repo y arranque sin claves tiene algo que preguntar."""
    nombres = [i["filename"] for i in api.get("/api/images").json()]

    assert nombres == ["coche.jpg", "libreta.png", "playa.jpg"]


def test_las_imagenes_salen_en_orden_de_subida(api: TestClient) -> None:
    """El orden es contrato: "la tercera" depende de el."""
    fechas = [i["created_at"] for i in api.get("/api/images").json()]

    assert fechas == sorted(fechas)


# ---------------------------------------------------------------------------
# Subida
# ---------------------------------------------------------------------------
def test_subir_una_imagen(api_vacia: TestClient) -> None:
    respuesta = api_vacia.post("/api/images", files={"file": ("nueva.png", FOTO, "image/png")})

    assert respuesta.status_code == status.HTTP_201_CREATED
    cuerpo = respuesta.json()
    assert cuerpo["deduplicated"] is False
    assert cuerpo["image"]["filename"] == "nueva.png"
    assert cuerpo["image"]["status"] == "indexed"


def test_subir_dos_veces_lo_mismo_no_crea_dos_registros(api_vacia: TestClient) -> None:
    """Decision D10-A vista desde fuera."""
    primera = api_vacia.post("/api/images", files={"file": ("a.png", FOTO, "image/png")})
    segunda = api_vacia.post("/api/images", files={"file": ("b.png", FOTO, "image/png")})

    assert segunda.json()["deduplicated"] is True
    assert segunda.json()["image"]["id"] == primera.json()["image"]["id"]
    assert len(api_vacia.get("/api/images").json()) == 1


def test_un_pdf_se_rechaza_con_400(api_vacia: TestClient) -> None:
    """Culpa de quien llama: 400 y se dice por que."""
    respuesta = api_vacia.post(
        "/api/images", files={"file": ("x.pdf", b"%PDF-1.4", "application/pdf")}
    )

    assert respuesta.status_code == status.HTTP_400_BAD_REQUEST
    assert respuesta.json()["error"] == "invalid_image"
    assert "application/pdf" in respuesta.json()["detail"]


def test_una_imagen_demasiado_grande_se_rechaza_con_413() -> None:
    """Se corta mientras se recibe, no despues de tenerla entera en memoria."""
    for c in cliente(seed=False, max_upload_bytes=1024):
        respuesta = c.post("/api/images", files={"file": ("grande.png", b"x" * 5000, "image/png")})

        assert respuesta.status_code == status.HTTP_413_CONTENT_TOO_LARGE
        assert c.get("/api/images").json() == []


def test_los_bytes_se_pueden_recuperar(api: TestClient) -> None:
    imagen = api.get("/api/images").json()[0]

    respuesta = api.get(f"/api/images/{imagen['id']}/bytes")

    assert respuesta.status_code == status.HTTP_200_OK
    assert respuesta.headers["content-type"] == imagen["media_type"]


def test_una_imagen_que_no_existe_da_404(api: TestClient) -> None:
    assert api.get(f"/api/images/{uuid4()}").status_code == status.HTTP_404_NOT_FOUND


def test_registro_sin_bytes_da_410_y_no_404(api: TestClient) -> None:
    """404 sugeriria que la imagen nunca existio. 410 dice lo que pasa de verdad:
    el registro esta y sus bytes no."""
    imagen = api.get("/api/images").json()[0]
    estado = api.app.state.imagent  # type: ignore[attr-defined]
    estado.providers.blobs._blobs.clear()

    respuesta = api.get(f"/api/images/{imagen['id']}/bytes")

    assert respuesta.status_code == status.HTTP_410_GONE


# ---------------------------------------------------------------------------
# Conversacion
# ---------------------------------------------------------------------------
def test_una_pregunta_devuelve_respuesta_y_thread(api: TestClient) -> None:
    cuerpo = api.post("/api/chat", json={"message": "¿en cuales aparece un coche?"}).json()

    assert cuerpo["answer"]
    assert cuerpo["thread_id"]
    assert cuerpo["incomplete"] is False


def test_la_escalada_a_vision_funciona_de_punta_a_punta(api: TestClient) -> None:
    """Sin claves, sin base de datos, por HTTP: el caso central del proyecto."""
    estado = api.app.state.imagent  # type: ignore[attr-defined]
    antes = len(estado.providers.vision.on_demand_calls)

    cuerpo = api.post("/api/chat", json={"message": "¿en alguna hay algo escrito a mano?"}).json()

    assert len(estado.providers.vision.on_demand_calls) - antes == 3
    assert "manuscrita" in cuerpo["answer"]
    assert "libreta.png" in cuerpo["answer"]


def test_una_pregunta_que_se_resuelve_buscando_no_paga_vision(api: TestClient) -> None:
    estado = api.app.state.imagent  # type: ignore[attr-defined]
    antes = len(estado.providers.vision.on_demand_calls)

    api.post("/api/chat", json={"message": "¿en cuales aparece un coche?"})

    assert len(estado.providers.vision.on_demand_calls) == antes


def test_el_mismo_thread_continua_la_conversacion(api: TestClient) -> None:
    primera = api.post("/api/chat", json={"message": "¿hay coches?"}).json()

    segunda = api.post(
        "/api/chat", json={"message": "¿y libretas?", "thread_id": primera["thread_id"]}
    ).json()

    assert segunda["thread_id"] == primera["thread_id"]


def test_el_presupuesto_agotado_sale_marcado_en_la_respuesta() -> None:
    """Lo que el front necesita para poder avisar al usuario."""
    for c in cliente(budget={"max_vision_images": 0}):
        cuerpo = c.post("/api/chat", json={"message": "¿en alguna hay algo escrito a mano?"}).json()

        assert cuerpo["incomplete"] is True
        assert cuerpo["incomplete_reason"] == "budget_vision"


def test_las_degradaciones_salen_en_la_respuesta_no_solo_en_los_logs() -> None:
    """Un fallo que solo esta en el log es silencioso para todo el que no tenga
    acceso al log."""

    class StoreRoto(InMemoryVectorStore):
        async def search(self, *args: Any, **kwargs: Any) -> list[Any]:
            raise VectorStoreError("connection refused")

    providers = montar_providers(store=StoreRoto(dimensions=DIMENSIONS))
    for c in cliente(providers=providers):
        cuerpo = c.post("/api/chat", json={"message": "¿hay coches?"}).json()

        assert cuerpo["warnings"]
        assert any(w["event"] == "retrieval.failed" for w in cuerpo["warnings"])
        # Y aun asi se responde: el turno no se cae por una busqueda fallida.
        assert cuerpo["answer"]


def test_un_mensaje_vacio_se_rechaza(api: TestClient) -> None:
    respuesta = api.post("/api/chat", json={"message": ""})

    assert respuesta.status_code == status.HTTP_422_UNPROCESSABLE_CONTENT


# ---------------------------------------------------------------------------
# Correlacion
# ---------------------------------------------------------------------------
def test_cada_respuesta_lleva_su_identificador(api: TestClient) -> None:
    assert api.get("/health").headers["x-request-id"]


def test_se_respeta_el_identificador_que_manda_el_cliente(api: TestClient) -> None:
    """Permite seguir una peticion a traves de varios servicios."""
    respuesta = api.get("/health", headers={"x-request-id": "abc123"})

    assert respuesta.headers["x-request-id"] == "abc123"


# ---------------------------------------------------------------------------
# Apagado
# ---------------------------------------------------------------------------
def test_al_apagar_se_sueltan_los_recursos() -> None:
    """Sin esto, la conexion de SQLite deja su hilo vivo despues del event loop
    y el proceso termina con un "Event loop is closed" desde ese hilo. Un fallo
    al apagar es facil de ignorar y siempre significa que algo no se solto."""
    cerrados: list[str] = []

    class RepoQueSeCierra(InMemoryImageRepository):
        async def close(self) -> None:
            cerrados.append("repositorio")

    providers = montar_providers(repository=RepoQueSeCierra())

    for _ in cliente(providers=providers, seed=False):
        pass

    assert cerrados == ["repositorio"]


def test_lo_que_no_se_puede_cerrar_no_estorba() -> None:
    """Cerrar es una capacidad OPCIONAL: el almacen en memoria no tiene nada
    que soltar y no por eso tiene que implementar un metodo vacio."""
    for c in cliente(seed=False):
        assert c.get("/health").status_code == status.HTTP_200_OK
