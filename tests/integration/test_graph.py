"""Tests del grafo completo.

Recorren los cuatro nodos de verdad, con proveedores fake pero con el grafo
real, el checkpointer real (en memoria) y el enrutado real.

El guion del coordinador se escribe como una lista de decisiones: cada elemento
es lo que el modelo devolveria en una vuelta. Se lee como el recorrido esperado
del grafo, y si el grafo da mas vueltas de las previstas la cola se agota y el
test lo dice en vez de pasar por casualidad.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from imagent.agents.coordinator import EVENTO_CANDIDATAS_INVENTADAS
from imagent.agents.graph import Conversation, build_graph, memory_checkpointer
from imagent.agents.routing import CoordinatorDecision, IncompleteReason, Route
from imagent.config import BudgetSettings, Settings
from imagent.providers.fake.text import FakeTextProvider
from imagent.providers.registry import Providers
from tests.helpers import Escena, ingerir_escena, montar_providers


def plan(
    action: Route,
    *,
    sufficient: bool = False,
    query: str = "",
    candidatas: list[UUID] | None = None,
    gap: str = "",
) -> CoordinatorDecision:
    return CoordinatorDecision(
        action=action,
        sufficient=sufficient,
        standalone_query=query,
        candidate_image_ids=candidatas or [],
        gap=gap,
    )


async def montar_conversacion(
    *,
    planes: list[Any],
    respuesta: str = "Respuesta redactada.",
    budget: BudgetSettings | None = None,
) -> tuple[Conversation, Escena, FakeTextProvider]:
    texto = FakeTextProvider(structured=planes, completions=[respuesta])
    providers = montar_providers(text=texto)
    escena = await ingerir_escena(providers)

    settings = Settings(budget=budget) if budget else Settings()
    grafo = build_graph(providers, settings=settings, checkpointer=memory_checkpointer())
    return Conversation(grafo, settings=settings), escena, texto


# ---------------------------------------------------------------------------
# Camino normal: recuperar y responder
# ---------------------------------------------------------------------------
async def test_una_pregunta_que_se_resuelve_buscando() -> None:
    """Dos vueltas al coordinador y ni una llamada al modelo de vision."""
    conversacion, escena, texto = await montar_conversacion(
        planes=[
            plan(Route.RETRIEVE, query="coche rojo aparcado en la acera"),
            plan(Route.RESPOND, sufficient=True),
        ],
        respuesta="En coche.jpg aparece un coche rojo aparcado.",
    )

    resultado = await conversacion.ask(thread_id="t", question="¿en cuales hay un coche?")

    assert resultado.answer == "En coche.jpg aparece un coche rojo aparcado."
    assert resultado.incomplete is False
    assert escena.vision.on_demand_calls == []
    assert texto.pending == (0, 0)


# ---------------------------------------------------------------------------
# LA escalada
# ---------------------------------------------------------------------------
async def test_escala_a_vision_cuando_lo_indexado_no_basta() -> None:
    """El caso que justifica el proyecto entero.

    Nadie se fijo en la caligrafia durante la ingesta, asi que buscar no puede
    encontrarla. El coordinador decide pagar por volver a mirar, el agente de
    vision la descubre, y la respuesta la incluye.
    """
    conversacion, escena, texto = await montar_conversacion(
        planes=[], respuesta="Si: en libreta.png hay una nota manuscrita."
    )

    # El guion se rellena a posteriori porque necesita los ids de la escena.
    texto.queue_structured(
        *[
            plan(Route.RETRIEVE, query="texto escrito a mano"),
            plan(
                Route.VISION,
                candidatas=[escena.id("libreta"), escena.id("coche")],
                gap="¿hay algo manuscrito en la imagen?",
            ),
            plan(Route.RESPOND, sufficient=True),
        ]
    )

    resultado = await conversacion.ask(
        thread_id="t", question="¿en alguna hay algo escrito a mano?"
    )

    assert len(escena.vision.on_demand_calls) == 2
    assert "manuscrita" in resultado.answer
    assert resultado.incomplete is False
    assert texto.pending == (0, 0)


async def test_el_coordinador_puede_señalar_una_imagen_que_la_busqueda_no_devolvio() -> None:
    """Sin el catalogo, la escalada seria imposible en el caso que mas importa.

    Si nadie indexo la caligrafia, buscar "escrito a mano" no devuelve nada, y
    sin ids que señalar el coordinador no puede mandar mirar ninguna imagen.
    """
    conversacion, escena, texto = await montar_conversacion(planes=[])
    texto.queue_structured(
        *[
            plan(
                Route.VISION,
                candidatas=[escena.id("libreta")],
                gap="¿hay algo manuscrito?",
            ),
            plan(Route.RESPOND, sufficient=True),
        ]
    )

    resultado = await conversacion.ask(thread_id="t", question="¿hay algo escrito a mano?")

    # Ni una sola busqueda, y aun asi se ha mirado la imagen correcta.
    assert escena.vision.on_demand_calls[0][0] is not None
    assert len(escena.vision.on_demand_calls) == 1
    assert resultado.incomplete is False


# ---------------------------------------------------------------------------
# Los limites
# ---------------------------------------------------------------------------
async def test_presupuesto_de_vision_agotado_responde_marcado_como_incompleto() -> None:
    """El otro test central: se responde lo que se sepa, y se dice que falta."""
    conversacion, escena, texto = await montar_conversacion(
        planes=[],
        budget=BudgetSettings(max_vision_images=0),
    )
    texto.queue_structured(
        *[
            plan(Route.RETRIEVE, query="escrito a mano"),
            plan(Route.VISION, candidatas=[escena.id("libreta")], gap="¿manuscrito?"),
        ]
    )

    resultado = await conversacion.ask(thread_id="t", question="¿hay algo escrito a mano?")

    assert resultado.incomplete is True
    assert resultado.incomplete_reason is IncompleteReason.BUDGET_VISION
    # Y lo que importa: NO se ha pagado ni una sola llamada de vision.
    assert escena.vision.on_demand_calls == []


async def test_el_presupuesto_recorta_lo_que_pide_el_coordinador() -> None:
    """Señala tres imagenes; el presupuesto son dos; se miran dos."""
    conversacion, escena, texto = await montar_conversacion(
        planes=[],
        budget=BudgetSettings(max_vision_images=2),
    )
    texto.queue_structured(
        *[
            plan(
                Route.VISION,
                candidatas=[escena.id("libreta"), escena.id("coche"), escena.id("playa")],
                gap="¿manuscrito?",
            ),
            plan(Route.RESPOND, sufficient=True),
        ]
    )

    await conversacion.ask(thread_id="t", question="¿hay algo escrito a mano?")

    assert len(escena.vision.on_demand_calls) == 2


async def test_presupuesto_de_iteraciones_agotado_corta_y_responde() -> None:
    """Un coordinador que no para solo. El sistema para por el.

    El guion pide buscar una y otra vez con consultas distintas, que es
    perfectamente legal segun D13-B. Lo que lo detiene es el contador.
    """
    conversacion, _escena, texto = await montar_conversacion(
        planes=[
            plan(Route.RETRIEVE, query="consulta uno"),
            plan(Route.RETRIEVE, query="consulta dos"),
            plan(Route.RETRIEVE, query="consulta tres"),
            plan(Route.RETRIEVE, query="consulta cuatro"),
            plan(Route.RETRIEVE, query="consulta cinco"),
            plan(Route.RETRIEVE, query="consulta seis"),
            plan(Route.RETRIEVE, query="consulta siete"),
        ],
        budget=BudgetSettings(max_graph_iterations=3),
    )

    resultado = await conversacion.ask(thread_id="t", question="da igual")

    assert resultado.incomplete is True
    assert resultado.incomplete_reason is IncompleteReason.BUDGET_ITERATIONS
    # Tres vueltas al coordinador: dos que buscan y la tercera que ya no puede.
    assert len(texto.structured_calls) == 3


async def test_el_cinturon_de_seguridad_no_llega_a_saltar() -> None:
    """El freno es el presupuesto; el recursion_limit solo esta por si el freno
    tiene un bug. Si saltara, la excepcion impediria devolver nada."""
    conversacion, _escena, _texto = await montar_conversacion(
        planes=[plan(Route.RETRIEVE, query=f"consulta {i}") for i in range(20)],
        budget=BudgetSettings(max_graph_iterations=6),
    )

    resultado = await conversacion.ask(thread_id="t", question="da igual")

    assert resultado.incomplete_reason is IncompleteReason.BUDGET_ITERATIONS
    assert resultado.answer


async def test_repetir_la_misma_busqueda_corta_el_turno() -> None:
    """D13-B en el grafo real: reformular vale, repetirse no."""
    conversacion, _escena, _texto = await montar_conversacion(
        planes=[
            plan(Route.RETRIEVE, query="coches"),
            plan(Route.RETRIEVE, query="¿COCHES?"),
        ]
    )

    resultado = await conversacion.ask(thread_id="t", question="¿coches?")

    assert resultado.incomplete_reason is IncompleteReason.REPEATED_QUERY


# ---------------------------------------------------------------------------
# Memoria entre turnos
# ---------------------------------------------------------------------------
async def test_el_historial_persiste_entre_turnos() -> None:
    conversacion, _escena, texto = await montar_conversacion(
        planes=[
            plan(Route.RETRIEVE, query="coche rojo"),
            plan(Route.RESPOND, sufficient=True),
            plan(Route.RESPOND, sufficient=True),
        ],
        respuesta="Primera respuesta.",
    )
    texto.queue_completions("Segunda respuesta.")

    await conversacion.ask(thread_id="hilo", question="¿en cuales hay un coche?")
    await conversacion.ask(thread_id="hilo", question="¿y de esa, que pone?")

    # La segunda llamada al coordinador ve la conversacion entera.
    _, ultimo_prompt, _ = texto.structured_calls[-1]
    assert "¿en cuales hay un coche?" in ultimo_prompt
    assert "Primera respuesta." in ultimo_prompt


async def test_dos_hilos_no_se_mezclan() -> None:
    conversacion, _escena, texto = await montar_conversacion(
        planes=[plan(Route.RESPOND, sufficient=True), plan(Route.RESPOND, sufficient=True)],
        respuesta="Respuesta A.",
    )
    texto.queue_completions("Respuesta B.")

    await conversacion.ask(thread_id="hilo-a", question="pregunta de A")
    await conversacion.ask(thread_id="hilo-b", question="pregunta de B")

    _, prompt_de_b, _ = texto.structured_calls[-1]
    assert "pregunta de A" not in prompt_de_b


async def test_el_presupuesto_se_reinicia_en_cada_turno() -> None:
    """Si se arrastrase, la segunda pregunta no tendria derecho a mirar nada."""
    conversacion, escena, texto = await montar_conversacion(
        planes=[], budget=BudgetSettings(max_vision_images=1)
    )
    texto.queue_structured(
        *[
            plan(Route.VISION, candidatas=[escena.id("libreta")], gap="¿manuscrito?"),
            plan(Route.RESPOND, sufficient=True),
            plan(Route.VISION, candidatas=[escena.id("playa")], gap="¿iluminada?"),
            plan(Route.RESPOND, sufficient=True),
        ]
    )
    texto.queue_completions("Segunda respuesta.")

    await conversacion.ask(thread_id="hilo", question="¿escrito a mano?")
    segundo = await conversacion.ask(thread_id="hilo", question="¿bien iluminada?")

    assert len(escena.vision.on_demand_calls) == 2
    assert segundo.incomplete is False


async def test_la_evidencia_no_se_arrastra_entre_turnos() -> None:
    """La respuesta al turno dos no puede apoyarse en lo que se busco en el uno."""
    conversacion, _escena, texto = await montar_conversacion(
        planes=[
            plan(Route.RETRIEVE, query="coche rojo aparcado"),
            plan(Route.RESPOND, sufficient=True),
            plan(Route.RESPOND, sufficient=True),
        ],
        respuesta="Primera.",
    )
    texto.queue_completions("Segunda.")

    await conversacion.ask(thread_id="hilo", question="¿coches?")
    await conversacion.ask(thread_id="hilo", question="¿otra cosa?")

    _, prompt_del_segundo, _ = texto.structured_calls[-1]
    assert "(no se ha recuperado nada)" in prompt_del_segundo


# ---------------------------------------------------------------------------
# Degradacion visible
# ---------------------------------------------------------------------------
async def test_las_candidatas_inventadas_se_descartan_y_se_notan() -> None:
    """Un modelo puede devolver un UUID con toda la seguridad del mundo."""
    conversacion, escena, texto = await montar_conversacion(planes=[])
    texto.queue_structured(
        *[
            plan(Route.VISION, candidatas=[uuid4()], gap="¿manuscrito?"),
            plan(Route.RESPOND, sufficient=True),
        ]
    )

    resultado = await conversacion.ask(thread_id="t", question="da igual")

    assert escena.vision.on_demand_calls == []
    assert EVENTO_CANDIDATAS_INVENTADAS in [d.event for d in resultado.degradations]
    assert resultado.incomplete_reason is IncompleteReason.NO_CANDIDATES


async def test_las_degradaciones_llegan_al_resultado_del_turno() -> None:
    """Un camino degradado que solo se ve en los logs no se ve en una demo."""
    conversacion, _escena, texto = await montar_conversacion(planes=[])
    texto.queue_structured(
        *[
            plan(Route.VISION, candidatas=[uuid4(), uuid4()], gap="x"),
            plan(Route.RESPOND, sufficient=True),
        ]
    )

    resultado = await conversacion.ask(thread_id="t", question="da igual")

    assert resultado.degradations
    assert all(d.detail for d in resultado.degradations)


@pytest.mark.parametrize("motivo", list(IncompleteReason))
def test_todos_los_motivos_de_incompletitud_tienen_explicacion(motivo: IncompleteReason) -> None:
    """El redactor traduce el enum a lenguaje natural; si alguien añade un motivo
    y olvida la traduccion, el usuario leeria el valor crudo del enum."""
    from imagent.agents.responder import explicar

    explicacion = explicar(motivo)

    assert explicacion.startswith("incompleta: ")
    assert motivo.value not in explicacion


def test_el_grafo_tiene_topologia_de_radios(providers: Providers) -> None:
    """Una sola arista condicional, y todos los radios vuelven al centro."""
    grafo = build_graph(providers, settings=Settings(), checkpointer=memory_checkpointer())

    nodos = set(grafo.get_graph().nodes) - {"__start__", "__end__"}
    assert nodos == {"coordinator", "retrieve", "vision", "respond"}

    # La afirmacion fuerte: hay UNA sola arista condicional, y sale del
    # coordinador. Es lo que hace que el enrutado tenga un unico sitio.
    assert list(grafo.builder.branches) == ["coordinator"]

    # Y los radios siempre vuelven al centro, sin atajos entre ellos.
    aristas = grafo.builder.edges
    assert ("retrieve", "coordinator") in aristas
    assert ("vision", "coordinator") in aristas
    assert not [a for a in aristas if a[0] in {"retrieve", "vision"} and a[1] != "coordinator"]
