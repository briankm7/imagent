"""Tests de la tabla de enrutado.

Es la logica central del proyecto y se prueba entera sin grafo, sin proveedores
y sin generar una sola palabra de prosa. Eso es lo que compra partir el
coordinador en dos nodos y que `route()` sea una funcion pura.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from imagent.agents.budget import Budget
from imagent.agents.routing import (
    CoordinatorDecision,
    IncompleteReason,
    Route,
    RoutingOutcome,
    route,
)


def presupuesto(*, iteraciones: int = 6, imagenes: int = 3, segundos: float = 45.0) -> Budget:
    return Budget(
        iterations_left=iteraciones,
        vision_images_left=imagenes,
        vision_seconds_left=segundos,
    )


def plan(
    action: Route,
    *,
    sufficient: bool = False,
    query: str = "",
    candidatas: list[UUID] | None = None,
) -> CoordinatorDecision:
    return CoordinatorDecision(
        action=action,
        sufficient=sufficient,
        standalone_query=query,
        candidate_image_ids=candidatas or [],
    )


def enrutar(
    decision: CoordinatorDecision | None,
    *,
    budget: Budget | None = None,
    queries_done: list[str] | None = None,
    has_evidence: bool = True,
) -> RoutingOutcome:
    return route(
        decision=decision,
        budget=budget or presupuesto(),
        queries_done=queries_done or [],
        has_evidence=has_evidence,
    )


# ---------------------------------------------------------------------------
# El orden de las comprobaciones
# ---------------------------------------------------------------------------
def test_sin_iteraciones_se_responde_pase_lo_que_pase() -> None:
    """El corte de presupuesto va ANTES de mirar que pidio el modelo.

    Si fuera despues, una propuesta del coordinador podria colarse por delante
    del limite, y el limite dejaria de serlo.
    """
    quiere_mirar = plan(Route.VISION, candidatas=[uuid4()])

    resultado = enrutar(quiere_mirar, budget=presupuesto(iteraciones=0))

    assert resultado.route is Route.RESPOND
    assert resultado.incomplete_reason is IncompleteReason.BUDGET_ITERATIONS
    assert resultado.vision_images == 0


def test_sin_iteraciones_ni_siquiera_se_busca() -> None:
    resultado = enrutar(plan(Route.RETRIEVE, query="coches"), budget=presupuesto(iteraciones=0))

    assert resultado.route is Route.RESPOND
    assert resultado.incomplete_reason is IncompleteReason.BUDGET_ITERATIONS


def test_un_plan_que_no_llego_responde_en_vez_de_reventar() -> None:
    """Es un bug, pero tumbar el turno entero seria peor que responder a medias."""
    resultado = enrutar(None)

    assert resultado.route is Route.RESPOND
    assert resultado.incomplete_reason is IncompleteReason.INVALID_PLAN


# ---------------------------------------------------------------------------
# Recuperacion (decision D13-B)
# ---------------------------------------------------------------------------
def test_una_busqueda_nueva_va_a_recuperacion() -> None:
    resultado = enrutar(plan(Route.RETRIEVE, query="imagenes con coches"))

    assert resultado.route is Route.RETRIEVE
    assert resultado.incomplete is False


def test_reformular_la_busqueda_esta_permitido() -> None:
    """Comportamiento de agente de verdad: 'busca caligrafia en vez de escrito a mano'."""
    resultado = enrutar(
        plan(Route.RETRIEVE, query="caligrafia"),
        queries_done=["texto escrito a mano"],
    )

    assert resultado.route is Route.RETRIEVE


def test_repetir_la_misma_busqueda_no() -> None:
    """No puede dar un resultado distinto: solo quemaria iteraciones."""
    resultado = enrutar(
        plan(Route.RETRIEVE, query="imagenes con coches"),
        queries_done=["imagenes con coches"],
    )

    assert resultado.route is Route.RESPOND
    assert resultado.incomplete_reason is IncompleteReason.REPEATED_QUERY


def test_la_repeticion_se_detecta_pese_a_tildes_y_mayusculas() -> None:
    """Si no, cambiar una tilde bastaria para saltarse el corte."""
    resultado = enrutar(
        plan(Route.RETRIEVE, query="¿Qué  imágenes tienen COCHES?"),
        queries_done=["que imagenes tienen coches"],
    )

    assert resultado.incomplete_reason is IncompleteReason.REPEATED_QUERY


def test_buscar_sin_consulta_es_un_plan_invalido() -> None:
    resultado = enrutar(plan(Route.RETRIEVE, query="   "))

    assert resultado.route is Route.RESPOND
    assert resultado.incomplete_reason is IncompleteReason.INVALID_PLAN


# ---------------------------------------------------------------------------
# La escalada a vision
# ---------------------------------------------------------------------------
def test_la_escalada_ocurre_con_presupuesto_y_candidatas() -> None:
    """El caso de '¿en alguna hay algo escrito a mano?'."""
    resultado = enrutar(plan(Route.VISION, candidatas=[uuid4(), uuid4()]))

    assert resultado.route is Route.VISION
    assert resultado.vision_images == 2
    assert resultado.incomplete is False


def test_el_presupuesto_recorta_las_candidatas() -> None:
    """El coordinador senala diez; el presupuesto son tres; salen tres."""
    diez = [uuid4() for _ in range(10)]

    resultado = enrutar(plan(Route.VISION, candidatas=diez), budget=presupuesto(imagenes=3))

    assert resultado.route is Route.VISION
    assert resultado.vision_images == 3


def test_sin_presupuesto_de_vision_no_se_escala() -> None:
    """El otro test central: se responde lo que se sepa, marcado."""
    resultado = enrutar(plan(Route.VISION, candidatas=[uuid4()]), budget=presupuesto(imagenes=0))

    assert resultado.route is Route.RESPOND
    assert resultado.incomplete_reason is IncompleteReason.BUDGET_VISION
    assert resultado.vision_images == 0


def test_sin_tiempo_tampoco_se_escala() -> None:
    """Aunque queden imagenes: el presupuesto de tiempo es independiente."""
    resultado = enrutar(
        plan(Route.VISION, candidatas=[uuid4()]),
        budget=presupuesto(imagenes=3, segundos=0.0),
    )

    assert resultado.route is Route.RESPOND
    assert resultado.incomplete_reason is IncompleteReason.BUDGET_VISION


def test_querer_mirar_sin_senalar_imagenes_no_es_lo_mismo_que_no_tener_presupuesto() -> None:
    """Llevan al mismo sitio pero significan cosas distintas, y quien lea los
    logs necesita saber cual de las dos fue."""
    resultado = enrutar(plan(Route.VISION, candidatas=[]))

    assert resultado.route is Route.RESPOND
    assert resultado.incomplete_reason is IncompleteReason.NO_CANDIDATES


def test_sin_candidatas_manda_esa_razon_aunque_tampoco_haya_presupuesto() -> None:
    """El plan esta mal antes de que el presupuesto llegue a importar."""
    resultado = enrutar(plan(Route.VISION, candidatas=[]), budget=presupuesto(imagenes=0))

    assert resultado.incomplete_reason is IncompleteReason.NO_CANDIDATES


# ---------------------------------------------------------------------------
# Responder
# ---------------------------------------------------------------------------
def test_responder_con_evidencia_suficiente_no_marca_nada() -> None:
    resultado = enrutar(plan(Route.RESPOND, sufficient=True), has_evidence=True)

    assert resultado.route is Route.RESPOND
    assert resultado.incomplete is False


def test_decir_que_basta_sin_tener_nada_manda_a_buscar() -> None:
    """La guarda contra una afirmacion que el coordinador no puede sostener.

    Visto contra un modelo real: decia que le bastaba la evidencia sin haber
    recogido ninguna, y el redactor contestaba "no dispongo de informacion" con
    la respuesta marcada como COMPLETA. Es la peor combinacion posible, porque
    el usuario no tiene forma de saber que el sistema no lo intento.
    """
    resultado = enrutar(
        plan(Route.RESPOND, sufficient=True, query="hay coches"),
        has_evidence=False,
        queries_done=[],
    )

    assert resultado.route is Route.RETRIEVE


def test_tras_una_busqueda_vacia_si_se_responde() -> None:
    """La guarda no puede convertirse en un bucle: si ya se busco y no habia
    nada, insistir no va a cambiarlo."""
    resultado = enrutar(
        plan(Route.RESPOND, sufficient=True),
        has_evidence=False,
        queries_done=["coches"],
    )

    assert resultado.route is Route.RESPOND


def test_rendirse_sin_evidencia_no_se_reintenta() -> None:
    """La guarda solo corrige a quien AFIRMA que le basta. Quien admite que no
    le basta ya esta diciendo la verdad."""
    resultado = enrutar(plan(Route.RESPOND, sufficient=False), has_evidence=False)

    assert resultado.route is Route.RESPOND
    assert resultado.incomplete_reason is IncompleteReason.NO_EVIDENCE


def test_rendirse_marca_la_respuesta_como_incompleta() -> None:
    """El coordinador sabe que le falta informacion y aun asi responde: el
    usuario tiene derecho a saberlo."""
    resultado = enrutar(plan(Route.RESPOND, sufficient=False))

    assert resultado.route is Route.RESPOND
    assert resultado.incomplete_reason is IncompleteReason.NO_EVIDENCE


# ---------------------------------------------------------------------------
# El contrato de la salida estructurada
# ---------------------------------------------------------------------------
def test_una_accion_inventada_se_rechaza_al_validar() -> None:
    """El enum cerrado de D3-C: si el modelo devuelve algo que no existe, no
    llega al grafo."""
    with pytest.raises(ValueError, match="action"):
        CoordinatorDecision(action="teletransportar", sufficient=False)  # type: ignore[arg-type]


def test_route_no_depende_de_langgraph() -> None:
    """La logica de enrutado se puede probar sin construir un grafo.

    Si algun dia este import aparece en routing.py, el fichero ha dejado de ser
    la funcion pura que justifica partir el coordinador en dos nodos.
    """
    import inspect

    import imagent.agents.routing as modulo

    assert "langgraph" not in inspect.getsource(modulo)
