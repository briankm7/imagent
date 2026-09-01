"""Tests del presupuesto.

Es la pieza que acota lo que el sistema puede gastar, asi que casi todos estos
tests son sobre los limites, no sobre el camino normal.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from imagent.agents.budget import Budget
from imagent.config import BudgetSettings


def presupuesto(*, iteraciones: int = 6, imagenes: int = 3, segundos: float = 45.0) -> Budget:
    return Budget(
        iterations_left=iteraciones,
        vision_images_left=imagenes,
        vision_seconds_left=segundos,
    )


def test_se_construye_desde_la_configuracion() -> None:
    budget = Budget.for_turn(
        BudgetSettings(max_graph_iterations=4, max_vision_images=2, max_vision_seconds=30.0)
    )

    assert budget.iterations_left == 4
    assert budget.vision_images_left == 2
    assert budget.vision_seconds_left == 30.0


def test_gastar_no_muta_el_presupuesto_anterior() -> None:
    """Un nodo que revienta a mitad no puede dejar el contador descuadrado."""
    original = presupuesto(iteraciones=6)

    gastado = original.spend_iteration()

    assert original.iterations_left == 6
    assert gastado.iterations_left == 5


def test_es_inmutable() -> None:
    with pytest.raises(ValidationError):
        presupuesto().iterations_left = 0  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Iteraciones
# ---------------------------------------------------------------------------
def test_can_iterate_mientras_quede_alguna() -> None:
    assert presupuesto(iteraciones=1).can_iterate is True
    assert presupuesto(iteraciones=0).can_iterate is False


def test_gastar_una_iteracion_que_no_existe_es_un_bug_y_revienta() -> None:
    """No es una condicion del entorno: significa que el router despacho sin
    comprobar antes si podia. Tiene que doler."""
    agotado = presupuesto(iteraciones=0)

    with pytest.raises(ValidationError):
        agotado.spend_iteration()


def test_una_conversacion_larga_agota_las_iteraciones() -> None:
    budget = presupuesto(iteraciones=3)

    for _ in range(3):
        budget = budget.spend_iteration()

    assert budget.can_iterate is False


# ---------------------------------------------------------------------------
# El clamp de imagenes: decision D3-C
# ---------------------------------------------------------------------------
def test_el_clamp_recorta_lo_que_pide_el_modelo() -> None:
    """El sitio exacto donde la propuesta del coordinador deja de ser una orden.

    Si senala veinte candidatas y el presupuesto son tres, salen tres. No se le
    pide que lo reconsidere: se recorta.
    """
    assert presupuesto(imagenes=3).allowance(20) == 3


def test_el_clamp_no_infla_lo_que_pide_el_modelo() -> None:
    """Si pide menos de lo permitido, se respeta: el presupuesto es un techo,
    no una cuota que haya que gastar."""
    assert presupuesto(imagenes=3).allowance(1) == 1


def test_sin_presupuesto_el_clamp_es_cero() -> None:
    assert presupuesto(imagenes=0).allowance(5) == 0


def test_pedir_cero_imagenes_es_valido() -> None:
    assert presupuesto(imagenes=3).allowance(0) == 0


def test_pedir_imagenes_negativas_es_un_bug() -> None:
    with pytest.raises(ValueError, match="negativas"):
        presupuesto().allowance(-1)


# ---------------------------------------------------------------------------
# El presupuesto de tiempo
# ---------------------------------------------------------------------------
def test_can_look_exige_las_dos_condiciones() -> None:
    """Con imagenes de sobra pero sin segundos, la escalada no puede ocurrir."""
    assert presupuesto(imagenes=3, segundos=10.0).can_look is True
    assert presupuesto(imagenes=0, segundos=10.0).can_look is False
    assert presupuesto(imagenes=3, segundos=0.0).can_look is False


def test_el_timeout_efectivo_es_el_minimo() -> None:
    """La ultima mirada de un lote no puede gastar mas de lo que sobra aunque su
    timeout individual sea mayor."""
    assert presupuesto(segundos=45.0).time_allowance(25.0) == 25.0
    assert presupuesto(segundos=4.0).time_allowance(25.0) == 4.0


def test_el_tiempo_se_descuenta_con_lo_realmente_transcurrido() -> None:
    """Cobrar el timeout en vez de lo tardado seria un presupuesto que miente:
    tres miradas rapidas dejarian el turno sin margen sin motivo."""
    budget = presupuesto(imagenes=3, segundos=45.0)

    gastado = budget.spend_vision(images=3, seconds=2.0)

    assert gastado.vision_seconds_left == 43.0
    assert gastado.vision_images_left == 0


def test_pasarse_de_tiempo_por_decimas_no_revienta() -> None:
    """Medir no es gratis: el tiempo se corta en cero. Las imagenes no, que esas
    se cuentan de una en una."""
    budget = presupuesto(imagenes=2, segundos=1.0)

    gastado = budget.spend_vision(images=1, seconds=1.4)

    assert gastado.vision_seconds_left == 0.0
    assert gastado.can_look is False


def test_gastar_mas_imagenes_de_las_que_quedan_es_un_bug() -> None:
    with pytest.raises(ValidationError):
        presupuesto(imagenes=2).spend_vision(images=3, seconds=1.0)


def test_gastar_cantidades_negativas_es_un_bug() -> None:
    with pytest.raises(ValueError, match="negativa"):
        presupuesto().spend_vision(images=-1, seconds=1.0)


# ---------------------------------------------------------------------------
# El recorrido completo
# ---------------------------------------------------------------------------
def test_un_turno_que_agota_la_vision_no_puede_volver_a_mirar() -> None:
    """El escenario del test de presupuesto agotado, en miniatura."""
    budget = Budget.for_turn(
        BudgetSettings(max_graph_iterations=6, max_vision_images=3, max_vision_seconds=45.0)
    )

    autorizadas = budget.allowance(10)
    budget = budget.spend_vision(images=autorizadas, seconds=8.0)

    assert autorizadas == 3
    assert budget.can_look is False
    assert budget.allowance(1) == 0
    # Pero el grafo todavia puede dar vueltas: son presupuestos independientes.
    assert budget.can_iterate is True


def test_presupuesto_de_vision_cero_desactiva_la_escalada() -> None:
    """La configuracion del test de "no hay presupuesto desde el principio"."""
    budget = Budget.for_turn(
        BudgetSettings(max_graph_iterations=6, max_vision_images=0, max_vision_seconds=45.0)
    )

    assert budget.can_look is False
    assert budget.can_iterate is True
