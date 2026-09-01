"""Tests del estado del grafo.

Los dos primeros defienden la frontera entre lo que persiste y lo que se
reinicia, que es la parte del proyecto que hay que poder explicar.
"""

from __future__ import annotations

from uuid import uuid4

from imagent.agents.budget import Budget
from imagent.agents.state import PERSISTENT_KEYS, AgentState, merge_retrieved, new_turn
from imagent.domain.models import Aspect, RetrievedImage


def presupuesto() -> Budget:
    return Budget(iterations_left=6, vision_images_left=3, vision_seconds_left=45.0)


def recuperada(nombre: str, score: float, aspectos: dict[Aspect, float], image_id=None):
    return RetrievedImage(
        image_id=image_id or uuid4(),
        filename=nombre,
        score=score,
        matched_aspects=aspectos,
    )


# ---------------------------------------------------------------------------
# El reinicio de turno
# ---------------------------------------------------------------------------
def test_new_turn_reinicia_todos_los_campos_no_persistentes() -> None:
    """El test que hace imposible olvidar un campo.

    Olvidar uno no falla: arrastraria evidencia de un turno al siguiente y el
    sistema respondería peor sin que nada se queje. Por eso se comprueba contra
    las anotaciones del TypedDict y no contra una lista escrita a mano.
    """
    esperados = set(AgentState.__annotations__) - PERSISTENT_KEYS

    entrada = set(new_turn(question="¿hay coches?", budget=presupuesto()))

    assert esperados <= entrada


def test_new_turn_no_toca_el_foco_de_la_conversacion() -> None:
    """`focus_image_ids` persiste sin reducer: al no escribirlo, el turno nuevo
    conserva el que dejo el anterior. Eso es lo que hace que "y de esa..."
    signifique algo."""
    assert "focus_image_ids" not in new_turn(question="¿y de esa?", budget=presupuesto())


def test_new_turn_abre_el_turno_con_la_pregunta() -> None:
    entrada = new_turn(question="¿en cuales aparece un coche?", budget=presupuesto())

    assert entrada["question"] == "¿en cuales aparece un coche?"
    assert len(entrada["messages"]) == 1
    assert entrada["messages"][0].content == "¿en cuales aparece un coche?"


def test_new_turn_empieza_sin_evidencia_ni_plan() -> None:
    entrada = new_turn(question="x", budget=presupuesto())

    assert entrada["retrieved"] == []
    assert entrada["vision_findings"] == []
    assert entrada["queries_done"] == []
    assert entrada["decision"] is None
    assert entrada["incomplete_reason"] is None
    assert entrada["answer"] is None


def test_solo_messages_lleva_reducer() -> None:
    """Decision D12-A, comprobada sobre el propio tipo.

    Si alguien añade un reducer a otro canal, el reinicio entre turnos deja de
    funcionar en silencio: escribir [] en un canal acumulativo no lo vacia.
    """
    con_reducer = {
        nombre
        for nombre, anotacion in AgentState.__annotations__.items()
        if "Annotated" in str(anotacion)
    }

    assert con_reducer == {"messages"}


# ---------------------------------------------------------------------------
# La mezcla de evidencia
# ---------------------------------------------------------------------------
def test_mezclar_sin_nada_previo_devuelve_lo_nuevo() -> None:
    nueva = recuperada("coche.jpg", 0.8, {Aspect.DESCRIPTION: 0.8})

    assert merge_retrieved([], [nueva]) == [nueva]


def test_mezclar_une_imagenes_distintas() -> None:
    """Reformular no reemplaza la evidencia anterior: la completa."""
    primera = recuperada("coche.jpg", 0.8, {Aspect.DESCRIPTION: 0.8})
    segunda = recuperada("libreta.png", 0.6, {Aspect.OBJECTS: 0.6})

    resultado = merge_retrieved([primera], [segunda])

    assert [r.filename for r in resultado] == ["coche.jpg", "libreta.png"]


def test_una_imagen_repetida_se_queda_con_la_puntuacion_mayor() -> None:
    image_id = uuid4()
    floja = recuperada("coche.jpg", 0.4, {Aspect.DESCRIPTION: 0.4}, image_id)
    fuerte = recuperada("coche.jpg", 0.9, {Aspect.OCR: 0.9}, image_id)

    (resultado,) = merge_retrieved([floja], [fuerte])

    assert resultado.score == 0.9


def test_una_imagen_repetida_acumula_los_aspectos_que_casaron() -> None:
    """El coordinador usa esto para juzgar: no es lo mismo casar por descripcion
    que casar por OCR."""
    image_id = uuid4()
    por_descripcion = recuperada("coche.jpg", 0.4, {Aspect.DESCRIPTION: 0.4}, image_id)
    por_ocr = recuperada("coche.jpg", 0.9, {Aspect.OCR: 0.9}, image_id)

    (resultado,) = merge_retrieved([por_descripcion], [por_ocr])

    assert resultado.matched_aspects == {Aspect.DESCRIPTION: 0.4, Aspect.OCR: 0.9}


def test_la_mezcla_ordena_de_mayor_a_menor() -> None:
    floja = recuperada("playa.jpg", 0.2, {})
    fuerte = recuperada("coche.jpg", 0.9, {})

    resultado = merge_retrieved([floja], [fuerte])

    assert [r.filename for r in resultado] == ["coche.jpg", "playa.jpg"]


def test_el_desempate_de_la_mezcla_es_estable() -> None:
    """Dos imagenes con la misma puntuacion no pueden ordenarse distinto en dos
    ejecuciones o los tests serian inestables."""
    a = recuperada("a.jpg", 0.5, {})
    b = recuperada("b.jpg", 0.5, {})

    primera = [r.image_id for r in merge_retrieved([a], [b])]
    segunda = [r.image_id for r in merge_retrieved([b], [a])]

    assert primera == segunda


def test_la_mezcla_no_muta_las_listas_de_entrada() -> None:
    previa = [recuperada("coche.jpg", 0.8, {})]

    merge_retrieved(previa, [recuperada("playa.jpg", 0.2, {})])

    assert len(previa) == 1
