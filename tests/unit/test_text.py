"""Tests de la normalizacion de texto."""

from __future__ import annotations

from imagent.domain.text import fingerprint, normalize, strip_accents


def test_strip_accents_conserva_la_letra() -> None:
    assert strip_accents("caligrafía española") == "caligrafia espanola"


def test_normalize_baja_a_minusculas_y_colapsa_espacios() -> None:
    assert normalize("  Qué   PONE  ") == "que pone"


def test_normalize_conserva_la_puntuacion() -> None:
    """El guion del fake de vision busca subcadenas dentro del texto, asi que
    normalize no puede destruirlo."""
    assert normalize("¿Manuscrito?") == "¿manuscrito?"


def test_fingerprint_quita_la_puntuacion() -> None:
    """Sin esto, una interrogacion bastaria para saltarse el corte de busquedas
    repetidas."""
    assert fingerprint("¿Qué  imágenes tienen COCHES?") == "que imagenes tienen coches"


def test_fingerprint_iguala_lo_que_es_la_misma_consulta() -> None:
    assert fingerprint("¿Hay algo escrito a mano?") == fingerprint("hay algo escrito a mano")


def test_fingerprint_no_iguala_consultas_distintas() -> None:
    assert fingerprint("caligrafia") != fingerprint("escrito a mano")
