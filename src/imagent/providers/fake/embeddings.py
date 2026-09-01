"""EmbeddingProvider determinista y offline (decision P6-B).

No hay modelo detras: cada token del texto suma en una posicion fija del
vector y luego se normaliza. Es una bolsa de palabras hasheada, tambien
llamada hashing trick.

Por que esto y no un vector pseudoaleatorio a partir del hash del texto
completo: un vector aleatorio es determinista pero **no tiene estructura**.
"coche" no se pareceria a "coche rojo" mas que a "cartel de la calle", y
entonces ningun test de recuperacion podria afirmar nada mas alla de "devuelve
algo". Con esto, buscar "coche" recupera la imagen descrita con "coche", y los
tests de integracion del grafo comprueban comportamiento de verdad.

LIMITACION, y va en el README: esto es parecido LEXICO, no semantico.
"automovil" no casa con "coche". Los tests estan escritos sabiendolo. Sirve
para verificar el cableado del sistema, no la calidad de la recuperacion real.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from imagent.domain.text import strip_accents
from imagent.providers.base import Vector

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Minusculas, sin acentos, solo alfanumerico.

    Quitar los acentos importa en español: "que pone en el cartel" y "qué pone
    en el cartel" tienen que producir el mismo vector.
    """
    return _TOKEN.findall(strip_accents(text.lower()))


class FakeEmbeddingProvider:
    """Implementa EmbeddingProvider sin salir del proceso."""

    def __init__(self, dimensions: int = 768) -> None:
        if dimensions < 1:
            raise ValueError("dimensions tiene que ser positivo")
        self._dimensions = dimensions

        # Por simetria con los otros fakes: un test necesita poder afirmar que
        # los tres aspectos se embeben en UNA llamada y no en tres.
        self.documents_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def model_name(self) -> str:
        return f"fake-embeddings-{self._dimensions}"

    def _bucket(self, token: str) -> int:
        """Posicion estable del token dentro del vector.

        blake2b y no la `hash()` de Python: `hash()` de un str esta aleatorizada
        por proceso (PYTHONHASHSEED), asi que los vectores cambiarian entre
        ejecuciones y el determinismo, que es la razon de ser de este fake, se
        perderia sin que nada fallara de forma evidente.
        """
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._dimensions

    def _vector(self, text: str) -> Vector:
        vector = [0.0] * self._dimensions
        for token in tokenize(text):
            vector[self._bucket(token)] += 1.0

        norma = math.sqrt(sum(v * v for v in vector))
        if norma == 0.0:
            # Texto vacio o solo simbolos. Un vector de ceros haria que el
            # coseno fuese 0/0; se devuelve un unitario fijo para que el
            # almacen no tenga que tratar un caso especial.
            vector[0] = 1.0
            return vector

        return [v / norma for v in vector]

    async def embed_documents(self, texts: Sequence[str]) -> list[Vector]:
        self.documents_calls.append(list(texts))
        return [self._vector(text) for text in texts]

    async def embed_query(self, text: str) -> Vector:
        self.query_calls.append(text)
        # En el proveedor real esto NO es igual que embed_documents: Gemini
        # distingue RETRIEVAL_QUERY de RETRIEVAL_DOCUMENT. Aqui coinciden
        # porque el fake no tiene esa nocion.
        return self._vector(text)
