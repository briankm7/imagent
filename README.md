# imagent

Agente multimodal con memoria: subes imágenes y preguntas sobre ellas en lenguaje
natural. Tres agentes orquestados con LangGraph, y una decisión explícita y
acotada sobre cuándo vale la pena volver a mirar una imagen.

```
¿en cuáles aparece un coche?          → 0 llamadas al modelo de visión
¿en alguna hay algo escrito a mano?   → 3 llamadas al modelo de visión
```

Esa diferencia es el proyecto.

---

## Arrancar sin nada

Sin API keys, sin base de datos, sin Docker:

```bash
python -m venv .venv
.venv/Scripts/activate      # Windows;  source .venv/bin/activate en Linux/macOS
pip install -e ".[dev]"
pytest                      # 317 tests
uvicorn imagent.api.main:app --reload
```

Abre <http://localhost:8000>. Arranca en modo `fake` con tres imágenes de ejemplo
ya indexadas, así que puedes preguntar desde el primer segundo. Todo lo que
genera va marcado con `(modo sin modelo)`: no hay ningún modelo detrás y no
queremos que lo parezca.

### Con modelos reales, sin pagar

Se puede ejecutar entero con proveedores de verdad a coste cero:

- **Gemini** tiene capa gratuita con una clave de Google AI Studio. Medido el
  2 de septiembre de 2026 contra `gemini-3.6-flash`: **20 peticiones al dia**.
  Da para probar el sistema, no para usarlo a diario — una sola pregunta que
  escale a vision se lleva media docena. Los limites cambian; comprueba la
  pagina vigente.
- **Qdrant** en local con Docker no cuesta nada.
- SQLite, disco y checkpointer no cuestan nada nunca.

```bash
pip install -e ".[gemini,qdrant,dev]"
cp .env.example .env                 # pon tu IMAGENT_GOOGLE_API_KEY
docker compose -f docker/docker-compose.yml up -d qdrant

IMAGENT_PROVIDER_MODE=gemini IMAGENT_VECTOR_STORE_MODE=qdrant python -m imagent.check
```

`python -m imagent.check` gasta **tres llamadas** y te dice si tus claves, tus
modelos y tu Qdrant funcionan. Comprueba contratos, no codigos de estado: que el
embedding tenga la dimension configurada y norma 1, que la salida estructurada se
valide contra el esquema, y que el modelo de vision **vea de verdad** —le enseña
un PNG rojo generado al vuelo y le pregunta el color—. Cada proveedor se
comprueba solo si es el real, asi que las combinaciones mixtas (Gemini contra
memoria, o fakes contra Qdrant) tambien dicen algo util.

---

## El problema

Las descripciones se generan **una vez**, al subir la imagen. Cuando alguien
pregunta *"¿en alguna hay algo escrito a mano?"*, la búsqueda no encuentra nada:
en la ingesta nadie se fijó en la caligrafía.

Ahí el coordinador tiene que elegir entre volver a mirar N imágenes —lo caro— o
responder que no lo sabe. **Esa decisión está en el código, no en el prompt**, y
está acotada por un presupuesto:

| Límite | Por defecto | Qué acota |
|---|---|---|
| `max_graph_iterations` | 6 | vueltas del grafo antes de cortar |
| `max_vision_images` | 3 | imágenes que se pueden volver a mirar por consulta |
| `max_vision_seconds` | 45 | tiempo total de visión por consulta |

Cuando un límite se agota, el sistema **no falla**: responde con lo que tiene y
lo marca como incompleto, con el motivo. Hay siete motivos distintos, y salen
tanto en los logs como en la respuesta HTTP.

## La arquitectura

```
                            START
                              │
                              ▼
        ┌──────────────► coordinator ◄──────────────┐
        │              (decide, no habla)           │
        │            ╱        │        ╲            │
        │   "retrieve"    "vision"   "respond"      │
        │          ╱          │          ╲          │
        │         ▼           ▼           ▼         │
        └──── retrieve     vision      respond      │
                 │            │            │        │
                 └────────────┴────────────┘        │
                              └─────────────────────┘
                                                   END
```

Topología de radios: **una sola arista condicional**, la que sale del
coordinador. `retrieve` y `vision` siempre vuelven a él y nunca hablan entre
ellos, así que hay un único sitio donde se decide y un único sitio donde se
aplica el presupuesto.

| Agente | Qué hace | Coste | Contrato de fallo |
|---|---|---|---|
| **coordinador** | decide el plan; única voz hacia el usuario | 1 llamada de texto por vuelta | degrada a un plan de reserva determinista |
| **recuperación** | búsqueda semántica sobre lo indexado; **nunca mira una imagen** | 1 embedding + 1 búsqueda | degrada: se responde sin evidencia |
| **visión** | vuelve a mirar imágenes concretas | N llamadas multimodales | degrada **por imagen**: que falle la segunda de tres no tira las otras dos |

La **ingesta no es un agente**: es un pipeline lineal sin ramificaciones. Que
llame al mismo proveedor de visión no la convierte en uno.

## Decisiones que merece la pena mirar

**El orden de escritura de la ingesta.** Se escriben los puntos en el almacén y
*después* se marca el registro como indexado. Al revés tendrías un registro que
afirma estar indexado sin puntos detrás: un hueco que ninguna consulta revela.
El id de cada punto es `uuid5(imagen, aspecto)`, determinista, así que un
reintento sobrescribe en vez de duplicar. Está fijado en
[un test de secuencia](tests/unit/test_ingestion.py), no de resultado: el estado
final es idéntico en ambos órdenes, y solo se nota cuando algo se rompe a mitad.

**Un punto de Qdrant por (imagen, aspecto).** Descripción, texto visible y
objetos se indexan por separado. Con un solo punto por imagen, una descripción
de 80 palabras diluye un OCR de 4, y *"¿qué pone en el cartel?"* recupera mal.

**Cuatro estados por imagen**, no dos. `ANALYZED` existe porque es el punto donde
ya has pagado la llamada cara: si el indexado falla después, el análisis sigue
guardado y reintentar es gratis.

**`route()` es una función pura** de tres argumentos. No toca el estado del
grafo, no llama a ningún modelo y no importa LangGraph. Los 20 tests de enrutado
corren sin construir un grafo y sin generar una palabra de prosa.

**El coordinador propone, el código dispone.** El modelo devuelve una acción de
un enum cerrado y una lista de imágenes candidatas. El código valida los ids
contra la evidencia real (un modelo se inventa un UUID con toda la seguridad del
mundo), recorta la lista al presupuesto, y puede ignorar la acción entera.

**Solo `messages` lleva reducer.** La frontera entre lo que persiste y lo que se
reinicia cada turno se lee en el tipo del estado. Escribir `[]` en un canal
acumulativo de LangGraph no lo vacía, así que reiniciar la evidencia con
reducers exigiría un centinela que además tendría que sobrevivir a la
serialización del checkpointer.

## Fiabilidad

**Dos frenos, no uno.** El presupuesto de iteraciones corta con elegancia y
devuelve una respuesta marcada. El `recursion_limit` de LangGraph lanza una
excepción y es el cinturón de seguridad: si salta, es un bug de la contabilidad.
Con solo el segundo, la única forma de parar sería una excepción, y una excepción
no puede devolver *"esto es lo que encontré, pero incompleto"*.

**Dos mecanismos de fallo con nombres distintos**, porque son dos contratos
distintos:

- `degrade_on(...)` se traga el error declarado y sigue. Es la **única** forma
  legítima de que un error deje de propagarse, y **siempre** emite un WARNING con
  nombre de evento estable. Lo que no está en la lista propaga: un bug tuyo nunca
  se degrada.
- `_fase(...)` en la ingesta deja constancia y **propaga**. El fallo sale a la
  luz y además queda diagnosticable en el registro, con la fase donde murió.

Ninguno captura `CancelledError`: un cancelado no es una degradación.

**El camino degradado es audible en tres sitios a la vez**: un evento en el
estado del grafo, un log con `x-request-id` correlacionado, y un array `warnings`
en la respuesta HTTP. Un fallo que solo está en el log es silencioso para todo el
que no tenga acceso al log.

**Timeouts distintos según quién espera**: recuperación 5 s (el usuario está
mirando la pantalla), visión bajo demanda 25 s por imagen, visión de ingesta 60 s.

## El patrón de proveedores

Todo lo que sale del proceso está detrás de un `Protocol`: visión, texto,
embeddings, almacén de vectores, repositorio y almacén de bytes. Cada uno tiene
una implementación real y una determinista offline.

**El CI no puede salir a la red aunque quisiera.** `google-genai` y
`qdrant-client` están en extras opcionales que el CI no instala, y hay un paso
que falla si alguien los mueve a las dependencias base. No es que los tests
eviten la red: es que no hay con qué llegar a ella.

Aun así, los adaptadores reales importan sus SDK **dentro de los métodos**, así
que el test de conformidad comprueba en CI que cumplen los Protocol aunque el SDK
no exista. Sin eso, la forma de los proveedores reales —justo los que ningún otro
test cubre— solo se verificaría en producción.

Y donde hay dos implementaciones que sí corren en CI (repositorio en memoria vs
SQLite, bytes en memoria vs disco), **los mismos tests de contrato corren contra
las dos**. Un fake que se comporta distinto que la implementación real no protege
de nada. Por el mismo motivo, el almacén de vectores en memoria **replica los
rechazos de Qdrant**: buscar sin colección y escribir un vector de dimensión
equivocada fallan igual en los dos.

---

## Limitaciones

Esto no es una lista de "trabajo futuro". Son cosas que el proyecto **no hace
bien** y que conviene saber antes de mirarlo.

**El modo offline no entiende nada.** Los embeddings fake son una bolsa de
palabras hasheada: parecido **léxico**, no semántico. `"coche"` y `"automóvil"`
dan coseno exactamente 0, y hay
[un test que lo deja escrito](tests/unit/test_fake_embeddings.py). Sirve para
verificar el cableado del sistema, no la calidad de la recuperación. El
planificador offline decide con una lista de palabras clave.

**El catálogo no escala.** Al coordinador se le enumeran las imágenes indexadas
(hasta `coordinator_catalogue_limit`, 50 por defecto) para que pueda señalar una
que la búsqueda no devolvió — sin eso, la escalada a visión sería imposible en el
caso que más importa. Con miles de imágenes hay que cambiarlo por una
preselección.

**El filtrado por metadatos no está expuesto.** El repositorio sabe filtrar por
estado y por rango de fechas, pero el coordinador no tiene forma de pedirlo:
*"¿en las de ayer hay algún coche?"* se resuelve como una búsqueda semántica
normal.

**Los dos almacenes pueden divergir.** Un registro marcado como indexado cuyos
puntos ya no están en Qdrant se detecta al recuperar y se salta con un evento
`retrieval.orphan_points`, pero no hay ninguna reconciliación que lo arregle.

**La ingesta es síncrona.** Subir diez imágenes son diez peticiones que esperan
al modelo de visión. La máquina de estados ya está modelada para pasarlo a
segundo plano, pero no está hecho.

**Un turno puede tardar mucho.** Con el presupuesto por defecto, una consulta que
escale a visión puede llegar a 45 s. No hay streaming ni respuestas parciales.

**Sin autenticación y sin multiusuario.** Cualquiera que llegue al puerto ve
todas las imágenes. Los `thread_id` no están protegidos: quien adivine uno lee
esa conversación.

**Los hallazgos de visión no se re-indexan.** Si vuelves a preguntar por la
caligrafía, se vuelve a pagar la mirada. Escribirlos de vuelta al índice haría
que la segunda vez fuese barata, a cambio de meter salida de un modelo en el
índice.

**Sin verificador de fundamentación.** Nada comprueba que la respuesta redactada
se apoye de verdad en la evidencia recuperada.

**Qdrant no se ha ejecutado contra un servidor real.** El adaptador de Gemini
si: `imagent.check` y una prueba de punta a punta pasaron el 2 de septiembre de
2026. El de Qdrant solo tiene los tests que verifican la llamada que construye.

**Los modelos que razonan son lentos.** Contra `gemini-3.6-flash`, una decision
del coordinador con el catalogo y la evidencia delante puede pasar de 15 s, y un
turno con tres vueltas se va de un minuto. Los timeouts por defecto (45 s) estan
puestos para eso. La palanca para mejorarlo no es bajarlos —eso solo convierte
lentitud en degradacion— sino `thinking_config` en el adaptador de Gemini, que
esta sin probar por falta de cuota.

**Los nombres de modelo caducan.** `gemini-2.5-flash` dejo de estar disponible
para cuentas nuevas y hubo que cambiarlo. Por eso son configuracion y no
constantes, y por eso existe `imagent.check`.

**No hay type checker.** `Protocol` es tipado estructural que verifica un
verificador estático, y aquí no se ejecuta mypy. Lo sustituye un test de
conformidad que compara firmas y corrutinas en tiempo de ejecución, que es menos
que mypy.

**El front es lo mínimo.** HTML y JS sin dependencias, sin build y sin estados de
carga más allá de deshabilitar el botón.

---

## Estructura

```
src/imagent/
├── config.py           configuración; único sitio que lee el entorno
├── demo.py             escenario de ejemplo, usado por tests y por el modo fake
├── domain/             modelos y errores; sin FastAPI, sin LangGraph, sin SDK
├── providers/          Protocols + implementaciones reales y offline
│   ├── base.py         los seis contratos con el exterior
│   ├── fake/           deterministas, sin red
│   ├── real/           gemini, qdrant, sqlite
│   └── registry.py     único módulo que sabe si estamos en modo real
├── services/           ingesta e indexado: lineales, sin agentes
├── agents/             el grafo
│   ├── budget.py       el presupuesto
│   ├── routing.py      route(): función pura, la pieza central
│   ├── state.py        el estado y la frontera persiste/reinicia
│   ├── coordinator.py  retrieval.py  vision.py  responder.py
│   └── graph.py        StateGraph, checkpointer y un turno de conversación
├── api/                FastAPI
└── web/                front mínimo
```

72 ficheros Python, ~8600 líneas contando tests. 317 tests, `ruff` limpio.

## Configuración

Variables con prefijo `IMAGENT_`; las secciones anidadas con doble guion bajo
(`IMAGENT_BUDGET__MAX_VISION_IMAGES=1`). Ver [.env.example](.env.example).

Tres ejes independientes, porque las combinaciones útiles no las expresa un solo
flag: `PROVIDER_MODE` (`fake`|`gemini`), `VECTOR_STORE_MODE` (`memory`|`qdrant`)
y `REPOSITORY_MODE` (`memory`|`sqlite`). Querrás Gemini real contra un almacén en
memoria mientras desarrollas.

Una variable `IMAGENT_*` mal escrita **revienta al arrancar**. `extra="forbid"`
de pydantic-settings solo cubre el fichero `.env`, no las variables del proceso,
así que hay una guardia explícita que enumera el entorno.

Los modelos de Gemini se configuran (`IMAGENT_VISION_MODEL`, `IMAGENT_TEXT_MODEL`,
`IMAGENT_EMBEDDING_MODEL`) y los valores por defecto pueden haber quedado
desactualizados; conviene comprobarlos contra la documentación vigente.
