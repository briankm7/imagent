// Front minimo a proposito: HTML y JS sin dependencias ni build.
//
// Lo unico que guarda entre mensajes es el thread_id, que es la unica clave del
// estado de la conversacion: el historial, el foco y la evidencia viven en el
// servidor, en el checkpointer del grafo. Si el navegador se recarga, basta con
// mandar el mismo thread_id para seguir donde se estaba.

const galeria = document.getElementById("galeria");
const mensajes = document.getElementById("mensajes");
const formulario = document.getElementById("formulario");
const entrada = document.getElementById("entrada");
const soltar = document.getElementById("soltar");
const fichero = document.getElementById("fichero");
const boton = formulario.querySelector("button");

let threadId = null;

async function pedir(url, opciones) {
  const respuesta = await fetch(url, opciones);
  const cuerpo = await respuesta.json().catch(() => ({}));
  if (!respuesta.ok) {
    throw new Error(cuerpo.detail || cuerpo.error || `error ${respuesta.status}`);
  }
  return cuerpo;
}

async function cargarEstado() {
  const salud = await pedir("/health");
  document.getElementById("modo").textContent =
    `${salud.provider_mode} · ${salud.vector_store_mode} · ${salud.images_indexed} indexadas`;
  await cargarGaleria();
}

async function cargarGaleria() {
  const imagenes = await pedir("/api/images");
  galeria.replaceChildren();

  for (const imagen of imagenes) {
    const li = document.createElement("li");
    if (imagen.status === "failed") li.classList.add("fallida");

    const miniatura = document.createElement("img");
    miniatura.src = `/api/images/${imagen.id}/bytes`;
    miniatura.alt = "";

    const nombre = document.createElement("div");
    nombre.className = "nombre";
    nombre.textContent = imagen.filename;

    const descripcion = document.createElement("div");
    descripcion.className = "desc";
    // Una imagen que fallo se ve, y se ve por que. Ocultarla dejaria al usuario
    // con un hueco invisible en su coleccion.
    descripcion.textContent =
      imagen.status === "failed"
        ? `fallo en ${imagen.failed_stage}: ${imagen.failure_reason}`
        : imagen.description;

    const texto = document.createElement("div");
    texto.append(nombre, descripcion);
    li.append(miniatura, texto);
    galeria.appendChild(li);
  }
}

async function subir(ficheros) {
  for (const f of ficheros) {
    const datos = new FormData();
    datos.append("file", f);
    try {
      const salida = await pedir("/api/images", { method: "POST", body: datos });
      if (salida.deduplicated) {
        agregarMensaje("agente", `${f.name} ya estaba subida; no se ha vuelto a analizar.`);
      }
    } catch (error) {
      agregarMensaje("agente", `No se pudo subir ${f.name}: ${error.message}`, { error: true });
    }
  }
  await cargarEstado();
}

function agregarMensaje(quien, texto, extra = {}) {
  const div = document.createElement("div");
  div.className = `msg ${quien}${extra.error ? " error" : ""}`;
  div.textContent = texto;

  // Lo incompleto y lo degradado se ENSEÑAN. Un sistema que responde peor sin
  // decirlo es peor que uno que falla: quien lee la respuesta no tiene forma de
  // saber que le falta algo.
  if (extra.incompleteReason) {
    const pie = document.createElement("div");
    pie.className = "incompleta";
    pie.textContent = `Respuesta incompleta (${extra.incompleteReason})`;
    div.appendChild(pie);
  }
  if (extra.warnings && extra.warnings.length) {
    const avisos = document.createElement("div");
    avisos.className = "avisos";
    avisos.textContent = extra.warnings.map((w) => `⚠ ${w.event}: ${w.detail}`).join("\n");
    div.appendChild(avisos);
  }

  mensajes.appendChild(div);
  mensajes.scrollTop = mensajes.scrollHeight;
}

formulario.addEventListener("submit", async (evento) => {
  evento.preventDefault();
  const texto = entrada.value.trim();
  if (!texto) return;

  agregarMensaje("yo", texto);
  entrada.value = "";
  boton.disabled = true;

  try {
    const salida = await pedir("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ message: texto, thread_id: threadId }),
    });
    threadId = salida.thread_id;
    agregarMensaje("agente", salida.answer, {
      incompleteReason: salida.incomplete ? salida.incomplete_reason : null,
      warnings: salida.warnings,
    });
  } catch (error) {
    agregarMensaje("agente", error.message, { error: true });
  } finally {
    boton.disabled = false;
    entrada.focus();
  }
});

document.getElementById("examinar").addEventListener("click", () => fichero.click());
fichero.addEventListener("change", () => subir(fichero.files));

for (const nombre of ["dragenter", "dragover", "dragleave", "drop"]) {
  soltar.addEventListener(nombre, (e) => {
    e.preventDefault();
    soltar.classList.toggle("encima", nombre === "dragenter" || nombre === "dragover");
  });
}
soltar.addEventListener("drop", (e) => subir(e.dataTransfer.files));

cargarEstado().catch((error) => agregarMensaje("agente", error.message, { error: true }));
