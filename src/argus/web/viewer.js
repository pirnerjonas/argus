/* The API owns dataset access and ordering; this file owns browser interaction. */
"use strict";
const $ = (id) => document.getElementById(id);
const state = { offset: 0, total: 0, items: [], request: 0, controller: null, resets: [] };

async function json(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error("Unable to load images. Check the terminal and try again.");
  return response.json();
}

function zoomable(viewport, img) {
  let zoom = 1, x = 0, y = 0, drag = null;
  const draw = () => { img.style.transform = `translate(${x}px, ${y}px) scale(${zoom})`; };
  const reset = () => { zoom = 1; x = 0; y = 0; draw(); };
  state.resets.push(reset);
  viewport.addEventListener("wheel", (event) => {
    event.preventDefault();
    zoom = Math.max(1, Math.min(10, zoom * (event.deltaY < 0 ? 1.2 : 1 / 1.2)));
    if (zoom === 1) { x = 0; y = 0; }
    draw();
  }, { passive: false });
  viewport.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    drag = { x: event.clientX - x, y: event.clientY - y };
    viewport.setPointerCapture(event.pointerId);
  });
  viewport.addEventListener("pointermove", (event) => {
    if (!drag || zoom === 1) return;
    x = event.clientX - drag.x; y = event.clientY - drag.y; draw();
  });
  for (const event of ["pointerup", "pointercancel", "lostpointercapture"]) {
    viewport.addEventListener(event, () => { drag = null; });
  }
}

function render() {
  $("viewer").replaceChildren();
  state.resets = [];
  for (const item of state.items) {
    const figure = document.createElement("figure");
    const viewport = document.createElement("div");
    viewport.className = "viewport";
    const caption = document.createElement("figcaption");
    caption.textContent = `${item.class_name ? item.class_name + " · " : ""}${item.filename}`;
    if (item.object_size !== null && item.object_size !== undefined) {
      caption.textContent += ` · largest box ${Math.round(item.object_size).toLocaleString()} px²`;
    }
    if (item.id !== null) {
      const img = document.createElement("img");
      img.alt = item.filename;
      img.draggable = false;
      img.src = `api/image/${item.id}?annotations=${$("annotations").checked ? 1 : 0}`;
      img.addEventListener("error", () => {
        const message = document.createElement("p");
        message.className = "error";
        message.textContent = "Image or annotations could not be loaded. You can continue to the next image.";
        viewport.replaceChildren(message);
      });
      viewport.append(img);
      zoomable(viewport, img);
    }
    figure.append(viewport, caption);
    $("viewer").append(figure);
  }
}

async function load() {
  const request = ++state.request;
  if (state.controller) state.controller.abort();
  state.controller = new AbortController();
  $("prev").disabled = $("next").disabled = true;
  $("status").textContent = $("sort").value === "object_size"
    ? "Loading… first object-size sort indexes annotation bounds and may take a moment."
    : "Loading…";
  const params = new URLSearchParams({
    search: $("search").value, sort: $("sort").value,
    descending: $("direction").value, offset: state.offset,
  });
  try {
    const data = await json(`api/images?${params}`, state.controller.signal);
    if (request !== state.request) return;
    state.total = data.total;
    state.items = data.total ? data.items : [];
    $("position").textContent = data.total ? `${state.offset + 1} / ${data.total}` : "0 / 0";
    $("status").textContent = data.total ? "" : "No images match your search.";
    render();
    $("prev").disabled = state.offset <= 0;
    $("next").disabled = state.offset + 1 >= state.total;
  } catch (error) {
    if (request !== state.request || error.name === "AbortError") return;
    $("status").textContent = error.message;
    $("position").textContent = "Unavailable";
    $("viewer").replaceChildren();
  }
}

function move(delta) {
  const button = delta < 0 ? $("prev") : $("next");
  if (button.disabled) return;
  state.offset += delta;
  load();
}
$("prev").onclick = () => move(-1);
$("next").onclick = () => move(1);
$("reset").onclick = () => state.resets.forEach((reset) => reset());
$("annotations").onchange = render;
$("filters").onsubmit = (event) => event.preventDefault();
let searchTimer;
$("search").oninput = () => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { state.offset = 0; load(); }, 200);
};
for (const id of ["sort", "direction"]) {
  $(id).onchange = () => { state.offset = 0; load(); };
}
document.addEventListener("keydown", (event) => {
  if (["INPUT", "SELECT", "TEXTAREA", "BUTTON"].includes(event.target.tagName)) return;
  const key = event.key.toLowerCase();
  if (["arrowright", "n"].includes(key)) { event.preventDefault(); move(1); }
  if (["arrowleft", "p"].includes(key)) { event.preventDefault(); move(-1); }
  if (key === "r") $("reset").click();
  if (key === "t" && !$("annotations").disabled) {
    $("annotations").checked = !$("annotations").checked; render();
  }
});
(async () => {
  try {
    const config = await json("api/config");
    document.title = $("title").textContent = config.title;
    $("viewer").classList.toggle("grid", config.classification);
    $("annotations").disabled = config.classification;
    if (config.sorts.includes("object_size")) {
      $("sort").add(new Option("Largest bounding box (px²)", "object_size"));
    }
    await load();
  } catch (error) { $("status").textContent = error.message; }
})();
