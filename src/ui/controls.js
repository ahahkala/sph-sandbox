// Wires the control panel to the live params object and app callbacks.
// callbacks: { backend, maxCount, onRestart(preset), onCountChange(),
//              onPhysicsChange(), onPauseToggle(), onDucksClear() }

import { DT_FORCE, DT_PBF } from "../config.js";

export const COUNTS = [
  1000, 3000, 6000, 12000, 25000, 50000, 100000, 175000, 262144,
  524288, 1048576,
];

const fmtCount = (n) =>
  n >= 1000000 ? (n / 1048576).toFixed(0) + "M" :
  n >= 10000 ? Math.round(n / 1000) + "k" : String(n);

export function setupControls(params, callbacks) {
  const $ = (id) => document.getElementById(id);
  const query = new URLSearchParams(location.search);

  // accordion: click the header bar to collapse/expand the control panel
  $("ui-header").addEventListener("click", () => {
    $("ui").classList.toggle("collapsed");
  });

  // shared serialization of every persisted control, used both to keep the
  // address bar in sync as controls change and to build the URL a backend
  // switch reloads into (a canvas can't change context type live). Declared
  // as a function so it hoists — it's only ever called after the elements
  // and state it closes over (presetEl, pauseBtn, params, …) are set up.
  function buildQuery() {
    const q = new URLSearchParams();
    if (callbacks.backend === "webgpu") q.set("backend", "webgpu");
    q.set("scene", presetEl.value);
    q.set("phys", params.physics);
    q.set("mode", params.renderMode);
    q.set("n", params.count);
    q.set("gravity", params.gravity);
    q.set("gravdir", params.gravityCenter ? "center" : "down");
    q.set("viscosity", params.viscosity);
    q.set("stiffness", params.stiffness);
    q.set("nearstiff", params.nearStiffness);
    q.set("vorticity", params.vorticity);
    q.set("pointer", params.pointerStrength);
    q.set("obstsize", params.obstacleRadius);
    q.set("paused", pauseBtn.classList.contains("active") ? "1" : "0");
    return q;
  }
  const syncURL = () => history.replaceState(null, "", "?" + buildQuery().toString());

  // slider values are overridden from the URL before the sliders bind, so
  // bindSlider's `el.value = params[key]` picks up the shared/restored value
  const numOverride = (queryKey, paramsKey) => {
    const v = parseFloat(query.get(queryKey));
    if (Number.isFinite(v)) params[paramsKey] = v;
  };
  numOverride("gravity", "gravity");
  numOverride("viscosity", "viscosity");
  numOverride("stiffness", "stiffness");
  numOverride("nearstiff", "nearStiffness");
  numOverride("vorticity", "vorticity");
  numOverride("pointer", "pointerStrength");
  numOverride("obstsize", "obstacleRadius");
  if (query.has("gravdir")) params.gravityCenter = query.get("gravdir") === "center";

  const bindSlider = (id, key) => {
    const el = $(id);
    const out = $(id + "-val");
    const update = () => {
      params[key] = parseFloat(el.value);
      out.textContent = el.value;
    };
    el.addEventListener("input", update);
    el.value = params[key];
    update();
  };

  bindSlider("gravity", "gravity");

  // gravity direction: straight down or toward the world center
  const gravButtons = { down: $("grav-down"), center: $("grav-center") };
  const setGravDir = (center) => {
    params.gravityCenter = center;
    gravButtons.down.classList.toggle("active", !center);
    gravButtons.center.classList.toggle("active", center);
  };
  gravButtons.down.addEventListener("click", () => setGravDir(false));
  gravButtons.center.addEventListener("click", () => setGravDir(true));
  setGravDir(params.gravityCenter);

  bindSlider("viscosity", "viscosity");
  bindSlider("stiffness", "stiffness");
  bindSlider("nearstiff", "nearStiffness");
  bindSlider("vorticity", "vorticity");
  bindSlider("pointer", "pointerStrength");
  bindSlider("obstsize", "obstacleRadius");

  // particle count: slider indexes into the COUNTS table (log-ish scale)
  const counts = COUNTS.filter((c) => c <= (callbacks.maxCount || 262144));
  const countEl = $("count");
  const countOut = $("count-val");
  countEl.min = 0;
  countEl.max = counts.length - 1;
  countEl.step = 1;
  const applyCount = (n) => {
    let idx = counts.findIndex((c) => c >= n);
    if (idx < 0) idx = counts.length - 1;
    countEl.value = idx;
    params.count = counts[idx];
    countOut.textContent = fmtCount(params.count);
  };
  countEl.addEventListener("input", () => {
    params.count = counts[countEl.value];
    countOut.textContent = fmtCount(params.count);
  });
  countEl.addEventListener("change", () => callbacks.onCountChange());

  // backend switch: needs a page reload (a canvas is permanently bound to
  // its first context type); current settings ride along in the URL
  const backendButtons = { webgl: $("backend-webgl"), webgpu: $("backend-webgpu") };
  backendButtons[callbacks.backend]?.classList.add("active");
  if (!navigator.gpu) {
    backendButtons.webgpu.disabled = true;
    backendButtons.webgpu.title = "WebGPU unsupported in this browser";
  }
  for (const [name, btn] of Object.entries(backendButtons)) {
    btn.addEventListener("click", () => {
      if (name === callbacks.backend) return;
      const q = buildQuery();
      if (name === "webgpu") q.set("backend", "webgpu");
      else q.delete("backend");
      location.search = q.toString();
    });
  }

  // physics mode toggle (also switches the timestep)
  const physButtons = { force: $("phys-force"), pbf: $("phys-pbf") };
  const forceOnlySliders = ["stiffness", "nearstiff", "vorticity"];
  const setPhysics = (name, fire = true) => {
    if (name !== "force" && name !== "pbf") return;
    params.physics = name;
    params.dt = name === "pbf" ? DT_PBF : DT_FORCE;
    for (const [key, btn] of Object.entries(physButtons)) {
      btn.classList.toggle("active", key === name);
    }
    // PBF replaces pressure forces with constraint projection and has no
    // curl pass, so these sliders only act in force mode
    const pbf = name === "pbf";
    for (const id of forceOnlySliders) {
      const el = $(id);
      el.disabled = pbf;
      const row = el.closest(".row");
      row.classList.toggle("inert", pbf);
      row.title = pbf ? "force mode only" : "";
    }
    if (fire) callbacks.onPhysicsChange();
  };
  physButtons.force.addEventListener("click", () => setPhysics("force"));
  physButtons.pbf.addEventListener("click", () => setPhysics("pbf"));

  // render mode toggle
  const modeButtons = { liquid: $("mode-liquid"), dots: $("mode-dots") };
  const setMode = (mode) => {
    params.renderMode = mode;
    for (const [name, btn] of Object.entries(modeButtons)) {
      btn.classList.toggle("active", name === mode);
    }
  };
  modeButtons.liquid.addEventListener("click", () => setMode("liquid"));
  modeButtons.dots.addEventListener("click", () => setMode("dots"));
  setMode(params.renderMode);

  // pointer tools: obstacle draw/erase and duck add/remove. Toggling one on
  // turns the others off, so the pointer always has a single meaning; with
  // none active the pointer pushes and pulls fluid.
  const tools = {
    drawObstacles: $("obstacle-draw"),
    eraseObstacles: $("obstacle-erase"),
    addDucks: $("duck-add"),
    removeDucks: $("duck-remove"),
  };
  const drawBtn = tools.drawObstacles;
  const eraseBtn = tools.eraseObstacles;
  const syncTools = () => {
    let active = false;
    for (const [flag, btn] of Object.entries(tools)) {
      btn.classList.toggle("active", params[flag]);
      active = active || params[flag];
    }
    document.getElementById("glcanvas").style.cursor = active ? "cell" : "crosshair";
  };
  for (const [flag, btn] of Object.entries(tools)) {
    btn.addEventListener("click", () => {
      const on = !params[flag];
      for (const other of Object.keys(tools)) params[other] = false;
      params[flag] = on;
      syncTools();
    });
  }
  $("obstacle-clear").addEventListener("click", () => {
    params.obstacles.length = 0;
  });
  $("duck-clear").addEventListener("click", () => callbacks.onDucksClear());
  syncTools();

  // scene preset
  const presetEl = $("preset");
  presetEl.addEventListener("change", () => callbacks.onRestart(presetEl.value));

  // pause / restart / reset
  const pauseBtn = $("pause");
  const syncPause = (paused) => {
    pauseBtn.classList.toggle("active", paused);
    pauseBtn.textContent = paused ? "Resume" : "Pause";
  };
  pauseBtn.addEventListener("click", () => syncPause(callbacks.onPauseToggle()));
  // Restart re-seeds the current scene, keeping every setting
  $("restart").addEventListener("click", () => callbacks.onRestart(presetEl.value));
  // Reset goes back to the factory settings. Every default lives in
  // createParams() and in the controls' own initialisation, and the URL is
  // rewritten from params on each change, so reloading without a query string
  // is what actually restores them all — patching params in place would leave
  // the sliders, tools and solver allocation to be re-synced by hand.
  $("reset").addEventListener("click", () => {
    location.href = location.pathname;
  });

  // arrow keys apply a steering force to every duck while held
  const arrows = {
    ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1],
  };
  const held = new Set();
  const syncThrust = () => {
    params.duckThrust.x = 0;
    params.duckThrust.y = 0;
    for (const code of held) {
      params.duckThrust.x += arrows[code][0];
      params.duckThrust.y += arrows[code][1];
    }
  };

  window.addEventListener("keydown", (e) => {
    if (e.code === "Space") {
      e.preventDefault();
      syncPause(callbacks.onPauseToggle());
      syncURL();
    } else if (e.code === "KeyR") {
      callbacks.onRestart(presetEl.value);
    } else if (e.code === "KeyO") {
      drawBtn.click();
    } else if (e.code === "KeyE") {
      eraseBtn.click();
    } else if (arrows[e.code]) {
      e.preventDefault(); // keep the page/sliders from scrolling
      held.add(e.code);
      syncThrust();
    }
  });
  window.addEventListener("keyup", (e) => {
    if (arrows[e.code]) {
      held.delete(e.code);
      syncThrust();
    }
  });
  window.addEventListener("blur", () => {
    held.clear();
    syncThrust();
  });

  // restore scene/physics/count/mode/paused from the URL — the
  // numeric sliders and gravity direction were already restored above,
  // before their controls bound to params
  if (query.has("scene") && [...presetEl.options].some((o) => o.value === query.get("scene"))) {
    presetEl.value = query.get("scene");
  }
  setPhysics(query.get("phys") || params.physics, false);
  applyCount(query.get("n") ? parseInt(query.get("n"), 10) : params.count);
  if (query.get("mode") === "dots" || query.get("mode") === "liquid") setMode(query.get("mode"));
  if (query.get("paused") === "1") syncPause(callbacks.onPauseToggle());

  // keep the address bar in sync with every control change so the current
  // state is always shareable/reloadable; delegated on the panel so it
  // fires after the control's own handler has already updated params.
  // A page opened without a query string keeps a clean URL until something
  // is actually changed — that's what makes Reset's clearing of the query
  // stick instead of being overwritten with the defaults right away.
  if (query.toString()) syncURL();
  $("ui-content").addEventListener("input", syncURL);
  $("ui-content").addEventListener("change", syncURL);
  $("ui-content").addEventListener("click", syncURL);

  return {
    currentPreset: () => presetEl.value,
  };
}
