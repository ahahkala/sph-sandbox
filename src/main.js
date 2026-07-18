// App bootstrap and main loop. All physics runs on the GPU; the fixed
// timestep + accumulator keeps sim speed identical across display refresh
// rates. Two physics modes share the WebGL2 pipeline (force SPH and PBF);
// a WebGPU backend (?backend=webgpu) scales to 1M+ particles.

import { createParams, WORLD_HEIGHT, REF_COUNT } from "./config.js";
import { GPUSolver } from "./sim/gpu/gpusolver.js";
import { spawners } from "./sim/spawn.js";
import { Duck } from "./sim/duck.js";
import { Renderer } from "./render/renderer.js";
import { setupControls } from "./ui/controls.js";
import { setupPointer } from "./ui/pointer.js";
import { Hud } from "./ui/hud.js";

const canvas = document.getElementById("glcanvas");
const params = createParams();
const world = { w: WORLD_HEIGHT, h: WORLD_HEIGHT }; // corrected in resize()
const query = new URLSearchParams(location.search);

const backend = query.get("backend") === "webgpu" ? "webgpu" : "webgl";

function fatal(message) {
  document.body.innerHTML = "<p style='color:#fff;padding:20px'>" + message + "</p>";
}

async function boot() {
  let solver, renderer;
  if (backend === "webgpu") {
    const { startWebGPU } = await import("./webgpu/app.js");
    try {
      ({ solver, renderer } = await startWebGPU(canvas, world, params));
    } catch (err) {
      fatal("WebGPU backend failed: " + err.message +
        " — <a style='color:#8fb4ff' href='" + location.pathname + "'>use the WebGL2 backend</a>");
      throw err;
    }
  } else {
    try {
      renderer = new Renderer(canvas, params);
      solver = new GPUSolver(renderer.gl, world, params);
    } catch (err) {
      fatal("WebGL2 with float render targets is required: " + err.message);
      throw err;
    }
  }
  run(solver, renderer);
}

function run(solver, renderer) {
  const hud = new Hud(document.getElementById("hud"));

  function resize() {
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = window.innerWidth, h = window.innerHeight;
    world.h = WORLD_HEIGHT * Math.sqrt(params.count / REF_COUNT);
    world.w = world.h * (w / h);
    solver.setWorld(world.w, world.h);
    renderer.resize(Math.floor(w * dpr), Math.floor(h * dpr));
  }
  window.addEventListener("resize", resize);

  let paused = false;

  // rubber duck rigid body: hull circles couple into the solvers' obstacle
  // path, the fluid reaction force is sampled from the GPU once per frame
  const duck = new Duck(params);
  let duckForce = null;

  function respawn(presetName) {
    if (params.duck) {
      duck.reset(world);
      duckForce = null;
    }
    const spawn = spawners[presetName] || spawners.damBreak;
    const n = solver.count;
    const state = {
      px: new Float32Array(n), py: new Float32Array(n),
      vx: new Float32Array(n), vy: new Float32Array(n),
      count: n, world, h: params.h,
    };
    spawn(state);
    solver.seed(state);
    hud.setCount(n, backend, params.physics);
  }

  const controls = setupControls(params, {
    backend,
    maxCount: solver.maxCount,
    onReset: (preset) => respawn(preset),
    onCountChange: () => {
      params.obstacles.length = 0; // world rescales; stored positions invalid
      resize();
      solver.alloc(params.count);
      respawn(controls.currentPreset());
    },
    onPhysicsChange: () => {
      hud.setCount(solver.count, backend, params.physics);
    },
    onPauseToggle: () => (paused = !paused),
    onDuckToggle: () => {
      if (params.duck) duck.reset(world);
      else duck.remove();
      duckForce = null;
    },
  });
  setupPointer(canvas, solver, params, duck);

  // scene/physics/count/mode/duck/gravity/etc. are already restored from
  // the URL inside setupControls; ?t= (fast-forward) and ?ob= (obstacles,
  // fractions of world size) are one-shot and applied below/after resize.
  resize();
  solver.alloc(params.count);
  respawn(controls.currentPreset());

  // ?ob=fx,fy,fr;… adds obstacles as fractions of world size (after resize
  // so the world dimensions are final)
  if (query.get("ob")) {
    for (const group of query.get("ob").split(";")) {
      const [fx, fy, fr] = group.split(",").map(parseFloat);
      if (fr > 0) {
        params.obstacles.push({ x: fx * world.w, y: fy * world.h, r: fr * world.h });
      }
    }
  }

  // ?t= is wall-clock viewing seconds: scaled like the live loop so the
  // same t shows the same stage of the flow at every count
  async function fastForwardSim() {
    const fastForward = parseFloat(query.get("t"));
    if (!(fastForward > 0)) return;
    const simSeconds = fastForward * (world.h / WORLD_HEIGHT);
    const steps = Math.min(Math.round(simSeconds / params.dt), 20000);
    for (let i = 0; i < steps; i++) {
      if (params.duck && duck.pose) {
        // resample sparsely: WebGL2 reads back synchronously, and on
        // WebGPU the async readback must be awaited (the render loop
        // isn't turning, so a stale value would never resolve)
        if (i % 8 === 0) {
          duckForce = solver.sampleDuckForce(params.duckCircles, duck.pose.x, duck.pose.y);
          if (solver.duckReadPromise) {
            await solver.duckReadPromise;
            duckForce = solver.duckForce;
          }
        }
        duck.step(params.dt, world, duckForce);
      }
      solver.step(params.dt);
    }
  }

  // --- fixed-timestep loop ------------------------------------------------
  let accumulator = 0;
  let lastTime = performance.now();

  function frame(now) {
    const frameDt = Math.min((now - lastTime) / 1000, 1 / 30); // clamp stalls
    lastTime = now;

    let simMs = 0;
    if (!paused) {
      // world velocities are count-invariant but the world grows with
      // sqrt(count), so advance sim time faster in bigger worlds to keep
      // on-screen motion speed the same at every count
      accumulator += frameDt * (world.h / WORLD_HEIGHT);
      const t0 = performance.now();
      let steps = 0;
      while (accumulator >= params.dt && steps < params.maxSubsteps) {
        if (params.duck && duck.pose) duck.step(params.dt, world, duckForce);
        solver.step(params.dt);
        accumulator -= params.dt;
        steps++;
      }
      if (steps === params.maxSubsteps) accumulator = 0; // can't keep up: drop time
      // one reduction + readback per frame; the force is held constant
      // across next frame's substeps (sync on WebGL2, async on WebGPU)
      if (steps > 0 && params.duck && duck.pose) {
        duckForce = solver.sampleDuckForce(params.duckCircles, duck.pose.x, duck.pose.y);
      }
      simMs = performance.now() - t0;
    }

    renderer.render(solver.posVelTexture(), solver.count, world);
    hud.frame(simMs);
    window.__frames = (window.__frames || 0) + 1; // headless tooling waits on this
    window.__params = params;                     // headless tooling inspects these
    window.__duck = duck;                         // (tools/test-duck-input.mjs)
    requestAnimationFrame(frame);
  }
  fastForwardSim().then(() => {
    lastTime = performance.now();
    requestAnimationFrame(frame);
  });
}

boot();
