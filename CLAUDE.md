# sph-sim

2D SPH fluid sim, GPU-only physics: WebGL2 backend (262k particles, stencil
grid) and WebGPU backend (compute shaders, exact grid, 1M+). Both implement
force SPH + PBF + obstacles + vorticity. ES modules, no build step,
no runtime deps. See README.md for architecture and the feature matrix.

- Run: `npm run dev` → http://localhost:8123 (ES modules need a server, not file://)
- Test WebGL2 solver: `npm test` — run after touching src/sim/gpu/.
- Test WebGPU solver: `node tools/test-headless.mjs test/webgpu.html` — run
  after touching src/webgpu/. Prints SKIPPED where WebGPU is unavailable.
- Screenshot validation: `node tools/shot.mjs "?n=25000&t=1.5"` →
  shots/last.png. Always use `?t=` to fast-forward (rAF-based waiting is
  unreliable headless); `sw` as a trailing arg forces SwiftShader.
- Headless tooling uses puppeteer-core driving installed Chrome; test pages
  signal completion via document.title (GPU-PASS/GPU-FAIL/GPU-SKIP).
- WebGPU debugging: errors are async — the solver installs
  device.onuncapturederror and the test harness echoes page console errors.
  Mind WGSL uniform struct sizes (buffers must match padded struct size).
- The WebGL2 and WebGPU solvers (force SPH and PBF) implement identical
  math and MUST stay in sync — both test suites check the same invariants
  (the 2-particle repulsion gap 0.800 → 1.193 must match across backends).
  Mind WGSL reserved words when porting GLSL (e.g. `meta`).
- Scaling invariant: world height = 70·sqrt(count/3000) and gravity divides
  by the same factor (config.js). Physics constants are only valid with this
  coupling; never scale h or dt with count. On-screen speed is kept
  count-invariant by advancing sim time worldScale× real time (accumulator
  in main.js and ?t= are both scaled) — so substeps per frame
  grow with sqrt(count) (~4·scale in force mode; maxSubsteps must cover it).
- Force mode dt = 1/240, PBF dt = 1/60 (switched in controls.js). PBF
  iteration count is forced odd (texture ping-pong choreography).
- Boundaries (both modes, both backends): smooth spring-damper zone over the
  last h before walls/obstacles (`wallStiffness`/`wallDamping` in config.js);
  the hard clamp + velocity flip is only a fast-penetration safety net. Never
  make the clamp the primary contact — it causes edge jitter (the "resting
  pool stays calm" tests guard this). PBF applies the same spring as a
  per-iteration position nudge (boundaryAccel·dt² in deltaP) with damping
  halved — full wallDamping in position space measurably re-adds jitter.
- `restDensity` must match the spawn lattice density (~1.7 at spacing
  0.55·h); keep slider ranges in index.html in sync with src/config.js.
- Rubber ducks (src/sim/duck.js): CPU rigid bodies coupled two-way, owned by
  `DuckFlock`, which publishes the aggregate `params.duckCircles` (hulls of
  all ducks) and `params.duckPoses` (for the renderers), and resolves
  duck-duck contact on the CPU (`DuckFlock.collide()`, run once per substep
  before stepping the bodies; it accumulates into each duck's `extFx/extFy/
  extTq`, which `Duck.step` consumes and zeroes). Solvers append
  `params.duckCircles` to the obstacle upload (never push duck circles into
  `params.obstacles`: renderers and the eraser read that), and
  `solver.sampleDuckForce(circles, cx, cy, slot)` reduces the spring reaction
  over all particles on the GPU, once per duck per frame — sync 1×1 readback
  on WebGL2, async fixed-point atomics on WebGPU (a frame stale; each duck
  needs its own `slot` so the in-flight readbacks don't collide, and the ?t=
  fast-forward awaits `duckReadPromise` then reads `duckForceAt(slot)`).
  Interaction: the Add/Remove duck tools are pointer tools like the obstacle
  Draw/Erase ones — `params.addDucks`/`removeDucks`/`drawObstacles`/
  `eraseObstacles` are mutually exclusive (one `tools` table in controls.js
  enforces it); add/remove act on pointerdown only, so a drag can't spew
  ducks. Drag with no tool active grabs and throws a duck (spring in
  duck.step), arrow keys set `params.duckThrust` for all of them; `?duck=n`
  spawns n up front. Wiring is tested by `node tools/test-duck-input.mjs`.
