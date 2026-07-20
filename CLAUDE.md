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
- Rigid bodies (src/sim/body.js): generic CPU rigid bodies coupled two-way,
  one shape per entry in the `BODY_TYPES` table (src/sim/bodytypes.js: hull
  circles, sprite PNG, `size`, `densityRel`, `hullArea`, `inertiaFactor` —
  `densityRel` alone decides float vs sink, e.g. duck 0.45, brick 1.9). Adding
  a shape = one table entry + one painter in tools/make-sprites.mjs; `RigidBody`
  and both renderers are fully generic over it (nothing else is per-type).
  Owned by `BodyFlock`, which publishes the aggregate `params.bodyCircles`
  (hulls of all bodies) and `params.bodyPoses` (`{x,y,angle,scale,type}` — the
  renderers read `type` to pick the sprite texture), and resolves body-body
  contact on the CPU (`BodyFlock.collide()`, run once per substep before
  stepping the bodies; it accumulates into each body's `extFx/extFy/extTq`,
  which `RigidBody.step` consumes and zeroes; reduced mass, so a brick shoves a
  duck not vice versa). Solvers append `params.bodyCircles` to the obstacle
  upload (never push them into `params.obstacles`: renderers and the eraser read
  that), and `solver.sampleBodyForce(circles, cx, cy, slot)` reduces the spring
  reaction over all particles on the GPU, once per body per frame — sync 1×1
  readback on WebGL2, async fixed-point atomics on WebGPU (a frame stale; each
  body needs its own `slot` so the in-flight readbacks don't collide, and the
  ?t= fast-forward awaits `bodyReadPromise` then reads `bodyForceAt(slot)`).
  Sprites: transparent PNGs in assets/, generated analytically in the body's
  local frame by `tools/make-sprites.mjs` (npm run sprites; the PNGs are
  committed) so artwork lines up with the hull by construction; each renderer
  draws one textured quad over the type's local-unit `rect`. Interaction: the
  `Object` dropdown sets `params.bodyType` (what Add spawns; `?obj=` presets
  it); Add/Remove are pointer tools like the obstacle Draw/Erase ones —
  `params.addBodies`/`removeBodies`/`drawObstacles`/`eraseObstacles` are
  mutually exclusive (one `tools` table in controls.js); add/remove act on
  pointerdown only, so a drag can't spew bodies. Drag with no tool active grabs
  and throws a body, arrow keys set `params.bodyThrust` for all of them; one
  URL param per type spawns up front (`?duck=n`, `?brick=n`). Wiring +
  collision are tested by `node tools/test-body-input.mjs`.
