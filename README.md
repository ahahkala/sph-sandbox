# sph-sim

Interactive 2D SPH (smoothed particle hydrodynamics) fluid simulation in the
browser. All physics runs on the GPU:

- **WebGL2 backend** (default) — fragment-shader SPH up to 262k particles,
  with two physics modes (force SPH and PBF), vorticity confinement,
  drawable obstacles, and a rigid-body rubber duck floating on the fluid.
- **WebGPU backend** (`?backend=webgpu`) — compute-shader SPH with an exact
  neighbor grid (atomic counting sort + prefix sum), up to **1M+ particles**.

## Run

ES modules can't load over `file://`, so serve the directory:

```
npm run dev        # static server -> http://localhost:8123
```

Controls: drag to push fluid, right-drag to pull; a drag starting on the
duck grabs it (release mid-swing to throw), and the arrow keys steer it.
`o` toggles obstacle drawing (drag stamps circular walls, right-drag
erases), `e` toggles the eraser (drag removes walls near the cursor),
`space` pause, `r` reset.

URL parameters for reproducible states:
`?backend=webgpu&scene=drop&mode=dots&phys=pbf&n=100000&t=3&ob=0.5,0.8,0.06`
— `t` fast-forwards the sim synchronously; `ob` places obstacles as
`x,y,r` fractions of world size (`;`-separated); `duck=0` disables the duck.

## Test / validate

```
npm test                                        # WebGL2 physics suite
node tools/test-headless.mjs test/webgpu.html   # WebGPU physics suite
node tools/test-duck-input.mjs                  # duck keyboard/drag wiring
node tools/shot.mjs "?n=25000&t=1.5"            # screenshot -> shots/last.png
```

Both suites run in headless Chrome (via `puppeteer-core`, the only dev
dependency; the app itself has no dependencies). Test pages verify physics
invariants: pressure repulsion, dam-break settling, energy decay, bounds,
obstacle exclusion, PBF constraint relaxation — and print an ASCII density
map of the settled pool. `tools/shot.mjs` waits for the app to render
before capturing, so async WebGPU startup and `?t=` fast-forwards work.

## Architecture

```
index.html                    shell: canvas, control panel, styles
src/
  main.js                     bootstrap, backend pick, fixed-timestep loop
  config.js                   all tunables + shared constants
  sim/
    spawn.js                  initial conditions: damBreak, doubleDam, drop, sphere
    duck.js                   rubber duck: polygon mesh + rigid-body dynamics
    gpu/                      WebGL2 solver
      gpusolver.js            pipeline orchestration (force + PBF modes)
      shaders.js              force SPH: scatter, density+curl MRT, force
      pbfshaders.js           PBF: predict, lambda, deltaP, finalize
  webgpu/                     WebGPU backend
    app.js                    adapter/device bootstrap
    wgpusolver.js             compute pipeline (counting-sort grid)
    wgsl.js                   all WGSL (sim + render)
    wgpurenderer.js           instanced-quad renderer
  render/                     WebGL2 renderer
    renderer.js               liquid/dots modes, obstacle overlay
    shaders.js                GLSL: splat, composite, obstacles
    glutils.js                program/texture/target helpers
  ui/                         controls, pointer (incl. obstacle drawing), hud
tools/                        dev server + headless validation scripts
test/gpu.html                 WebGL2 physics checks
test/webgpu.html              WebGPU physics checks
```

### Feature matrix

|                        | WebGL2 | WebGPU |
|------------------------|--------|--------|
| Force SPH              | ✓      | ✓      |
| PBF (position-based)   | ✓      | ✓      |
| Vorticity confinement  | ✓      | ✓      |
| Obstacles              | ✓      | ✓      |
| Rubber duck (rigid body) | ✓    | ✓      |
| Max particles          | 262144 | 1048576 |
| Neighbor grid          | 8 slots/cell (stencil routing) | exact (counting sort) |

### How the physics works

**Force SPH** (dt = 1/240): Müller-style pressure + viscosity with a
Clavet-style near-pressure term (anti-clumping / surface tension), plus
vorticity confinement — per-particle 2D curl is computed alongside density
(MRT), and a force `ε (N̂ × ω)` sharpens vortices that numerical diffusion
would smear out.

**PBF** (dt = 1/60): predict positions, then (default 3) Jacobi iterations
of density-constraint projection (`lambda` / `deltaP` with the tensile
`s_corr` term), then velocities from position deltas + XSPH viscosity.
Stable at frame-scale timesteps — that is its selling point.

**Neighbor search, WebGL2**: particles scatter their indices into a grid
texture pair holding 8 slots/cell via stencil routing (every fragment
increments the cell's stencil; pass *k* keeps arrival ordinal *k*).
Cells are h/2 with a 5×5 gather, so the slot cap only bites near 10×
rest density — dropped neighbors would break force symmetry and show
up as jitter relative to the WebGPU exact grid.

**Neighbor search, WebGPU**: exact counting sort — atomic per-cell counts,
a three-dispatch prefix sum (per-workgroup scan, serial block-sum scan,
offset add), then atomic scatter into a sorted index array.

**Obstacles**: circles (no count limit — a growable data texture on WebGL2,
a growable storage buffer on WebGPU) evaluated analytically in the collision
step and drawn as an SDF overlay. Cleared when the particle count changes
(the world rescales).

**Rubber duck** (`src/sim/duck.js`): a 2D rigid body (position + rotation)
whose hull is approximated by two circles appended to the obstacle list, so
the existing boundary springs displace fluid around it in both modes on both
backends. The fluid→duck force is the Newton's-third-law reaction of those
springs, summed over all particles by a per-frame GPU reduction
(`solver.sampleDuckForce`): WebGL2 additive-blends per-particle contributions
into a 1×1 float target and reads back synchronously; WebGPU accumulates
fixed-point atomics and reads back asynchronously (the body integrates with a
frame-stale force). Buoyancy is emergent — pressure presses particles into
the spring zone until the contact sum carries the duck's weight — and the
center of mass sits below the hull-circle centers, which gives the
metacentric righting torque that keeps the duck upright. The visible duck is
a polygon mesh (body/wing/head/beak/eye triangle fans) drawn by both
renderers with the rigid transform.

### Scaling to large counts

All physics constants were tuned (and are verified) at 3000 particles in a
70-unit world. The world grows with `sqrt(count / 3000)` and gravity
shrinks by the same factor — a pure non-dimensional rescaling that keeps
velocities, timestep stability, and the settled-fluid look identical at
every count. More particles = more fluid, viewed from further away.

### Rendering

Liquid mode splats particles into a half-res field texture (density +
velocity + speed), then a composite pass thresholds it into a lit surface
(depth gradient, speed whitecap tint, diffuse + specular from the field gradient,
rim light). Dots mode draws additive sprites colored by speed. Particle
positions are read directly from solver GPU state in the vertex shader —
nothing round-trips through JavaScript.

## Ideas to expand

- More rigid bodies (the duck's circle-hull + GPU force reduction generalizes).
- Polygonal / freehand obstacle SDFs via jump-flooding.
- Multiphase fluids (per-particle density/color, buoyancy).
- Thermal convection (temperature advection + buoyant force).
