// Central tunables. `params` is mutated live by the UI; solvers read it
// every step, so slider changes take effect immediately.

import { DEFAULT_BODY_TYPE } from "./sim/bodytypes.js"

// Reference world height (world units) at the reference particle count.
// The world grows with sqrt(count / REF_COUNT) and gravity shrinks by the
// same factor, so dynamics and stability are identical at every count.
export const WORLD_HEIGHT = 70
export const REF_COUNT = 3000

// posVel state texture width for the WebGL2 backend (one texel per
// particle); height = ceil(N/512). 512 x 512 = 262144 particles max.
export const TEX_WIDTH = 512

// Physics timesteps. Force SPH needs small explicit steps; PBF's constraint
// projection is stable at frame-scale steps (that is its selling point).
export const DT_FORCE = 1 / 240
export const DT_PBF = 1 / 60

export function createParams() {
  return {
    // SPH core
    h: 1.6, // smoothing radius (world units)
    restDensity: 1.7, // ≈ density of the spawn lattice (spacing 0.55 h)
    mass: 1.0,

    // user-tunable
    physics: "force", // "force" (Müller SPH) | "pbf" (position-based fluids)
    count: 100000,
    gravity: 15,
    gravityCenter: false, // pull toward the world center instead of straight down
    viscosity: 1.5,
    stiffness: 1000, // force mode: P = k (rho - rho0)
    nearStiffness: 0, // force mode: anti-clump / surface tension
    vorticity: 1.5, // vorticity confinement strength (force mode only)

    // PBF-specific (fixed; exposed here for tests/tuning)
    pbfIterations: 3,
    pbfRelax: 5.0, // constraint relaxation epsilon
    pbfSCorrK: 0.08, // tensile instability correction strength
    pbfSCorrDq: 0.3, // s_corr reference distance, fraction of h
    pbfXsph: 0.03, // XSPH viscosity per unit of the viscosity slider

    // stability
    dt: DT_FORCE, // active substep (switched with physics mode)
    maxSubsteps: 16, // per rendered frame; time scaling needs ~4·sqrt(N/3000) in force mode
    maxSpeed: 80, // velocity clamp
    boundaryDamp: 0.5, // wall restitution (hard-clamp safety net)
    wallStiffness: 300, // boundary spring: accel per unit penetration
    wallDamping: 40, // boundary normal damping, 1/s (kills edge jitter)

    // interaction
    pointerRadius: 8, // world units at the reference world size
    pointerStrength: 100,
    // pointer tools; at most one is active at a time (see setupControls)
    drawObstacles: false,
    eraseObstacles: false, // eraser tool: drag removes obstacles near the cursor
    addBodies: false, // click spawns a rigid body at the cursor
    removeBodies: false, // click removes the body under the cursor
    obstacleRadius: 3, // drawn wall stamp radius, world units at the reference world size
    obstacles: [], // {x, y, r} in world units
    // rigid bodies (ducks, bricks, … — see sim/bodytypes.js), owned by BodyFlock
    bodyType: DEFAULT_BODY_TYPE, // what the Add tool spawns
    bodyCircles: [], // hull circles {x, y, r} of all bodies, appended to obstacles by the solvers
    bodyPoses: [], // {x, y, angle, scale, type} per body, for the renderers
    bodyThrust: { x: 0, y: 0 }, // arrow-key force direction on the bodies (unit-ish)

    // rendering
    renderMode: "liquid", // "liquid" | "dots"
    speedRef: 30, // speed mapped to the hottest ramp color
  }
}
