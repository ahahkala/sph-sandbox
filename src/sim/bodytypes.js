// Rigid-body catalogue. Everything that makes one body type differ from
// another lives here; RigidBody (body.js) and both renderers are generic
// over this table.
//
// Local frame: y-down like the world, the body faces +x, origin at the
// center of mass. For a floater that means putting the origin below the
// hull circles — buoyant contact then acts above it, which is the
// metacentric righting that keeps a rubber duck upright.
//
// Fields:
//   label         name shown in the UI dropdown
//   sprite        transparent PNG drawn on the quad `rect` (tools/make-sprites.mjs)
//   rect          sprite quad in local units {x0, y0 (top), x1, y1 (bottom)}
//   hull          collision circles in local units; the solvers see only these
//   size          world units per local unit at the reference world height
//   densityRel    fraction of restDensity — below 1 floats, above 1 sinks
//   hullArea      local-unit area the hull displaces (sets the mass)
//   inertiaFactor moment of inertia as a fraction of mass · size²

const asset = (file) => new URL("../../assets/" + file, import.meta.url).href

export const BODY_TYPES = {
  duck: {
    label: "Duck",
    sprite: asset("duck.png"),
    rect: { x0: -1.5, y0: -1.85, x1: 1.75, y1: 1.05 },
    hull: [
      { x: 0, y: 0, r: 0.88 }, // body
      { x: 0.62, y: -1.12, r: 0.4 }, // head
      { x: -0.85, y: -0.4, r: 0.4 }, // tail
    ],
    size: 3.0,
    densityRel: 0.45,
    hullArea: 3.0,
    inertiaFactor: 0.4,
  },
  brick: {
    label: "Brick",
    sprite: asset("brick.png"),
    rect: { x0: -1.1, y0: -0.6, x1: 1.103, y1: 0.603 },
    hull: [
      { x: -0.809, y: -0.282, r: 0.233 },
      { x: 0.793, y: -0.308, r: 0.225 },
      { x: 0.801, y: 0.321, r: 0.224 },
      { x: -0.817, y: 0.321, r: 0.215 },
      { x: 0.002, y: 0.002, r: 0.512 },
      { x: -0.523, y: 0.008, r: 0.505 },
      { x: 0.506, y: 0.021, r: 0.503 },
    ],
    size: 2.5,
    densityRel: 5,
    hullArea: 2.001,
    inertiaFactor: 0.427,
  },
}

export const DEFAULT_BODY_TYPE = "duck"

export const bodyType = (name) =>
  BODY_TYPES[name] || BODY_TYPES[DEFAULT_BODY_TYPE]
