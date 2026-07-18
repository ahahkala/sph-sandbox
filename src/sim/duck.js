// Rubber duck: a 2D polygonal rigid body floating on the fluid.
//
// Two-way coupling without CPU particle access:
//  - fluid <- duck: the hull is approximated by a few circles that the
//    solvers append to the obstacle list, so the existing boundary-spring
//    collision displaces fluid around the duck;
//  - duck <- fluid: a GPU reduction pass sums the reaction of those spring
//    forces over all particles (solver.sampleDuckForce), and this class
//    integrates the rigid body from that. Buoyancy is emergent: pressure
//    presses particles into the spring zone until the contact sum carries
//    the duck's weight.
//
// Local frame: y-down like the world, duck faces +x, origin at the center
// of mass (placed low in the body so buoyant contact acts above it — the
// metacentric righting that keeps a rubber duck upright).

import { WORLD_HEIGHT } from "../config.js"

export const DUCK_SIZE = 3.0 // world units per local unit at the reference world height
const DENSITY_REL = 0.45 // fraction of restDensity: < 1 floats
const HULL_AREA = 3.0 // approx local-unit area of the hull circles
const LIN_DAMP = 0.1 // 1/s
const ANG_DAMP = 0.1 // 1/s
const MAX_OMEGA = 6.0 // rad/s
const GRAB_K = 80 // drag spring stiffness, 1/s^2
const GRAB_DAMP = 12 // drag damping at the grab point, 1/s
const KEY_ACCEL = 2.5 // arrow-key thrust in multiples of gravity
const BODY_UP = 0.0 // body-circle center height above the CoM

// --- mesh ------------------------------------------------------------------
// Interleaved triangle list [x, y, r, g, b] in local units (y-down).

const YELLOW = [1.0, 0.8, 0.12]
const WING = [0.9, 0.64, 0.1]
const BEAK = [0.96, 0.47, 0.1]
const EYE = [0.1, 0.09, 0.08]

function buildMesh() {
  const verts = [] // built y-up for sanity, flipped on emit
  const shade = (y) => 0.8 + 0.2 * Math.min(Math.max((y + 0.7) / 2.1, 0), 1)
  const push = (x, y, col, shaded) => {
    const s = shaded ? shade(y) : 1
    verts.push(x, -y, col[0] * s, col[1] * s, col[2] * s)
  }
  const fan = (cx, cy, pts, col, shaded) => {
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i],
        b = pts[(i + 1) % pts.length]
      push(cx, cy, col, shaded)
      push(a[0], a[1], col, shaded)
      push(b[0], b[1], col, shaded)
    }
  }
  const ellipse = (cx, cy, rx, ry, rot, n) => {
    const pts = []
    for (let i = 0; i < n; i++) {
      const t = (i / n) * Math.PI * 2
      const ex = Math.cos(t) * rx,
        ey = Math.sin(t) * ry
      pts.push([
        cx + ex * Math.cos(rot) - ey * Math.sin(rot),
        cy + ex * Math.sin(rot) + ey * Math.cos(rot),
      ])
    }
    return pts
  }

  // body: squashed ellipse with a tail bump swept up-back
  const bodyPts = []
  for (let i = 0; i < 48; i++) {
    const t = (i / 48) * Math.PI * 2
    const ct = Math.cos(t),
      st = Math.sin(t)
    let r = 1 / Math.sqrt((ct / 1.18) ** 2 + (st / 0.88) ** 2)
    let d = t - 2.65 // tail direction
    if (d > Math.PI) d -= 2 * Math.PI
    if (d < -Math.PI) d += 2 * Math.PI
    r += 0.5 * Math.exp(-((d / 0.38) ** 2))
    bodyPts.push([r * ct, BODY_UP + r * st])
  }
  fan(0, BODY_UP, bodyPts, YELLOW, true)
  fan(-0.22, 0.28, ellipse(-0.22, 0.28, 0.5, 0.3, -0.44, 24), WING, true)
  fan(0.62, 1.12, ellipse(0.62, 1.12, 0.5, 0.5, 0, 32), YELLOW, true)
  const beakPts = [
    [1.04, 1.24],
    [1.52, 1.12],
    [1.5, 1.0],
    [1.04, 0.98],
  ]
  fan(1.27, 1.11, beakPts, BEAK, false)
  fan(0.74, 1.28, ellipse(0.74, 1.28, 0.085, 0.085, 0, 12), EYE, false)
  return new Float32Array(verts)
}

export const DUCK_MESH = buildMesh() // stride 5: x, y, r, g, b
export const DUCK_STRIDE = 5

// Hull circles in local units (y-down): body + head + tail.
export const DUCK_HULL = [
  { x: 0, y: -BODY_UP, r: 0.88 },
  { x: 0.62, y: -1.12, r: 0.4 },
  { x: -0.85, y: -0.4, r: 0.4 },
]

// --- rigid body ------------------------------------------------------------

export class Duck {
  constructor(params) {
    this.params = params
    this.pose = null // {x, y, angle, scale}, published via params.duckPoses
    this.circles = [] // own world-space hull circles {x, y, r}
    this.grab = null // {lx, ly: grabbed local point, tx, ty: cursor target}
    this.fluidForce = null // latest solver.sampleDuckForce result
    // duck-duck contact accumulated by DuckFlock.collide() for this substep
    this.extFx = 0
    this.extFy = 0
    this.extTq = 0
  }

  // Is the world point inside this duck's hull (with a little slack)?
  hit(wx, wy) {
    return this.circles.some((c) => Math.hypot(wx - c.x, wy - c.y) < c.r * 1.2)
  }

  // Pointer drag: grab the hull point under the cursor; a spring toward the
  // cursor moves (and swings) the duck, so releasing mid-motion throws it.
  startDrag(wx, wy) {
    if (!this.pose || !this.hit(wx, wy)) return false
    const { x, y, angle, scale } = this.pose
    const c = Math.cos(angle),
      s = Math.sin(angle)
    const dx = wx - x,
      dy = wy - y
    this.grab = {
      lx: (dx * c + dy * s) / scale,
      ly: (-dx * s + dy * c) / scale,
      tx: wx,
      ty: wy,
    }
    return true
  }

  dragTo(wx, wy) {
    if (this.grab) {
      this.grab.tx = wx
      this.grab.ty = wy
    }
  }

  endDrag() {
    this.grab = null
  }

  reset(world, x = world.w * 0.5, y = world.h * 0.25) {
    const s = DUCK_SIZE * (world.h / WORLD_HEIGHT)
    this.pose = { x, y, angle: 0, scale: s }
    this.vx = 0
    this.vy = 0
    this.omega = 0
    this.grab = null
    this.fluidForce = null
    this.extFx = 0
    this.extFy = 0
    this.extTq = 0
    this.mass =DENSITY_REL * this.params.restDensity * HULL_AREA * s * s
    this.inertia = 0.4 * this.mass * s * s
    this.sync()
  }

  // Refresh this duck's world-space hull circles from its pose.
  sync() {
    const { x, y, angle, scale } = this.pose
    const c = Math.cos(angle),
      s = Math.sin(angle)
    const out = this.circles
    out.length = DUCK_HULL.length
    for (let i = 0; i < DUCK_HULL.length; i++) {
      const k = DUCK_HULL[i]
      out[i] = {
        x: x + (k.x * c - k.y * s) * scale,
        y: y + (k.x * s + k.y * c) * scale,
        r: k.r * scale,
      }
    }
  }

  // fluid = {fx, fy, torque} from solver.sampleDuckForce (may be a frame
  // stale; null before the first sample).
  step(dt, world, fluid) {
    const p = this.params
    const ws = world.h / WORLD_HEIGHT
    const g = p.gravity / ws

    // gravity, matching the fluid's direction mode
    let fx, fy
    if (p.gravityCenter) {
      const dx = world.w * 0.5 - this.pose.x,
        dy = world.h * 0.5 - this.pose.y
      const len = Math.hypot(dx, dy) || 1
      fx = (g / len) * dx * this.mass
      fy = (g / len) * dy * this.mass
    } else {
      fx = 0
      fy = g * this.mass
    }
    let tq = 0

    // duck-duck contact, resolved pairwise by the flock before this substep
    fx += this.extFx
    fy += this.extFy
    tq += this.extTq
    this.extFx = 0
    this.extFy = 0
    this.extTq = 0

    // fluid contact reaction, clamped so one bad frame can't launch the duck
    if (fluid) {
      const cap = 15 * this.mass * Math.max(g, 2)
      const mag = Math.hypot(fluid.fx, fluid.fy)
      const k = mag > cap ? cap / mag : 1
      fx += fluid.fx * k
      fy += fluid.fy * k
      const tcap = cap * this.pose.scale
      tq += Math.max(-tcap, Math.min(tcap, fluid.torque))
    }

    // arrow-key thrust
    const th = p.duckThrust
    if (th.x !== 0 || th.y !== 0) {
      const len = Math.hypot(th.x, th.y)
      const a = KEY_ACCEL * Math.max(g, 2) * this.mass
      fx += (th.x / len) * a
      fy += (th.y / len) * a
    }

    // pointer drag: spring-damper from the grabbed hull point to the cursor
    if (this.grab) {
      const gc = Math.cos(this.pose.angle),
        gs = Math.sin(this.pose.angle)
      const rx = (this.grab.lx * gc - this.grab.ly * gs) * this.pose.scale
      const ry = (this.grab.lx * gs + this.grab.ly * gc) * this.pose.scale
      const vgx = this.vx - this.omega * ry
      const vgy = this.vy + this.omega * rx
      let ax = GRAB_K * (this.grab.tx - (this.pose.x + rx)) - GRAB_DAMP * vgx
      let ay = GRAB_K * (this.grab.ty - (this.pose.y + ry)) - GRAB_DAMP * vgy
      const am = Math.hypot(ax, ay)
      const acap = 30 * Math.max(g, 2)
      if (am > acap) {
        ax *= acap / am
        ay *= acap / am
      }
      fx += ax * this.mass
      fy += ay * this.mass
      tq += rx * ay * this.mass - ry * ax * this.mass
    }

    // walls + drawn obstacles: same spring-damper the particles use,
    // evaluated per hull circle at that circle's velocity
    const K = p.wallStiffness,
      C = p.wallDamping
    const mShare = this.mass / this.circles.length
    for (const c of this.circles) {
      const rx = c.x - this.pose.x,
        ry = c.y - this.pose.y
      const cvx = this.vx - this.omega * ry
      const cvy = this.vy + this.omega * rx
      const spring = (pen, nx, ny) => {
        if (pen <= 0) return
        const vn = cvx * nx + cvy * ny
        const f = Math.max(K * pen - C * Math.min(vn, 0), 0) * mShare
        fx += f * nx
        fy += f * ny
        tq += rx * f * ny - ry * f * nx
      }
      spring(c.r - c.x, 1, 0)
      spring(c.x - (world.w - c.r), -1, 0)
      spring(c.r - c.y, 0, 1)
      spring(c.y - (world.h - c.r), 0, -1)
      for (const o of p.obstacles) {
        const dx = c.x - o.x,
          dy = c.y - o.y
        const dist = Math.hypot(dx, dy)
        const pen = c.r + o.r - dist
        if (pen > 0 && dist > 1e-4) spring(pen, dx / dist, dy / dist)
      }
    }

    fx -= LIN_DAMP * this.mass * this.vx
    fy -= LIN_DAMP * this.mass * this.vy
    tq -= ANG_DAMP * this.inertia * this.omega

    this.vx += (fx / this.mass) * dt
    this.vy += (fy / this.mass) * dt
    const sp = Math.hypot(this.vx, this.vy)
    if (sp > p.maxSpeed) {
      this.vx *= p.maxSpeed / sp
      this.vy *= p.maxSpeed / sp
    }
    this.omega += (tq / this.inertia) * dt
    this.omega = Math.max(-MAX_OMEGA, Math.min(MAX_OMEGA, this.omega))

    this.pose.x += this.vx * dt
    this.pose.y += this.vy * dt
    this.pose.angle += this.omega * dt
    // safety net: never let the center leave the box (e.g. after a resize)
    this.pose.x = Math.max(0, Math.min(world.w, this.pose.x))
    this.pose.y = Math.max(0, Math.min(world.h, this.pose.y))
    this.sync()
  }
}

// A flock of ducks, spawned and removed with the panel's duck tools. It owns
// the aggregate the rest of the app reads: params.duckCircles (appended to the
// obstacle upload by the solvers) and params.duckPoses (drawn by the
// renderers). Ducks collide with each other (see collide()) as well as with
// the walls, the drawn obstacles and the fluid.
export class DuckFlock {
  constructor(params) {
    this.params = params
    this.ducks = []
    this.dragging = null
  }

  spawn(world, x, y) {
    const duck = new Duck(this.params)
    duck.reset(world, x, y)
    this.ducks.push(duck)
    this.publish()
    return duck
  }

  // Remove the topmost (most recently spawned) duck under the point.
  removeAt(wx, wy) {
    for (let i = this.ducks.length - 1; i >= 0; i--) {
      if (this.ducks[i].hit(wx, wy)) {
        if (this.dragging === this.ducks[i]) this.dragging = null
        this.ducks.splice(i, 1)
        this.publish()
        return true
      }
    }
    return false
  }

  clear() {
    this.ducks.length = 0
    this.dragging = null
    this.publish()
  }

  startDrag(wx, wy) {
    for (let i = this.ducks.length - 1; i >= 0; i--) {
      if (this.ducks[i].startDrag(wx, wy)) {
        this.dragging = this.ducks[i]
        return true
      }
    }
    return false
  }

  dragTo(wx, wy) {
    if (this.dragging) this.dragging.dragTo(wx, wy)
  }

  endDrag() {
    if (this.dragging) this.dragging.endDrag()
    this.dragging = null
  }

  step(dt, world) {
    this.collide()
    for (const d of this.ducks) d.step(dt, world, d.fluidForce)
    this.publish()
  }

  // Duck-duck contact: every hull-circle pair gets the same spring-damper the
  // ducks already use against walls and obstacles, applied equal and opposite.
  // The per-circle mass share is combined as a reduced mass, so two colliding
  // ducks share the push instead of each being repelled as hard as a wall does.
  collide() {
    const p = this.params
    const K = p.wallStiffness,
      C = p.wallDamping
    for (let i = 0; i < this.ducks.length; i++) {
      const a = this.ducks[i]
      const ma = a.mass / a.circles.length
      for (let j = i + 1; j < this.ducks.length; j++) {
        const b = this.ducks[j]
        const mb = b.mass / b.circles.length
        const mEff = (ma * mb) / (ma + mb)
        for (const ca of a.circles) {
          for (const cb of b.circles) {
            const dx = ca.x - cb.x,
              dy = ca.y - cb.y
            const dist = Math.hypot(dx, dy)
            // penetration capped at a radius: a duck spawned on top of
            // another separates firmly but doesn't get launched
            const pen = Math.min(ca.r + cb.r - dist, Math.min(ca.r, cb.r))
            if (pen <= 0 || dist < 1e-4) continue
            const nx = dx / dist,
              ny = dy / dist
            // contact-point velocities (v + omega x r), relative, along n
            const arx = ca.x - a.pose.x,
              ary = ca.y - a.pose.y
            const brx = cb.x - b.pose.x,
              bry = cb.y - b.pose.y
            const rvx = a.vx - a.omega * ary - (b.vx - b.omega * bry)
            const rvy = a.vy + a.omega * arx - (b.vy + b.omega * brx)
            const vn = rvx * nx + rvy * ny
            const f = Math.max(K * pen - C * Math.min(vn, 0), 0) * mEff
            const fxc = f * nx,
              fyc = f * ny
            a.extFx += fxc
            a.extFy += fyc
            a.extTq += arx * fyc - ary * fxc
            b.extFx -= fxc
            b.extFy -= fyc
            b.extTq -= brx * fyc - bry * fxc
          }
        }
      }
    }
  }

  // One reduction + readback per duck; the force is held across the next
  // frame's substeps (sync on WebGL2, a frame stale on WebGPU).
  sampleForces(solver) {
    for (let i = 0; i < this.ducks.length; i++) {
      const d = this.ducks[i]
      d.fluidForce = solver.sampleDuckForce(d.circles, d.pose.x, d.pose.y, i)
    }
  }

  publish() {
    const circles = this.params.duckCircles
    circles.length = 0
    const poses = this.params.duckPoses
    poses.length = 0
    for (const d of this.ducks) {
      for (const c of d.circles) circles.push(c)
      poses.push(d.pose)
    }
  }
}
