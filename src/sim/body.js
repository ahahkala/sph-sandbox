// Rigid bodies floating in (or sinking through) the fluid — ducks, bricks,
// anything else listed in bodytypes.js. Everything here is generic over that
// table: a body is a hull of circles, a mass, and a sprite to draw.
//
// Two-way coupling without CPU particle access:
//  - fluid <- body: the hull circles are appended to the solvers' obstacle
//    list, so the existing boundary-spring collision displaces fluid around
//    the body;
//  - body <- fluid: a GPU reduction pass sums the reaction of those spring
//    forces over all particles (solver.sampleBodyForce), and this class
//    integrates the rigid body from that. Buoyancy is emergent: pressure
//    presses particles into the spring zone until the contact sum carries
//    the body's weight — so whether a body floats or sinks follows from its
//    density alone (BODY_TYPES[…].densityRel).

import { WORLD_HEIGHT } from "../config.js"
import { BODY_TYPES, bodyType, DEFAULT_BODY_TYPE } from "./bodytypes.js"

const LIN_DAMP = 0.1 // 1/s
const ANG_DAMP = 0.1 // 1/s
const MAX_OMEGA = 6.0 // rad/s
const GRAB_K = 80 // drag spring stiffness, 1/s^2
const GRAB_DAMP = 12 // drag damping at the grab point, 1/s
const KEY_ACCEL = 2.5 // arrow-key thrust in multiples of gravity

export class RigidBody {
  constructor(params, typeName = DEFAULT_BODY_TYPE) {
    this.params = params
    this.typeName = BODY_TYPES[typeName] ? typeName : DEFAULT_BODY_TYPE
    this.type = bodyType(this.typeName)
    this.pose = null // {x, y, angle, scale, type}, published via params.bodyPoses
    this.circles = [] // own world-space hull circles {x, y, r}
    this.grab = null // {lx, ly: grabbed local point, tx, ty: cursor target}
    this.fluidForce = null // latest solver.sampleBodyForce result
    // body-body contact accumulated by BodyFlock.collide() for this substep
    this.extFx = 0
    this.extFy = 0
    this.extTq = 0
  }

  // Is the world point inside this body's hull (with a little slack)?
  hit(wx, wy) {
    return this.circles.some((c) => Math.hypot(wx - c.x, wy - c.y) < c.r * 1.2)
  }

  // Pointer drag: grab the hull point under the cursor; a spring toward the
  // cursor moves (and swings) the body, so releasing mid-motion throws it.
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
    const t = this.type
    const s = t.size * (world.h / WORLD_HEIGHT)
    // the renderers read `type` off the pose to pick the sprite
    this.pose = { x, y, angle: 0, scale: s, type: this.typeName }
    this.vx = 0
    this.vy = 0
    this.omega = 0
    this.grab = null
    this.fluidForce = null
    this.extFx = 0
    this.extFy = 0
    this.extTq = 0
    this.mass = t.densityRel * this.params.restDensity * t.hullArea * s * s
    this.inertia = t.inertiaFactor * this.mass * s * s
    this.sync()
  }

  // Refresh this body's world-space hull circles from its pose.
  sync() {
    const { x, y, angle, scale } = this.pose
    const c = Math.cos(angle),
      s = Math.sin(angle)
    const hull = this.type.hull
    const out = this.circles
    out.length = hull.length
    for (let i = 0; i < hull.length; i++) {
      const k = hull[i]
      out[i] = {
        x: x + (k.x * c - k.y * s) * scale,
        y: y + (k.x * s + k.y * c) * scale,
        r: k.r * scale,
      }
    }
  }

  // fluid = {fx, fy, torque} from solver.sampleBodyForce (may be a frame
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

    // body-body contact, resolved pairwise by the flock before this substep
    fx += this.extFx
    fy += this.extFy
    tq += this.extTq
    this.extFx = 0
    this.extFy = 0
    this.extTq = 0

    // fluid contact reaction, clamped so one bad frame can't launch the body
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
    const th = p.bodyThrust
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

// Every rigid body in the scene, spawned and removed with the panel's object
// tools. It owns the aggregate the rest of the app reads: params.bodyCircles
// (appended to the obstacle upload by the solvers) and params.bodyPoses
// (drawn by the renderers). Bodies collide with each other (see collide()) as
// well as with the walls, the drawn obstacles and the fluid.
export class BodyFlock {
  constructor(params) {
    this.params = params
    this.bodies = []
    this.dragging = null
  }

  spawn(world, x, y, typeName = this.params.bodyType) {
    const b = new RigidBody(this.params, typeName)
    b.reset(world, x, y)
    this.bodies.push(b)
    this.publish()
    return b
  }

  // Remove the topmost (most recently spawned) body under the point.
  removeAt(wx, wy) {
    for (let i = this.bodies.length - 1; i >= 0; i--) {
      if (this.bodies[i].hit(wx, wy)) {
        if (this.dragging === this.bodies[i]) this.dragging = null
        this.bodies.splice(i, 1)
        this.publish()
        return true
      }
    }
    return false
  }

  clear() {
    this.bodies.length = 0
    this.dragging = null
    this.publish()
  }

  startDrag(wx, wy) {
    for (let i = this.bodies.length - 1; i >= 0; i--) {
      if (this.bodies[i].startDrag(wx, wy)) {
        this.dragging = this.bodies[i]
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
    for (const b of this.bodies) b.step(dt, world, b.fluidForce)
    this.publish()
  }

  // Body-body contact: every hull-circle pair gets the same spring-damper the
  // bodies already use against walls and obstacles, applied equal and
  // opposite. The per-circle mass shares are combined as a reduced mass, so
  // two colliding bodies share the push instead of each being repelled as
  // hard as a wall does — and a brick shoves a duck aside, not vice versa.
  collide() {
    const p = this.params
    const K = p.wallStiffness,
      C = p.wallDamping
    for (let i = 0; i < this.bodies.length; i++) {
      const a = this.bodies[i]
      const ma = a.mass / a.circles.length
      for (let j = i + 1; j < this.bodies.length; j++) {
        const b = this.bodies[j]
        const mb = b.mass / b.circles.length
        const mEff = (ma * mb) / (ma + mb)
        for (const ca of a.circles) {
          for (const cb of b.circles) {
            const dx = ca.x - cb.x,
              dy = ca.y - cb.y
            const dist = Math.hypot(dx, dy)
            // penetration capped at a radius: a body spawned on top of
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

  // One reduction + readback per body; the force is held across the next
  // frame's substeps (sync on WebGL2, a frame stale on WebGPU).
  sampleForces(solver) {
    for (let i = 0; i < this.bodies.length; i++) {
      const b = this.bodies[i]
      b.fluidForce = solver.sampleBodyForce(b.circles, b.pose.x, b.pose.y, i)
    }
  }

  publish() {
    const circles = this.params.bodyCircles
    circles.length = 0
    const poses = this.params.bodyPoses
    poses.length = 0
    for (const b of this.bodies) {
      for (const c of b.circles) circles.push(c)
      poses.push(b.pose)
    }
  }
}
