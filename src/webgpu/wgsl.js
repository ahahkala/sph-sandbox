// WGSL sources for the WebGPU backend. Same force-SPH math as the WebGL2
// shaders, but with an exact grid: atomic counting sort + prefix sum gives
// every particle its full neighbor list (no per-cell slot cap).

// All params are f32 (converted where indices are needed) to keep the
// uniform layout trivial. Mirrors the order in wgpusolver.writeParams().
export const PARAMS_STRUCT = /* wgsl */ `
struct Params {
  pointer: vec4f,         // x, y, signed strength (0 = off), radius
  world: vec2f,
  gridDims: vec2f,        // cols, rows
  count: f32,
  obstacleCount: f32,
  cellSize: f32,
  h: f32,
  h2: f32,
  poly6: f32,
  spiky3: f32,
  spikyGrad: f32,
  viscLap: f32,
  mass: f32,
  k: f32,
  kNear: f32,
  rho0: f32,
  mu: f32,
  vort: f32,
  g: f32,
  dt: f32,
  maxV: f32,
  damp: f32,
  eps: f32,
  wallK: f32,   // boundary spring stiffness (accel / unit depth)
  wallC: f32,   // boundary normal damping (1 / s)
  pbfRelax: f32,
  pbfSCorrK: f32,
  pbfInvWDq: f32,
  pbfXsph: f32,
  gravMode: f32,          // 0 = down, 1 = toward the world center
  pad0: f32,
}
`;
// NOTE: struct size is exactly 144 bytes — paramsData must stay in sync.

// Gravity: straight down, or a constant-magnitude pull toward the world
// center. Needs P in scope.
const gravityFn = /* wgsl */ `
fn gravity(pos: vec2f) -> vec2f {
  if (P.gravMode == 0.0) { return vec2f(0.0, P.g); }
  let d = P.world * 0.5 - pos;
  let len = length(d);
  if (len > 1e-4) { return (P.g / len) * d; }
  return vec2f(0.0);
}
`;

const cellOf = /* wgsl */ `
fn cellOf(pos: vec2f) -> vec2i {
  let c = vec2i(floor(pos / P.cellSize));
  return clamp(c, vec2i(0), vec2i(P.gridDims) - 1);
}
fn cellIndex(c: vec2i) -> u32 {
  return u32(c.y) * u32(P.gridDims.x) + u32(c.x);
}
`;

export const countWGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> posVel: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> cellCount: array<atomic<u32>>;
${cellOf}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(P.count)) { return; }
  let c = cellIndex(cellOf(posVel[i].xy));
  atomicAdd(&cellCount[c], 1u);
}
`;

// Exclusive prefix sum, three dispatches:
//  1. per-workgroup Hillis-Steele scan (256 elements) + block sums
//  2. serial exclusive scan of the block sums (single thread; few thousand)
//  3. add scanned block offsets back
export const scanBlockWGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> input: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<u32>;
@group(0) @binding(2) var<storage, read_write> blockSums: array<u32>;
var<workgroup> temp: array<u32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(local_invocation_id) lid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  let n = arrayLength(&input);
  let i = gid.x;
  let v = select(0u, input[i], i < n);
  temp[lid.x] = v;
  for (var off = 1u; off < 256u; off = off << 1u) {
    workgroupBarrier();
    var add = 0u;
    if (lid.x >= off) { add = temp[lid.x - off]; }
    workgroupBarrier();
    temp[lid.x] = temp[lid.x] + add;
  }
  workgroupBarrier();
  if (i < n) { output[i] = temp[lid.x] - v; }   // inclusive -> exclusive
  if (lid.x == 255u) { blockSums[wid.x] = temp[255u]; }
}
`;

export const scanSerialWGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> sums: array<u32>;
@compute @workgroup_size(1)
fn main() {
  let n = arrayLength(&sums);
  var acc = 0u;
  for (var i = 0u; i < n; i = i + 1u) {
    let v = sums[i];
    sums[i] = acc;
    acc = acc + v;
  }
}
`;

export const scanAddWGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read_write> output: array<u32>;
@group(0) @binding(1) var<storage, read> blockSums: array<u32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u,
        @builtin(workgroup_id) wid: vec3u) {
  if (gid.x >= arrayLength(&output)) { return; }
  output[gid.x] = output[gid.x] + blockSums[wid.x];
}
`;

export const scatterWGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> posVel: array<vec4f>;
@group(0) @binding(2) var<storage, read> cellStart: array<u32>;
@group(0) @binding(3) var<storage, read_write> cellFill: array<atomic<u32>>;
@group(0) @binding(4) var<storage, read_write> sortedIdx: array<u32>;
${cellOf}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(P.count)) { return; }
  let c = cellIndex(cellOf(posVel[i].xy));
  let slot = cellStart[c] + atomicAdd(&cellFill[c], 1u);
  sortedIdx[slot] = i;
}
`;

// Neighbor iteration shared by density and force passes.
const gatherLoop = (body) => /* wgsl */ `
  let cc = cellOf(pos);
  for (var gy = -1; gy <= 1; gy = gy + 1) {
    for (var gx = -1; gx <= 1; gx = gx + 1) {
      let c = cc + vec2i(gx, gy);
      if (c.x < 0 || c.y < 0 || c.x >= i32(P.gridDims.x) || c.y >= i32(P.gridDims.y)) { continue; }
      let ci = cellIndex(c);
      let s0 = cellStart[ci];
      let s1 = s0 + cellCount[ci];
      for (var s = s0; s < s1; s = s + 1u) {
        let j = sortedIdx[s];
        if (j == i) { continue; }
        ${body}
      }
    }
  }
`;

export const densityWGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> posVel: array<vec4f>;
@group(0) @binding(2) var<storage, read> cellStart: array<u32>;
@group(0) @binding(3) var<storage, read> cellCount: array<u32>;
@group(0) @binding(4) var<storage, read> sortedIdx: array<u32>;
@group(0) @binding(5) var<storage, read_write> density: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> curl: array<vec2f>;
${cellOf}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(P.count)) { return; }
  let pv = posVel[i];
  let pos = pv.xy;
  let vel = pv.zw;
  var rho = P.mass * P.poly6 * P.h2 * P.h2 * P.h2;
  var rhoNear = P.mass * P.spiky3 * P.h * P.h * P.h;
  var omega = 0.0;
  ${gatherLoop(/* wgsl */ `
        let pvj = posVel[j];
        let d = pos - pvj.xy;
        let r2 = dot(d, d);
        if (r2 < P.h2 && r2 > 1e-10) {
          let t = P.h2 - r2;
          rho = rho + P.mass * P.poly6 * t * t * t;
          let r = sqrt(r2);
          let w1 = P.h - r;
          rhoNear = rhoNear + P.mass * P.spiky3 * w1 * w1 * w1;
          let gradW = (-P.spikyGrad * w1 * w1 / r) * d;
          let dv = pvj.zw - vel;
          omega = omega + (P.mass / P.rho0) * (dv.x * gradW.y - dv.y * gradW.x);
        }
  `)}
  let p = max(P.k * (rho - P.rho0), 0.0);
  density[i] = vec4f(rho, rhoNear, p, P.kNear * rhoNear);
  curl[i] = vec2f(omega, abs(omega));
}
`;

// Smooth spring-damper boundary (see WebGL2 shaders.js): spring only pushes
// out, damping only resists approach; keeps resting particles off the hard
// clamp line so edges stay calm. Needs P + the obstacles binding in scope.
const boundaryFns = /* wgsl */ `
fn wallSpring(pen: f32, nrm: vec2f, vel: vec2f) -> vec2f {
  let vn = dot(vel, nrm);
  let mag = P.wallK * pen - P.wallC * min(vn, 0.0);
  return max(mag, 0.0) * nrm;
}
fn boundaryAccel(pos: vec2f, vel: vec2f, margin: f32) -> vec2f {
  var acc = vec2f(0.0);
  if (pos.x < margin)             { acc = acc + wallSpring(margin - pos.x, vec2f(1.0, 0.0), vel); }
  if (pos.x > P.world.x - margin) { acc = acc + wallSpring(pos.x - (P.world.x - margin), vec2f(-1.0, 0.0), vel); }
  if (pos.y < margin)             { acc = acc + wallSpring(margin - pos.y, vec2f(0.0, 1.0), vel); }
  if (pos.y > P.world.y - margin) { acc = acc + wallSpring(pos.y - (P.world.y - margin), vec2f(0.0, -1.0), vel); }
  for (var k = 0; k < i32(P.obstacleCount); k = k + 1) {
    let d = pos - obstacles[k].xy;
    let dist = length(d);
    let pen = obstacles[k].z + margin - dist;
    if (pen > 0.0 && dist > 1e-4) {
      acc = acc + wallSpring(pen, d / dist, vel);
    }
  }
  return acc;
}
`;

// Hard clamp + velocity flip (fast-penetration safety net). Statement chunk
// operating on local "pos"/"vel" vars.
const collideChunk = /* wgsl */ `
  if (pos.x < P.eps)             { pos.x = P.eps;             if (vel.x < 0.0) { vel.x = vel.x * -P.damp; } }
  if (pos.x > P.world.x - P.eps) { pos.x = P.world.x - P.eps; if (vel.x > 0.0) { vel.x = vel.x * -P.damp; } }
  if (pos.y < P.eps)             { pos.y = P.eps;             if (vel.y < 0.0) { vel.y = vel.y * -P.damp; } }
  if (pos.y > P.world.y - P.eps) { pos.y = P.world.y - P.eps; if (vel.y > 0.0) { vel.y = vel.y * -P.damp; } }
  for (var k = 0; k < i32(P.obstacleCount); k = k + 1) {
    let dvec = pos - obstacles[k].xy;
    let dist = length(dvec);
    let pen = obstacles[k].z + P.eps - dist;
    if (pen > 0.0) {
      var nrm = vec2f(0.0, -1.0);
      if (dist > 1e-4) { nrm = dvec / dist; }
      pos = pos + nrm * pen;
      let vn = dot(vel, nrm);
      if (vn < 0.0) { vel = vel - (1.0 + P.damp) * vn * nrm; }
    }
  }
`;

export const forceWGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> posVel: array<vec4f>;
@group(0) @binding(2) var<storage, read> cellStart: array<u32>;
@group(0) @binding(3) var<storage, read> cellCount: array<u32>;
@group(0) @binding(4) var<storage, read> sortedIdx: array<u32>;
@group(0) @binding(5) var<storage, read> density: array<vec4f>;
@group(0) @binding(6) var<storage, read> curl: array<vec2f>;
@group(0) @binding(7) var<storage, read_write> posVelOut: array<vec4f>;
@group(0) @binding(8) var<storage, read> obstacles: array<vec4f>;
${gravityFn}
${cellOf}
${boundaryFns}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(P.count)) { return; }
  let pv = posVel[i];
  var pos = pv.xy;
  var vel = pv.zw;
  let di = density[i];
  let omegaI = curl[i].x;

  var f = vec2f(0.0);
  var gradAbsOmega = vec2f(0.0);
  ${gatherLoop(/* wgsl */ `
        let pvj = posVel[j];
        let d = pos - pvj.xy;
        let r2 = dot(d, d);
        if (r2 < P.h2 && r2 > 1e-10) {
          let r = sqrt(r2);
          let w1 = P.h - r;
          let dj = density[j];
          let invRhoB = 1.0 / max(dj.x, 1e-6);
          let coef = P.mass * P.spikyGrad * w1 * w1 * invRhoB / r *
                     (0.5 * (di.z + dj.z) + 0.5 * (di.w + dj.w) * w1);
          f = f + coef * d;
          f = f + P.mu * P.mass * P.viscLap * w1 * invRhoB * (pvj.zw - vel);
          gradAbsOmega = gradAbsOmega +
            (P.mass * invRhoB * curl[j].y * -P.spikyGrad * w1 * w1 / r) * d;
        }
  `)}

  let invRho = 1.0 / max(di.x, 1e-6);
  var acc = f * invRho + gravity(pos);
  acc = acc + boundaryAccel(pos, vel, P.h);

  let nLen = length(gradAbsOmega);
  if (P.vort > 0.0 && nLen > 1e-6) {
    let nHat = gradAbsOmega / nLen;
    acc = acc + P.vort * omegaI * vec2f(nHat.y, -nHat.x);
  }

  if (P.pointer.z != 0.0) {
    let d = pos - P.pointer.xy;
    let r2 = dot(d, d);
    let R = P.pointer.w;
    if (r2 < R * R && r2 > 1e-4) {
      let r = sqrt(r2);
      acc = acc + P.pointer.z * (1.0 - r / R) / r * d;
    }
  }

  vel = vel + acc * P.dt;
  let sp = length(vel);
  if (sp > P.maxV) { vel = vel * (P.maxV / sp); }
  pos = pos + vel * P.dt;
  ${collideChunk}
  posVelOut[i] = vec4f(pos, vel);
}
`;

// Body reaction-force reduction (see WebGL2 bodyForceVS): every particle
// recomputes its boundary spring against the body hull circles; the negated
// force + torque about the body center are summed with fixed-point atomics
// (D.fix units per force unit). P.wallC is written pre-halved in PBF mode.
export const bodyForceWGSL = /* wgsl */ `
${PARAMS_STRUCT}
struct BodyParams {
  center: vec2f,
  count: f32,
  fix: f32,                  // fixed-point scale
  circles: array<vec4f, 8>,  // xy = center, z = radius
}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> posVel: array<vec4f>;
@group(0) @binding(2) var<uniform> D: BodyParams;
@group(0) @binding(3) var<storage, read_write> outF: array<atomic<i32>, 3>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(P.count)) { return; }
  let pv = posVel[i];
  var F = vec2f(0.0);
  var hit = false;
  for (var k = 0; k < i32(D.count); k = k + 1) {
    let d = pv.xy - D.circles[k].xy;
    let dist = length(d);
    let pen = D.circles[k].z + P.h - dist;
    if (pen > 0.0 && dist > 1e-4) {
      let nrm = d / dist;
      let vn = dot(pv.zw, nrm);
      let mag = max(P.wallK * pen - P.wallC * min(vn, 0.0), 0.0);
      F = F - mag * P.mass * nrm;   // reaction on the body
      hit = true;
    }
  }
  if (!hit) { return; }
  let r = pv.xy - D.center;
  atomicAdd(&outF[0], i32(F.x * D.fix));
  atomicAdd(&outF[1], i32(F.y * D.fix));
  atomicAdd(&outF[2], i32((r.x * F.y - r.y * F.x) * D.fix));
}
`;

// ---------------------------------------------------------------------------
// PBF (position-based fluids) — same math as the WebGL2 pbfshaders.js:
// predict -> [lambda -> deltaP] iterations -> finalize. Storage buffers
// replace the texture ping-pong; the neighbor grid is built once from the
// predicted positions.
// ---------------------------------------------------------------------------

export const predictWGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> posVel: array<vec4f>;
@group(0) @binding(2) var<storage, read_write> pred: array<vec4f>;
${gravityFn}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(P.count)) { return; }
  let pv = posVel[i];
  let pos = pv.xy;
  var vel = pv.zw;
  var acc = gravity(pos);
  if (P.pointer.z != 0.0) {
    let d = pos - P.pointer.xy;
    let r2 = dot(d, d);
    let R = P.pointer.w;
    if (r2 < R * R && r2 > 1e-4) {
      let r = sqrt(r2);
      acc = acc + P.pointer.z * (1.0 - r / R) / r * d;
    }
  }
  vel = vel + acc * P.dt;
  let sp = length(vel);
  if (sp > P.maxV) { vel = vel * (P.maxV / sp); }
  pred[i] = vec4f(pos + vel * P.dt, vel);
}
`;

// lambda_i = -C_i / (sum |grad C|^2 + eps),  C_i = rho_i/rho0 - 1 (>= 0)
export const pbfLambdaWGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> posVel: array<vec4f>;   // predicted
@group(0) @binding(2) var<storage, read> cellStart: array<u32>;
@group(0) @binding(3) var<storage, read> cellCount: array<u32>;
@group(0) @binding(4) var<storage, read> sortedIdx: array<u32>;
@group(0) @binding(5) var<storage, read_write> lambda: array<vec2f>;  // lambda, rho
${cellOf}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(P.count)) { return; }
  let pos = posVel[i].xy;
  var rho = P.mass * P.poly6 * P.h2 * P.h2 * P.h2;   // self term
  var gradSelf = vec2f(0.0);
  var gradSum = 0.0;
  let invRho0 = 1.0 / P.rho0;
  ${gatherLoop(/* wgsl */ `
        let d = pos - posVel[j].xy;
        let r2 = dot(d, d);
        if (r2 < P.h2 && r2 > 1e-10) {
          let t = P.h2 - r2;
          rho = rho + P.mass * P.poly6 * t * t * t;
          let r = sqrt(r2);
          let w1 = P.h - r;
          let gradW = (-P.spikyGrad * w1 * w1 / r) * d;
          gradSelf = gradSelf + gradW;
          gradSum = gradSum + dot(gradW, gradW) * invRho0 * invRho0;
        }
  `)}
  gradSum = gradSum + dot(gradSelf, gradSelf) * invRho0 * invRho0;
  let C = max(rho * invRho0 - 1.0, 0.0);
  lambda[i] = vec2f(-C / (gradSum + P.pbfRelax), rho);
}
`;

// x*_i += (1/rho0) sum (lambda_i + lambda_j + sCorr) grad_i W, then the soft
// boundary nudge (spring accel * dt^2, damping halved CPU-side) + hard clamp.
export const pbfDeltaPWGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> posVel: array<vec4f>;   // predicted (read)
@group(0) @binding(2) var<storage, read> cellStart: array<u32>;
@group(0) @binding(3) var<storage, read> cellCount: array<u32>;
@group(0) @binding(4) var<storage, read> sortedIdx: array<u32>;
@group(0) @binding(5) var<storage, read> lambda: array<vec2f>;
@group(0) @binding(6) var<storage, read_write> predOut: array<vec4f>;
@group(0) @binding(7) var<storage, read> obstacles: array<vec4f>;
${cellOf}
${boundaryFns}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(P.count)) { return; }
  let pv = posVel[i];
  var pos = pv.xy;
  var vel = pv.zw;
  let lambdaI = lambda[i].x;
  var dp = vec2f(0.0);
  ${gatherLoop(/* wgsl */ `
        let d = pos - posVel[j].xy;
        let r2 = dot(d, d);
        if (r2 < P.h2 && r2 > 1e-10) {
          let r = sqrt(r2);
          let w1 = P.h - r;
          let t = P.h2 - r2;
          let w = P.poly6 * t * t * t;
          let ratio = w * P.pbfInvWDq;
          let sCorr = -P.pbfSCorrK * ratio * ratio * ratio * ratio;
          let gradW = (-P.spikyGrad * w1 * w1 / r) * d;
          dp = dp + (lambdaI + lambda[j].x + sCorr) * gradW;
        }
  `)}
  pos = pos + dp / P.rho0;
  pos = pos + boundaryAccel(pos, vel, P.h) * P.dt * P.dt;
  ${collideChunk}
  predOut[i] = vec4f(pos, vel);
}
`;

// v = (x* - x_old)/dt, plus XSPH viscosity smoothing over neighbors.
export const pbfFinalizeWGSL = /* wgsl */ `
${PARAMS_STRUCT}
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read> posVel: array<vec4f>;      // final predicted
@group(0) @binding(2) var<storage, read> cellStart: array<u32>;
@group(0) @binding(3) var<storage, read> cellCount: array<u32>;
@group(0) @binding(4) var<storage, read> sortedIdx: array<u32>;
@group(0) @binding(5) var<storage, read> posVelOld: array<vec4f>;   // pre-step state
@group(0) @binding(6) var<storage, read_write> posVelOut: array<vec4f>;
${cellOf}
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= u32(P.count)) { return; }
  let pos = posVel[i].xy;
  var vel = (pos - posVelOld[i].xy) / P.dt;
  var xsph = vec2f(0.0);
  if (P.pbfXsph > 0.0) {
    ${gatherLoop(/* wgsl */ `
        let dpj = posVel[j].xy;
        let d = pos - dpj;
        let r2 = dot(d, d);
        if (r2 < P.h2) {
          let velJ = (dpj - posVelOld[j].xy) / P.dt;
          let t = P.h2 - r2;
          xsph = xsph + (P.mass / P.rho0) * P.poly6 * t * t * t * (velJ - vel);
        }
    `)}
  }
  vel = vel + P.pbfXsph * xsph;
  let sp = length(vel);
  if (sp > P.maxV) { vel = vel * (P.maxV / sp); }
  posVelOut[i] = vec4f(pos, vel);
}
`;

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

export const RENDER_STRUCT = /* wgsl */ `
struct RenderParams {
  world: vec2f,
  resolution: vec2f,
  halfSizePx: f32,
  invSpeedRef: f32,
  amp: f32,
  threshold: f32,
  invScale: f32,
  obstacleCount: f32,
  pad0: f32,
  pad1: f32,
}
`;

// Instanced-quad particle vertex shader (WebGPU has no point size).
export const particleQuadWGSL = /* wgsl */ `
${RENDER_STRUCT}
@group(0) @binding(0) var<uniform> R: RenderParams;
@group(0) @binding(1) var<storage, read> posVel: array<vec4f>;
struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
  @location(1) speed: f32,
  @location(2) vel: vec2f,   // normalized by speedRef (field G/B)
}
@vertex
fn vs(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> VSOut {
  let pv = posVel[ii];
  let corner = vec2f(f32(vi & 1u) * 2.0 - 1.0, f32(vi >> 1u) * 2.0 - 1.0);
  var clip = (pv.xy / R.world) * 2.0 - 1.0;
  clip.y = -clip.y;
  var out: VSOut;
  out.pos = vec4f(clip + corner * R.halfSizePx * 2.0 / R.resolution, 0.0, 1.0);
  out.uv = corner;
  out.vel = pv.zw * R.invSpeedRef;
  out.speed = length(out.vel);
  return out;
}

@fragment
fn fsDots(in: VSOut) -> @location(0) vec4f {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 1.0) { discard; }
  let a = smoothstep(1.0, 0.12, r2);
  let t = clamp(in.speed, 0.0, 1.0);
  let deep = vec3f(0.05, 0.18, 0.55);
  let mid = vec3f(0.10, 0.55, 0.95);
  let foam = vec3f(0.80, 0.96, 1.00);
  var col = select(mix(mid, foam, (t - 0.5) * 2.0), mix(deep, mid, t * 2.0), t < 0.5);
  col = col + (1.0 - smoothstep(0.0, 0.4, r2)) * 0.22;
  return vec4f(col, a * 0.35);
}

@fragment
fn fsField(in: VSOut) -> @location(0) vec4f {
  let u2 = dot(in.uv, in.uv);
  if (u2 > 1.0) { discard; }
  var w = 1.0 - u2;
  w = w * w * R.amp;
  return vec4f(w, w * in.vel.x, w * in.vel.y, w * in.speed);
}
`;

// Standalone obstacle overlay (dots mode; the composite pass draws its own
// copy in liquid mode). Alpha-blended fullscreen triangle over the sprites.
export const obstacleOverlayWGSL = /* wgsl */ `
${RENDER_STRUCT}
@group(0) @binding(0) var<uniform> R: RenderParams;
@group(0) @binding(1) var<storage, read> obstacles: array<vec4f>;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  var out: VSOut;
  out.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(xy.x, 1.0 - xy.y);
  return out;
}
@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let p = vec2f(in.uv.x * R.world.x, in.uv.y * R.world.y);
  var d = 1e9;
  for (var k = 0; k < i32(R.obstacleCount); k = k + 1) {
    d = min(d, length(p - obstacles[k].xy) - obstacles[k].z);
  }
  let px = R.world.y * 0.004;
  let fill = 1.0 - smoothstep(-px, px, d);
  if (fill <= 0.001) { discard; }
  var oc = vec3f(0.10, 0.13, 0.19);
  oc = oc + (1.0 - smoothstep(0.0, px * 6.0, abs(d))) * vec3f(0.16, 0.24, 0.38);
  return vec4f(oc, fill * 0.96);
}
`;

// Rigid-body sprite: one textured quad spanning the body type's local-unit
// rect, rotated + translated (mirrors the WebGL2 bodySpriteVS/FS).
export const bodySpriteWGSL = /* wgsl */ `
struct BodyDraw {
  pose: vec4f,   // x, y, cos(angle), sin(angle)
  rect: vec4f,   // local-unit quad: x0, y0 (top), x1, y1
  world: vec2f,
  scale: f32,
  pad0: f32,
}
@group(0) @binding(0) var<uniform> D: BodyDraw;
@group(0) @binding(1) var samp: sampler;
@group(0) @binding(2) var sprite: texture_2d<f32>;
struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  let uv = vec2f(f32(vi & 1u), f32((vi >> 1u) & 1u));
  let local = mix(D.rect.xy, D.rect.zw, uv);
  let p = D.pose.xy + vec2f(local.x * D.pose.z - local.y * D.pose.w,
                            local.x * D.pose.w + local.y * D.pose.z) * D.scale;
  var clip = (p / D.world) * 2.0 - 1.0;
  clip.y = -clip.y;
  var out: VSOut;
  out.pos = vec4f(clip, 0.0, 1.0);
  out.uv = uv;
  return out;
}
@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let c = textureSample(sprite, samp, in.uv);
  if (c.a < 0.004) { discard; }
  return c;
}
`;

export const compositeWGSL = /* wgsl */ `
${RENDER_STRUCT}
@group(0) @binding(0) var<uniform> R: RenderParams;
@group(0) @binding(1) var fieldTex: texture_2d<f32>;
@group(0) @binding(2) var fieldSamp: sampler;
@group(0) @binding(3) var<storage, read> obstacles: array<vec4f>;

struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
}
@vertex
fn vs(@builtin(vertex_index) vi: u32) -> VSOut {
  let xy = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u));
  var out: VSOut;
  out.pos = vec4f(xy * 2.0 - 1.0, 0.0, 1.0);
  out.uv = vec2f(xy.x, 1.0 - xy.y);
  return out;
}

fn background(uv: vec2f) -> vec3f {
  let top = vec3f(0.020, 0.028, 0.055);
  let bot = vec3f(0.045, 0.065, 0.110);
  var col = mix(top, bot, uv.y);
  let r = distance(uv, vec2f(0.5, 0.35));
  col = col + vec3f(0.010, 0.016, 0.030) * (1.0 - smoothstep(0.0, 0.8, r));
  return col;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  let f = textureSampleLevel(fieldTex, fieldSamp, in.uv, 0.0);
  let rho = f.r;
  let spd = clamp(f.a / max(rho, 1e-4), 0.0, 1.0);
  var col = background(in.uv);
  let T = R.threshold;
  let edge = smoothstep(T, T * 1.8, rho);

  if (edge > 0.001) {
    let texel = 1.0 / (R.resolution * 0.5);
    let gx = textureSampleLevel(fieldTex, fieldSamp, in.uv + vec2f(texel.x, 0.0), 0.0).r
           - textureSampleLevel(fieldTex, fieldSamp, in.uv - vec2f(texel.x, 0.0), 0.0).r;
    let gy = textureSampleLevel(fieldTex, fieldSamp, in.uv + vec2f(0.0, texel.y), 0.0).r
           - textureSampleLevel(fieldTex, fieldSamp, in.uv - vec2f(0.0, texel.y), 0.0).r;
    let n = normalize(vec3f(-gx * R.invScale * 1.8, -gy * R.invScale * 1.8, 1.0));
    let depth = clamp((rho - T) * R.invScale, 0.0, 1.0);
    var water = mix(vec3f(0.16, 0.55, 0.90), vec3f(0.010, 0.10, 0.34), sqrt(depth));
    let foam = smoothstep(0.55, 1.25, spd);
    water = mix(water, vec3f(0.62, 0.85, 1.00), foam * 0.6);
    let lightDir = normalize(vec3f(-0.35, -0.55, 0.75));
    let diff = max(dot(n, lightDir), 0.0);
    water = water * (0.78 + 0.30 * diff);
    let halfV = normalize(lightDir + vec3f(0.0, 0.0, 1.0));
    water = water + pow(max(dot(n, halfV), 0.0), 40.0) * 0.35;
    let rim = smoothstep(T, T * 1.35, rho) * (1.0 - smoothstep(T * 1.35, T * 2.6, rho));
    water = water + rim * vec3f(0.10, 0.18, 0.24);
    col = mix(col, water, edge);
  }

  // obstacle overlay
  let p = vec2f(in.uv.x * R.world.x, in.uv.y * R.world.y);
  var d = 1e9;
  for (var k = 0; k < i32(R.obstacleCount); k = k + 1) {
    d = min(d, length(p - obstacles[k].xy) - obstacles[k].z);
  }
  let px = R.world.y * 0.004;
  let fill = 1.0 - smoothstep(-px, px, d);
  if (fill > 0.001) {
    var oc = vec3f(0.10, 0.13, 0.19);
    oc = oc + (1.0 - smoothstep(0.0, px * 6.0, abs(d))) * vec3f(0.16, 0.24, 0.38);
    col = mix(col, oc, fill * 0.96);
  }
  return vec4f(col, 1.0);
}
`;
