// Force-SPH simulation GLSL (ES 3.00). Particle state lives in a posVel
// texture (RGBA32F: x, y, vx, vy), one texel per particle.
//
// Neighbor search: a uniform grid texture pair (h/2 cells, 5x5 gather)
// stores up to 8 particle indices per cell (4 RGBA channels x 2 textures),
// built each substep by
// scattering particles as 1px points with stencil routing — every fragment
// increments the cell's stencil whether the test passes or not, so a
// fragment's pre-increment value is its arrival ordinal in the cell; pass k
// (stencil cleared first) keeps ordinal k only.

export const scatterVS = `#version 300 es
precision highp float;
uniform sampler2D uPosVel;
uniform int uTexWidth;
uniform vec2 uGridDims;    // cols, rows
uniform float uCellSize;
flat out float vIndex;
void main() {
  ivec2 tc = ivec2(gl_VertexID % uTexWidth, gl_VertexID / uTexWidth);
  vec2 pos = texelFetch(uPosVel, tc, 0).xy;
  vec2 cell = clamp(floor(pos / uCellSize), vec2(0.0), uGridDims - 1.0);
  vec2 ndc = (cell + 0.5) / uGridDims * 2.0 - 1.0;
  gl_Position = vec4(ndc, 0.0, 1.0);
  gl_PointSize = 1.0;
  vIndex = float(gl_VertexID);
}`;

export const scatterFS = `#version 300 es
precision highp float;
flat in float vIndex;
out vec4 fragColor;
void main() {
  fragColor = vec4(vIndex);   // colorMask routes it to one channel
}`;

// Shared helpers for the gather passes.
export const gatherCommon = `
uniform sampler2D uPosVel;
uniform sampler2D uGridA;
uniform sampler2D uGridB;
uniform int uTexWidth;
uniform int uCount;
uniform ivec2 uGridDims;
uniform float uCellSize;
uniform float uH;
uniform float uH2;

ivec2 texelOf(int i) { return ivec2(i % uTexWidth, i / uTexWidth); }
`;

// Gravity: straight down, or a constant-magnitude pull toward the world
// center. Shared by the force integrator and the PBF predict pass.
export const gravityCommon = `
uniform float uG;
uniform int uGravMode;    // 0 = down, 1 = toward uGravCenter
uniform vec2 uGravCenter; // world center
vec2 gravity(vec2 pos) {
  if (uGravMode == 0) return vec2(0.0, uG);
  vec2 d = uGravCenter - pos;
  float len = length(d);
  return len > 1e-4 ? (uG / len) * d : vec2(0.0);
}
`;

// Collision with wall box + user-drawn circle obstacles; shared by the
// force integrator and the PBF position update.
export const collideCommon = `
uniform vec2 uWorldSize;
uniform float uEps;
uniform float uDamp;
uniform sampler2D uObstacleTex; // one texel per obstacle, 256-wide rows: xy = center, z = radius
uniform int uObstacleCount;
uniform float uWallK;          // boundary spring stiffness (accel / unit depth)
uniform float uWallC;          // boundary normal damping (1 / s)

// Smooth spring-damper boundary: force ramps up over the last smoothing
// radius before a wall or obstacle, so resting particles never sit on the
// hard clamp line getting discontinuous velocity kicks (edge jitter).
// The spring only pushes out; damping only resists approach.
vec2 wallSpring(float pen, vec2 nrm, vec2 vel) {
  float vn = dot(vel, nrm);
  float mag = uWallK * pen - uWallC * min(vn, 0.0);
  return max(mag, 0.0) * nrm;
}

vec2 boundaryAccel(vec2 pos, vec2 vel, float margin) {
  vec2 acc = vec2(0.0);
  if (pos.x < margin)                { acc += wallSpring(margin - pos.x, vec2(1.0, 0.0), vel); }
  if (pos.x > uWorldSize.x - margin) { acc += wallSpring(pos.x - (uWorldSize.x - margin), vec2(-1.0, 0.0), vel); }
  if (pos.y < margin)                { acc += wallSpring(margin - pos.y, vec2(0.0, 1.0), vel); }
  if (pos.y > uWorldSize.y - margin) { acc += wallSpring(pos.y - (uWorldSize.y - margin), vec2(0.0, -1.0), vel); }
  for (int k = 0; k < uObstacleCount; k++) {
    vec4 ob = texelFetch(uObstacleTex, ivec2(k & 255, k >> 8), 0);
    vec2 d = pos - ob.xy;
    float dist = length(d);
    float pen = ob.z + margin - dist;
    if (pen > 0.0 && dist > 1e-4) {
      acc += wallSpring(pen, d / dist, vel);
    }
  }
  return acc;
}

void collide(inout vec2 pos, inout vec2 vel) {
  if (pos.x < uEps)                { pos.x = uEps;                if (vel.x < 0.0) vel.x *= -uDamp; }
  if (pos.x > uWorldSize.x - uEps) { pos.x = uWorldSize.x - uEps; if (vel.x > 0.0) vel.x *= -uDamp; }
  if (pos.y < uEps)                { pos.y = uEps;                if (vel.y < 0.0) vel.y *= -uDamp; }
  if (pos.y > uWorldSize.y - uEps) { pos.y = uWorldSize.y - uEps; if (vel.y > 0.0) vel.y *= -uDamp; }
  for (int k = 0; k < uObstacleCount; k++) {
    vec4 ob = texelFetch(uObstacleTex, ivec2(k & 255, k >> 8), 0);
    vec2 d = pos - ob.xy;
    float dist = length(d);
    float pen = ob.z + uEps - dist;
    if (pen > 0.0) {
      vec2 nrm = dist > 1e-4 ? d / dist : vec2(0.0, -1.0);
      pos += nrm * pen;
      float vn = dot(vel, nrm);
      if (vn < 0.0) vel -= (1.0 + uDamp) * vn * nrm;
    }
  }
}
`;

// Body reaction-force reduction: every particle recomputes its boundary
// spring against the body's hull circles and the negated force (Newton's
// third law) + torque about the body center are summed by drawing all
// particles as points onto a 1x1 float target with additive blending.
// Contributions are pre-scaled by uOutScale so the half-float fallback
// target can't saturate.
export const bodyForceVS = `#version 300 es
precision highp float;
uniform sampler2D uPosVel;
uniform int uTexWidth;
uniform int uCount;
uniform float uWallK, uWallC, uMargin, uMass, uOutScale;
uniform vec2 uBodyCenter;
uniform int uBodyCount;
uniform vec4 uBody[8];   // xy = center, z = radius
flat out vec4 vForce;
void main() {
  ivec2 tc = ivec2(gl_VertexID % uTexWidth, gl_VertexID / uTexWidth);
  vec4 pv = texelFetch(uPosVel, tc, 0);
  vec2 F = vec2(0.0);
  float contacts = 0.0;
  for (int k = 0; k < uBodyCount; k++) {
    vec2 d = pv.xy - uBody[k].xy;
    float dist = length(d);
    float pen = uBody[k].z + uMargin - dist;
    if (pen > 0.0 && dist > 1e-4) {
      vec2 nrm = d / dist;
      float vn = dot(pv.zw, nrm);
      float mag = max(uWallK * pen - uWallC * min(vn, 0.0), 0.0);
      F -= mag * uMass * nrm;   // reaction on the body
      contacts += 1.0;
    }
  }
  vec2 r = pv.xy - uBodyCenter;
  vForce = vec4(F, r.x * F.y - r.y * F.x, contacts) * uOutScale;
  gl_PointSize = 1.0;
  // non-contacting (and out-of-range) particles rasterize nothing
  gl_Position = (gl_VertexID < uCount && contacts > 0.0)
    ? vec4(0.0, 0.0, 0.0, 1.0) : vec4(2.0, 2.0, 0.0, 1.0);
}`;

export const bodyForceFS = `#version 300 es
precision highp float;
flat in vec4 vForce;
out vec4 fragColor;
void main() { fragColor = vForce; }`;

// Pass: density + pressure + vorticity per particle (MRT):
//   out0 = (rho, rhoNear, P, Pnear)     out1 = (omega, |omega|, 0, 0)
export const densityFS = `#version 300 es
precision highp float;
${gatherCommon}
uniform float uMass, uPoly6, uSpiky3, uSpikyGrad, uK, uKNear, uRho0;
layout(location = 0) out vec4 outDensity;
layout(location = 1) out vec4 outCurl;

void main() {
  ivec2 tc = ivec2(gl_FragCoord.xy);
  int i = tc.y * uTexWidth + tc.x;
  if (i >= uCount) { outDensity = vec4(0.0); outCurl = vec4(0.0); return; }
  vec4 pv = texelFetch(uPosVel, tc, 0);
  vec2 pos = pv.xy;
  vec2 vel = pv.zw;
  ivec2 cc = clamp(ivec2(floor(pos / uCellSize)), ivec2(0), uGridDims - 1);

  // self-contribution up front: correct even if this particle lost the
  // fight for a grid slot in an overcrowded cell
  float rho = uMass * uPoly6 * uH2 * uH2 * uH2;
  float rhoNear = uMass * uSpiky3 * uH * uH * uH;
  float omega = 0.0;

  for (int gy = -2; gy <= 2; gy++)
  for (int gx = -2; gx <= 2; gx++) {
    ivec2 c = cc + ivec2(gx, gy);
    if (c.x < 0 || c.y < 0 || c.x >= uGridDims.x || c.y >= uGridDims.y) continue;
    vec4 slotsA = texelFetch(uGridA, c, 0);
    vec4 slotsB = texelFetch(uGridB, c, 0);
    for (int s = 0; s < 8; s++) {
      float idxF = s < 4 ? slotsA[s] : slotsB[s - 4];
      if (idxF < 0.0) continue;
      int j = int(idxF + 0.5);
      if (j == i) continue;
      vec4 pvj = texelFetch(uPosVel, texelOf(j), 0);
      vec2 d = pos - pvj.xy;
      float r2 = dot(d, d);
      if (r2 < uH2 && r2 > 1e-10) {
        float t = uH2 - r2;
        rho += uMass * uPoly6 * t * t * t;
        float r = sqrt(r2);
        float w1 = uH - r;
        rhoNear += uMass * uSpiky3 * w1 * w1 * w1;
        // 2D curl: omega += (m/rho0) (v_j - v_i) x gradW_i,
        // gradW_i = -uSpikyGrad w1^2 d/r
        vec2 gradW = (-uSpikyGrad * w1 * w1 / r) * d;
        vec2 dv = pvj.zw - vel;
        omega += (uMass / uRho0) * (dv.x * gradW.y - dv.y * gradW.x);
      }
    }
  }
  float p = max(uK * (rho - uRho0), 0.0);
  outDensity = vec4(rho, rhoNear, p, uKNear * rhoNear);
  outCurl = vec4(omega, abs(omega), 0.0, 0.0);
}`;

// Pass: forces + integration -> new (x, y, vx, vy). Symmetric pressure +
// near-pressure + viscosity + vorticity confinement, gravity, pointer
// force, velocity clamp, wall/obstacle collision.
export const forceFS = `#version 300 es
precision highp float;
${gatherCommon}
${collideCommon}
${gravityCommon}
uniform sampler2D uDensity;
uniform sampler2D uCurl;
uniform float uMass, uSpikyGrad, uViscLap, uMu, uVort, uDt, uMaxV;
uniform vec4 uPointer;   // x, y, signed strength (0 = inactive), radius
out vec4 outPosVel;

void main() {
  ivec2 tc = ivec2(gl_FragCoord.xy);
  int i = tc.y * uTexWidth + tc.x;
  vec4 pv = texelFetch(uPosVel, tc, 0);
  if (i >= uCount) { outPosVel = pv; return; }
  vec2 pos = pv.xy;
  vec2 vel = pv.zw;
  vec4 di = texelFetch(uDensity, tc, 0);   // rho, rhoNear, P, Pnear
  float omegaI = texelFetch(uCurl, tc, 0).x;
  ivec2 cc = clamp(ivec2(floor(pos / uCellSize)), ivec2(0), uGridDims - 1);

  vec2 f = vec2(0.0);
  vec2 gradAbsOmega = vec2(0.0);
  for (int gy = -2; gy <= 2; gy++)
  for (int gx = -2; gx <= 2; gx++) {
    ivec2 c = cc + ivec2(gx, gy);
    if (c.x < 0 || c.y < 0 || c.x >= uGridDims.x || c.y >= uGridDims.y) continue;
    vec4 slotsA = texelFetch(uGridA, c, 0);
    vec4 slotsB = texelFetch(uGridB, c, 0);
    for (int s = 0; s < 8; s++) {
      float idxF = s < 4 ? slotsA[s] : slotsB[s - 4];
      if (idxF < 0.0) continue;
      int j = int(idxF + 0.5);
      if (j == i) continue;
      vec4 pvj = texelFetch(uPosVel, texelOf(j), 0);
      vec2 d = pos - pvj.xy;
      float r2 = dot(d, d);
      if (r2 < uH2 && r2 > 1e-10) {
        float r = sqrt(r2);
        float w1 = uH - r;
        vec4 dj = texelFetch(uDensity, texelOf(j), 0);
        float invRhoB = 1.0 / max(dj.x, 1e-6);
        float coef = uMass * uSpikyGrad * w1 * w1 * invRhoB / r *
                     (0.5 * (di.z + dj.z) + 0.5 * (di.w + dj.w) * w1);
        f += coef * d;
        f += uMu * uMass * uViscLap * w1 * invRhoB * (pvj.zw - vel);
        // gradient of |omega| for vorticity confinement
        float absOmegaJ = texelFetch(uCurl, texelOf(j), 0).y;
        gradAbsOmega += (uMass * invRhoB * absOmegaJ * -uSpikyGrad * w1 * w1 / r) * d;
      }
    }
  }

  float invRho = 1.0 / max(di.x, 1e-6);
  vec2 acc = f * invRho + gravity(pos);
  acc += boundaryAccel(pos, vel, uH);

  // vorticity confinement: eps * (N-hat x omega z-hat)
  float nLen = length(gradAbsOmega);
  if (uVort > 0.0 && nLen > 1e-6) {
    vec2 nHat = gradAbsOmega / nLen;
    acc += uVort * omegaI * vec2(nHat.y, -nHat.x);
  }

  if (uPointer.z != 0.0) {
    vec2 d = pos - uPointer.xy;
    float r2 = dot(d, d);
    float R = uPointer.w;
    if (r2 < R * R && r2 > 1e-4) {
      float r = sqrt(r2);
      acc += uPointer.z * (1.0 - r / R) / r * d;
    }
  }

  vel += acc * uDt;
  float sp = length(vel);
  if (sp > uMaxV) vel *= uMaxV / sp;
  pos += vel * uDt;
  collide(pos, vel);

  outPosVel = vec4(pos, vel);
}`;
