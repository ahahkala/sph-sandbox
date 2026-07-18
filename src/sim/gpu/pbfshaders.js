// PBF (position-based fluids, Macklin & Müller 2013) GLSL. Runs at frame
// timesteps (dt = 1/60): predict positions, then a few Jacobi iterations of
// density-constraint projection, then derive velocities.
//
// Texture choreography per step (iterations forced odd, default 3):
//   A  = current state (x_old, v_old)
//   predict:  A -> P0                     (x*, v + g dt)
//   iterate:  lambda(Pk) -> L ; deltaP(Pk, L) -> Pk^1
//   finalize: (P_final, A) -> B           (x*, (x* - x_old)/dt + XSPH)
// where P0 = B and P1 = aux, so an odd iteration count lands P_final in aux
// and finalize can write B (the solver then swaps A/B as usual).

import { gatherCommon, collideCommon, gravityCommon } from "./shaders.js";

export const predictFS = `#version 300 es
precision highp float;
${gravityCommon}
uniform sampler2D uPosVel;
uniform int uTexWidth;
uniform int uCount;
uniform float uDt, uMaxV;
uniform vec4 uPointer;
out vec4 outPosVel;
void main() {
  ivec2 tc = ivec2(gl_FragCoord.xy);
  int i = tc.y * uTexWidth + tc.x;
  vec4 pv = texelFetch(uPosVel, tc, 0);
  if (i >= uCount) { outPosVel = pv; return; }
  vec2 pos = pv.xy;
  vec2 vel = pv.zw;
  vec2 acc = gravity(pos);
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
  outPosVel = vec4(pos + vel * uDt, vel);
}`;

// lambda_i = -C_i / (sum |grad C|^2 + eps),  C_i = rho_i/rho0 - 1 (clamped >= 0)
// out: (lambda, rho, 0, 0)
export const lambdaFS = `#version 300 es
precision highp float;
${gatherCommon}
uniform float uMass, uPoly6, uSpikyGrad, uRho0, uRelax;
out vec4 outLambda;
void main() {
  ivec2 tc = ivec2(gl_FragCoord.xy);
  int i = tc.y * uTexWidth + tc.x;
  if (i >= uCount) { outLambda = vec4(0.0); return; }
  vec2 pos = texelFetch(uPosVel, tc, 0).xy;
  ivec2 cc = clamp(ivec2(floor(pos / uCellSize)), ivec2(0), uGridDims - 1);

  float rho = uMass * uPoly6 * uH2 * uH2 * uH2;   // self term
  vec2 gradSelf = vec2(0.0);
  float gradSum = 0.0;
  float invRho0 = 1.0 / uRho0;

  for (int gy = -1; gy <= 1; gy++)
  for (int gx = -1; gx <= 1; gx++) {
    ivec2 c = cc + ivec2(gx, gy);
    if (c.x < 0 || c.y < 0 || c.x >= uGridDims.x || c.y >= uGridDims.y) continue;
    vec4 slotsA = texelFetch(uGridA, c, 0);
    vec4 slotsB = texelFetch(uGridB, c, 0);
    for (int s = 0; s < 8; s++) {
      float idxF = s < 4 ? slotsA[s] : slotsB[s - 4];
      if (idxF < 0.0) continue;
      int j = int(idxF + 0.5);
      if (j == i) continue;
      vec2 d = pos - texelFetch(uPosVel, texelOf(j), 0).xy;
      float r2 = dot(d, d);
      if (r2 < uH2 && r2 > 1e-10) {
        float t = uH2 - r2;
        rho += uMass * uPoly6 * t * t * t;
        float r = sqrt(r2);
        float w1 = uH - r;
        vec2 gradW = (-uSpikyGrad * w1 * w1 / r) * d;  // grad_i W_ij
        gradSelf += gradW;
        gradSum += dot(gradW, gradW) * invRho0 * invRho0;
      }
    }
  }
  gradSum += dot(gradSelf, gradSelf) * invRho0 * invRho0;
  float C = max(rho * invRho0 - 1.0, 0.0);
  float lambda = -C / (gradSum + uRelax);
  outLambda = vec4(lambda, rho, 0.0, 0.0);
}`;

// x*_i += (1/rho0) sum (lambda_i + lambda_j + sCorr) grad_i W ; then collide
export const deltaPFS = `#version 300 es
precision highp float;
${gatherCommon}
${collideCommon}
uniform sampler2D uLambda;
uniform float uSpikyGrad, uPoly6, uRho0, uSCorrK, uInvWDq;
uniform float uDt2;   // dt^2: converts the boundary spring accel to a position nudge
out vec4 outPosVel;
void main() {
  ivec2 tc = ivec2(gl_FragCoord.xy);
  int i = tc.y * uTexWidth + tc.x;
  vec4 pv = texelFetch(uPosVel, tc, 0);
  if (i >= uCount) { outPosVel = pv; return; }
  vec2 pos = pv.xy;
  vec2 vel = pv.zw;
  float lambdaI = texelFetch(uLambda, tc, 0).x;
  ivec2 cc = clamp(ivec2(floor(pos / uCellSize)), ivec2(0), uGridDims - 1);

  vec2 dp = vec2(0.0);
  for (int gy = -1; gy <= 1; gy++)
  for (int gx = -1; gx <= 1; gx++) {
    ivec2 c = cc + ivec2(gx, gy);
    if (c.x < 0 || c.y < 0 || c.x >= uGridDims.x || c.y >= uGridDims.y) continue;
    vec4 slotsA = texelFetch(uGridA, c, 0);
    vec4 slotsB = texelFetch(uGridB, c, 0);
    for (int s = 0; s < 8; s++) {
      float idxF = s < 4 ? slotsA[s] : slotsB[s - 4];
      if (idxF < 0.0) continue;
      int j = int(idxF + 0.5);
      if (j == i) continue;
      vec2 d = pos - texelFetch(uPosVel, texelOf(j), 0).xy;
      float r2 = dot(d, d);
      if (r2 < uH2 && r2 > 1e-10) {
        float r = sqrt(r2);
        float w1 = uH - r;
        float lambdaJ = texelFetch(uLambda, texelOf(j), 0).x;
        float t = uH2 - r2;
        float w = uPoly6 * t * t * t;
        float ratio = w * uInvWDq;
        float sCorr = -uSCorrK * ratio * ratio * ratio * ratio;
        vec2 gradW = (-uSpikyGrad * w1 * w1 / r) * d;
        dp += (lambdaI + lambdaJ + sCorr) * gradW;
      }
    }
  }
  pos += dp / uRho0;
  // soft boundary zone (position-space analogue of the force-mode spring):
  // smooth push-out over the last h so moving fluid never snaps onto the
  // hard clamp line — that snap is visible as edge jitter
  pos += boundaryAccel(pos, vel, uH) * uDt2;
  collide(pos, vel);
  outPosVel = vec4(pos, vel);
}`;

// v = (x* - x_old)/dt, plus XSPH viscosity smoothing over neighbors.
export const finalizeFS = `#version 300 es
precision highp float;
${gatherCommon}
uniform sampler2D uOldPosVel;   // pre-step state (x_old)
uniform float uMass, uPoly6, uRho0, uDt, uMaxV, uXsph;
out vec4 outPosVel;
void main() {
  ivec2 tc = ivec2(gl_FragCoord.xy);
  int i = tc.y * uTexWidth + tc.x;
  vec4 pv = texelFetch(uPosVel, tc, 0);      // final predicted (x*)
  if (i >= uCount) { outPosVel = pv; return; }
  vec2 pos = pv.xy;
  vec2 posOld = texelFetch(uOldPosVel, tc, 0).xy;
  vec2 vel = (pos - posOld) / uDt;
  ivec2 cc = clamp(ivec2(floor(pos / uCellSize)), ivec2(0), uGridDims - 1);

  vec2 xsph = vec2(0.0);
  if (uXsph > 0.0) {
    for (int gy = -1; gy <= 1; gy++)
    for (int gx = -1; gx <= 1; gx++) {
      ivec2 c = cc + ivec2(gx, gy);
      if (c.x < 0 || c.y < 0 || c.x >= uGridDims.x || c.y >= uGridDims.y) continue;
      vec4 slotsA = texelFetch(uGridA, c, 0);
      vec4 slotsB = texelFetch(uGridB, c, 0);
      for (int s = 0; s < 8; s++) {
        float idxF = s < 4 ? slotsA[s] : slotsB[s - 4];
        if (idxF < 0.0) continue;
        int j = int(idxF + 0.5);
        if (j == i) continue;
        ivec2 jt = texelOf(j);
        vec2 dpj = texelFetch(uPosVel, jt, 0).xy;
        vec2 d = pos - dpj;
        float r2 = dot(d, d);
        if (r2 < uH2) {
          vec2 velJ = (dpj - texelFetch(uOldPosVel, jt, 0).xy) / uDt;
          float t = uH2 - r2;
          xsph += (uMass / uRho0) * uPoly6 * t * t * t * (velJ - vel);
        }
      }
    }
  }
  vel += uXsph * xsph;
  float sp = length(vel);
  if (sp > uMaxV) vel *= uMaxV / sp;
  outPosVel = vec4(pos, vel);
}`;
