// Rendering GLSL (ES 3.00). Particle positions are always sourced from the
// posVel state texture (RGBA32F: x, y, vx, vy) via gl_VertexID, so the same
// path draws CPU- and GPU-simulated particles with zero readback.

export const particleVS = `#version 300 es
precision highp float;
uniform sampler2D uPosVel;
uniform int uTexWidth;
uniform vec2 uWorld;
uniform float uPointSize;
uniform float uInvSpeedRef;
out float vSpeed;
out vec2 vVel;   // velocity normalized by speedRef
void main() {
  ivec2 tc = ivec2(gl_VertexID % uTexWidth, gl_VertexID / uTexWidth);
  vec4 pv = texelFetch(uPosVel, tc, 0);
  vec2 clip = (pv.xy / uWorld) * 2.0 - 1.0;
  clip.y = -clip.y;               // world y-down -> clip y-up
  gl_Position = vec4(clip, 0.0, 1.0);
  gl_PointSize = uPointSize;
  vVel = pv.zw * uInvSpeedRef;
  vSpeed = length(vVel);
}`;

// --- dots mode: soft round sprites colored by speed -----------------------
export const dotsFS = `#version 300 es
precision mediump float;
in float vSpeed;
in vec2 vVel;
out vec4 fragColor;
vec3 palette(float t) {
  vec3 deep = vec3(0.05, 0.18, 0.55);
  vec3 mid  = vec3(0.10, 0.55, 0.95);
  vec3 foam = vec3(0.80, 0.96, 1.00);
  return t < 0.5 ? mix(deep, mid, t * 2.0) : mix(mid, foam, (t - 0.5) * 2.0);
}
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  float r2 = dot(d, d);
  if (r2 > 0.25) discard;
  float a = smoothstep(0.25, 0.03, r2);
  vec3 col = palette(clamp(vSpeed, 0.0, 1.0));
  col += (1.0 - smoothstep(0.0, 0.10, r2)) * 0.22;  // center highlight
  fragColor = vec4(col, a * 0.35);                  // dim: many sprites add up
}`;

// --- liquid mode pass 1: splat density + weighted velocity into a field ---
// R = density-ish weight, G/B = weight * velocity (normalized by speedRef),
// A = weight * speed. Additive blend accumulates; the composite pass
// divides by R to recover averages.
export const fieldFS = `#version 300 es
precision mediump float;
in float vSpeed;
in vec2 vVel;
uniform float uAmp;
out vec4 fragColor;
void main() {
  vec2 d = gl_PointCoord - vec2(0.5);
  float u2 = dot(d, d) * 4.0;      // 0 center -> 1 at sprite edge
  if (u2 > 1.0) discard;
  float w = (1.0 - u2);
  w = w * w * uAmp;
  fragColor = vec4(w, w * vVel.x, w * vVel.y, w * vSpeed);
}`;

// --- liquid mode pass 2: fullscreen composite ----------------------------
export const compositeVS = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aPos;
out vec2 vUV;
void main() {
  vUV = aPos * 0.5 + 0.5;
  gl_Position = vec4(aPos, 0.0, 1.0);
}`;

export const compositeFS = `#version 300 es
precision highp float;
in vec2 vUV;
uniform sampler2D uField;
uniform vec2 uTexel;       // 1 / field resolution
uniform float uThreshold;  // field value where the surface sits
uniform float uInvScale;   // 1 / (typical interior field value)
out vec4 fragColor;

vec3 background(vec2 uv) {
  vec3 top = vec3(0.020, 0.028, 0.055);
  vec3 bot = vec3(0.045, 0.065, 0.110);
  vec3 col = mix(top, bot, uv.y);
  float r = distance(uv, vec2(0.5, 0.35));
  col += vec3(0.010, 0.016, 0.030) * (1.0 - smoothstep(0.0, 0.8, r));
  return col;
}

void main() {
  vec4 f = texture(uField, vUV);
  float rho = f.r;
  float spd = clamp(f.a / max(rho, 1e-4), 0.0, 1.0);

  vec3 bg = background(vUV);
  float T = uThreshold;
  float edge = smoothstep(T, T * 1.8, rho);
  if (edge <= 0.001) { fragColor = vec4(bg, 1.0); return; }

  // pseudo-normal from the density gradient (4 taps)
  float gx = texture(uField, vUV + vec2(uTexel.x, 0.0)).r
           - texture(uField, vUV - vec2(uTexel.x, 0.0)).r;
  float gy = texture(uField, vUV + vec2(0.0, uTexel.y)).r
           - texture(uField, vUV - vec2(0.0, uTexel.y)).r;
  vec3 n = normalize(vec3(-gx * uInvScale * 1.8, -gy * uInvScale * 1.8, 1.0));

  // depth: how far inside the blob we are
  float depth = clamp((rho - T) * uInvScale, 0.0, 1.0);
  vec3 shallow = vec3(0.16, 0.55, 0.90);
  vec3 deep    = vec3(0.010, 0.10, 0.34);
  vec3 water = mix(shallow, deep, sqrt(depth));

  // fast fluid turns to foam
  float foam = smoothstep(0.55, 1.25, spd);
  water = mix(water, vec3(0.62, 0.85, 1.00), foam * 0.6);

  // lighting: soft diffuse + specular glint from upper-left
  vec3 lightDir = normalize(vec3(-0.35, -0.55, 0.75));
  float diff = max(dot(n, lightDir), 0.0);
  water *= 0.78 + 0.30 * diff;
  vec3 halfV = normalize(lightDir + vec3(0.0, 0.0, 1.0));
  float spec = pow(max(dot(n, halfV), 0.0), 40.0);
  water += spec * 0.35;

  // bright rim right at the surface
  float rim = smoothstep(T, T * 1.35, rho) * (1.0 - smoothstep(T * 1.35, T * 2.6, rho));
  water += rim * vec3(0.10, 0.18, 0.24);

  fragColor = vec4(mix(bg, water, edge), 1.0);
}`;

// --- duck mesh: rotate + translate the local-space triangle list ----------
export const duckVS = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aPos;
layout(location = 1) in vec3 aCol;
uniform vec2 uWorld;
uniform vec4 uPose;    // x, y, cos(angle), sin(angle)
uniform float uScale;
out vec3 vCol;
void main() {
  vec2 p = uPose.xy + vec2(aPos.x * uPose.z - aPos.y * uPose.w,
                           aPos.x * uPose.w + aPos.y * uPose.z) * uScale;
  vec2 clip = (p / uWorld) * 2.0 - 1.0;
  clip.y = -clip.y;
  gl_Position = vec4(clip, 0.0, 1.0);
  vCol = aCol;
}`;

export const duckFS = `#version 300 es
precision highp float;
in vec3 vCol;
out vec4 fragColor;
void main() { fragColor = vec4(vCol, 1.0); }`;

// --- obstacle overlay: analytic circle SDF drawn over the fluid ----------
export const obstacleFS = `#version 300 es
precision highp float;
in vec2 vUV;
uniform vec2 uWorld;
uniform sampler2D uObstacleTex; // one texel per obstacle, 256-wide rows: xy = center, z = radius
uniform int uObstacleCount;
out vec4 fragColor;
void main() {
  vec2 p = vec2(vUV.x * uWorld.x, (1.0 - vUV.y) * uWorld.y);
  float d = 1e9;
  for (int k = 0; k < uObstacleCount; k++) {
    vec4 ob = texelFetch(uObstacleTex, ivec2(k & 255, k >> 8), 0);
    d = min(d, length(p - ob.xy) - ob.z);
  }
  float px = uWorld.y * 0.004;                 // ~edge softness in world units
  float fill = 1.0 - smoothstep(-px, px, d);
  if (fill <= 0.001) discard;
  vec3 col = vec3(0.10, 0.13, 0.19);
  float rim = 1.0 - smoothstep(0.0, px * 6.0, abs(d));
  col += rim * vec3(0.16, 0.24, 0.38);
  fragColor = vec4(col, fill * 0.96);
}`;
