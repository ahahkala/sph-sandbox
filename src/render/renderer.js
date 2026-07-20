// WebGL2 renderer. Particle positions are read straight from the solver's
// posVel state texture in the vertex shader (gl_VertexID indexing) — no
// readback anywhere in the frame.
//
// Modes:
//   "liquid" — screen-space metaballs: splat a half-res density field,
//              then threshold + shade it into a lit surface
//   "dots"   — additive soft sprites colored by speed

import { createProgram, getUniforms, createTarget, deleteTarget, ObstacleTexture } from "./glutils.js";
import {
  particleVS, dotsFS, fieldFS, compositeVS, compositeFS, obstacleFS,
  bodySpriteVS, bodySpriteFS,
} from "./shaders.js";
import { TEX_WIDTH } from "../config.js";
import { BODY_TYPES, bodyType } from "../sim/bodytypes.js";

const FIELD_SCALE = 0.5;  // field texture resolution relative to canvas

export class Renderer {
  constructor(canvas, params) {
    this.canvas = canvas;
    this.params = params;
    const gl = canvas.getContext("webgl2", { alpha: false, antialias: false, premultipliedAlpha: false });
    if (!gl) throw new Error("WebGL2 not supported");
    this.gl = gl;

    this.floatColor = !!gl.getExtension("EXT_color_buffer_float");
    this.fieldAmp = this.floatColor ? 0.9 : 0.16; // byte fields saturate: keep splats dim
    this.maxPointSize = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE)[1];

    this.progDots = createProgram(gl, particleVS, dotsFS);
    this.uDots = getUniforms(gl, this.progDots);
    this.progField = createProgram(gl, particleVS, fieldFS);
    this.uField = getUniforms(gl, this.progField);
    this.progComposite = createProgram(gl, compositeVS, compositeFS);
    this.uComposite = getUniforms(gl, this.progComposite);
    this.progObstacle = createProgram(gl, compositeVS, obstacleFS);
    this.uObstacle = getUniforms(gl, this.progObstacle);
    this.obstacleTex = new ObstacleTexture(gl);
    this.progBody = createProgram(gl, bodySpriteVS, bodySpriteFS);
    this.uBody = getUniforms(gl, this.progBody);
    this.sprites = loadSprites(gl);

    // one VAO with the fullscreen quad, one empty VAO for attribute-less
    // particle draws (gl_VertexID) — leaving quad attributes enabled would
    // make large point draws fail with out-of-range attribute access
    this.quadVAO = gl.createVertexArray();
    gl.bindVertexArray(this.quadVAO);
    this.quadVBO = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.emptyVAO = gl.createVertexArray();
    gl.bindVertexArray(null);

    this.field = null;      // offscreen splat target, (re)built in resize()

    gl.disable(gl.DEPTH_TEST);
  }

  resize(pixelW, pixelH) {
    const gl = this.gl;
    this.canvas.width = pixelW;
    this.canvas.height = pixelH;

    deleteTarget(gl, this.field);
    const fw = Math.max(2, Math.round(pixelW * FIELD_SCALE));
    const fh = Math.max(2, Math.round(pixelH * FIELD_SCALE));
    this.field = this.floatColor
      ? createTarget(gl, fw, fh, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.LINEAR)
      : null;
    if (!this.field) {
      this.floatColor = false;
      this.fieldAmp = 0.16;
      this.field = createTarget(gl, fw, fh, gl.RGBA8, gl.RGBA, gl.UNSIGNED_BYTE, gl.LINEAR);
    }
  }

  render(posVelTex, count, world) {
    if (this.params.renderMode === "liquid") {
      this.renderLiquid(posVelTex, count, world);
    } else {
      this.renderDots(posVelTex, count, world);
      this.renderObstacles(world);
    }
    this.renderBodies(world);
  }

  setParticleUniforms(u, posVelTex, world, pointSize) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, posVelTex);
    gl.uniform1i(u.uPosVel, 0);
    gl.uniform1i(u.uTexWidth, TEX_WIDTH);
    gl.uniform2f(u.uWorld, world.w, world.h);
    gl.uniform1f(u.uPointSize, pointSize);
    gl.uniform1f(u.uInvSpeedRef, 1 / this.params.speedRef);
  }

  renderDots(posVelTex, count, world) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.clearColor(0.02, 0.028, 0.055, 1.0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.progDots);
    gl.bindVertexArray(this.emptyVAO);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
    const pxPerWorld = this.canvas.height / world.h;
    const size = Math.min(this.maxPointSize, Math.max(2, this.params.h * pxPerWorld * 1.3));
    this.setParticleUniforms(this.uDots, posVelTex, world, size);
    gl.drawArrays(gl.POINTS, 0, count);
    gl.disable(gl.BLEND);
  }

  renderLiquid(posVelTex, count, world) {
    const gl = this.gl;
    const field = this.field;

    // pass 1: splat particles into the density field
    gl.bindFramebuffer(gl.FRAMEBUFFER, field.fbo);
    gl.viewport(0, 0, field.width, field.height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.progField);
    gl.bindVertexArray(this.emptyVAO);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    // splat diameter ≈ 3 h so neighboring particles merge into one surface;
    // at huge counts splats shrink below a few pixels and the surface gets
    // grainy, so enforce a minimum screen-space size
    const size = Math.min(this.maxPointSize,
      Math.max(4.5, this.params.h * 3.0 * (field.height / world.h)));
    this.setParticleUniforms(this.uField, posVelTex, world, size);
    gl.uniform1f(this.uField.uAmp, this.fieldAmp);
    gl.drawArrays(gl.POINTS, 0, count);
    gl.disable(gl.BLEND);

    // pass 2: composite the field into a lit surface
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.progComposite);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, field.tex);
    gl.uniform1i(this.uComposite.uField, 0);
    gl.uniform2f(this.uComposite.uTexel, 1 / field.width, 1 / field.height);
    const interior = this.fieldAmp * (this.floatColor ? 3.0 : 3.5);
    gl.uniform1f(this.uComposite.uThreshold, interior * 0.18);
    gl.uniform1f(this.uComposite.uInvScale, 1 / interior);

    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);

    this.renderObstacles(world);
  }

  renderObstacles(world) {
    const obstacles = this.params.obstacles;
    if (!obstacles.length) return;
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.progObstacle);
    gl.uniform2f(this.uObstacle.uWorld, world.w, world.h);
    this.obstacleTex.upload(obstacles, 0);
    gl.uniform1i(this.uObstacle.uObstacleTex, 0);
    gl.uniform1i(this.uObstacle.uObstacleCount, obstacles.length);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
  }

  renderBodies(world) {
    const poses = this.params.bodyPoses;
    if (poses.length === 0) return;
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.canvas.width, this.canvas.height);
    gl.useProgram(this.progBody);
    gl.uniform2f(this.uBody.uWorld, world.w, world.h);
    gl.uniform1i(this.uBody.uSprite, 0);
    gl.activeTexture(gl.TEXTURE0);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.bindVertexArray(this.emptyVAO);
    for (const pose of poses) {
      const sprite = this.sprites[pose.type];
      if (!sprite || !sprite.ready) continue; // image still loading
      const r = bodyType(pose.type).rect;
      gl.bindTexture(gl.TEXTURE_2D, sprite.tex);
      gl.uniform4f(this.uBody.uPose, pose.x, pose.y, Math.cos(pose.angle), Math.sin(pose.angle));
      gl.uniform1f(this.uBody.uScale, pose.scale);
      gl.uniform4f(this.uBody.uRect, r.x0, r.y0, r.x1, r.y1);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    }
    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
  }
}

// One texture per body type. The PNGs load asynchronously; a body whose
// sprite hasn't arrived yet is simply skipped for a frame or two.
function loadSprites(gl) {
  const out = {};
  for (const [name, type] of Object.entries(BODY_TYPES)) {
    const tex = gl.createTexture();
    const slot = { tex, ready: false };
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    const img = new Image();
    img.onload = () => {
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
      slot.ready = true;
    };
    img.src = type.sprite;
    out[name] = slot;
  }
  return out;
}
