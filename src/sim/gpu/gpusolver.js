// GPU SPH solver (WebGL2). Two physics modes over one grid pipeline:
//
//  "force" (Müller SPH, dt = 1/240):
//    grid scatter -> density+curl (MRT) -> force+integrate (ping-pong posVel)
//  "pbf" (position-based fluids, dt = 1/60):
//    predict -> grid scatter -> [lambda -> deltaP] x iters -> finalize
//
// Particle state never leaves the GPU; the renderer reads posVelTexture()
// directly. See shaders.js / pbfshaders.js for the pass details.

import { createProgram, getUniforms, createTarget, deleteTarget, attachDepthStencil, createTexture, ObstacleTexture } from "../../render/glutils.js";
import { scatterVS, scatterFS, densityFS, forceFS, bodyForceVS, bodyForceFS } from "./shaders.js";
import { predictFS, lambdaFS, deltaPFS, finalizeFS } from "./pbfshaders.js";
import { TEX_WIDTH, WORLD_HEIGHT as REF_WORLD_H } from "../../config.js";

const quadVS = `#version 300 es
precision highp float;
layout(location = 0) in vec2 aPos;
void main() { gl_Position = vec4(aPos, 0.0, 1.0); }`;

export class GPUSolver {
  constructor(gl, world, params) {
    if (!(gl instanceof WebGL2RenderingContext)) throw new Error("GPUSolver needs WebGL2");
    if (!gl.getExtension("EXT_color_buffer_float")) {
      throw new Error("EXT_color_buffer_float unsupported");
    }
    this.gl = gl;
    this.world = world;
    this.params = params;
    this.maxCount = TEX_WIDTH * TEX_WIDTH;

    const h = params.h;
    this.h = h;
    // Grid cells are h/2 with a 5x5 gather (not h with 3x3): the 8-slot cap
    // per cell then only overflows near 10x rest density instead of ~2.4x,
    // which keeps neighborhoods complete (and forces symmetric) under
    // compression — dropped neighbors read as jitter vs the WebGPU exact grid.
    this.cellSize = h / 2;
    this.kern = {
      poly6: 4 / (Math.PI * h ** 8),
      spiky3: 10 / (Math.PI * h ** 5),
      spikyGrad: 30 / (Math.PI * h ** 5),
      viscLap: 40 / (Math.PI * h ** 5),
    };

    const progs = {
      scatter: [scatterVS, scatterFS],
      density: [quadVS, densityFS],
      force: [quadVS, forceFS],
      predict: [quadVS, predictFS],
      lambda: [quadVS, lambdaFS],
      deltaP: [quadVS, deltaPFS],
      finalize: [quadVS, finalizeFS],
      bodyForce: [bodyForceVS, bodyForceFS],
    };
    this.prog = {};
    this.u = {};
    for (const [name, [vs, fs]] of Object.entries(progs)) {
      this.prog[name] = createProgram(gl, vs, fs);
      this.u[name] = getUniforms(gl, this.prog[name]);
    }

    // quad VAO for gather passes, empty VAO for attribute-less scatter draws
    this.quadVAO = gl.createVertexArray();
    gl.bindVertexArray(this.quadVAO);
    this.quadVBO = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.quadVBO);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.emptyVAO = gl.createVertexArray();
    gl.bindVertexArray(null);

    this.pointer = { x: 0, y: 0, active: false, mode: 1 };
    this.obstacleTex = new ObstacleTexture(gl);

    // 1x1 accumulator for the body-force reduction; RGBA32F blending needs
    // EXT_float_blend, otherwise fall back to half float (scaled output)
    this.floatBlend = !!gl.getExtension("EXT_float_blend");
    this.bodyTarget = this.floatBlend
      ? createTarget(gl, 1, 1, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST)
      : createTarget(gl, 1, 1, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT, gl.NEAREST);
    this.bodyPixel = new Float32Array(4);
    this.bodyData = new Float32Array(8 * 4);

    this.posVel = [null, null]; // ping-pong state
    this.aux = null;            // PBF iteration scratch
    this.cur = 0;
    this.density = null;        // MRT target 0 (also PBF lambda)
    this.curlTex = null;        // MRT target 1
    this.gridA = null;
    this.gridB = null;
    this.count = 0;
    this.texHeight = 0;
    this.gridCols = 0;
    this.gridRows = 0;

    this.alloc(params.count);
    this.resizeGrid();
  }

  alloc(n) {
    const gl = this.gl;
    if (n > this.maxCount) n = this.maxCount;
    this.count = n;
    const texH = Math.max(1, Math.ceil(n / TEX_WIDTH));
    if (texH === this.texHeight) return;
    this.texHeight = texH;
    for (const t of this.posVel) deleteTarget(gl, t);
    deleteTarget(gl, this.aux);
    deleteTarget(gl, this.density);
    if (this.curlTex) gl.deleteTexture(this.curlTex);

    const mk = () => createTarget(gl, TEX_WIDTH, texH, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST);
    this.posVel = [mk(), mk()];
    this.aux = mk();
    this.density = mk();
    if (!this.posVel[0] || !this.posVel[1] || !this.aux || !this.density) {
      throw new Error("float render targets unsupported");
    }
    // second MRT attachment on the density FBO for the curl field
    this.curlTex = createTexture(gl, TEX_WIDTH, texH, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.density.fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT1, gl.TEXTURE_2D, this.curlTex, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  resizeGrid() {
    const gl = this.gl;
    const cols = Math.max(1, Math.ceil(this.world.w / this.cellSize) + 1);
    const rows = Math.max(1, Math.ceil(this.world.h / this.cellSize) + 1);
    if (cols === this.gridCols && rows === this.gridRows) return;
    this.gridCols = cols;
    this.gridRows = rows;
    deleteTarget(gl, this.gridA);
    deleteTarget(gl, this.gridB);
    // gridA owns the shared depth-stencil buffer; gridB borrows it
    this.gridA = createTarget(gl, cols, rows, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST, true);
    this.gridB = createTarget(gl, cols, rows, gl.RGBA32F, gl.RGBA, gl.FLOAT, gl.NEAREST);
    if (!this.gridA || !this.gridB) throw new Error("grid render targets unsupported");
    attachDepthStencil(gl, this.gridB, this.gridA.rb);
  }

  setWorld(w, h) {
    this.world.w = w;
    this.world.h = h;
    this.resizeGrid();
  }

  worldScale() {
    return this.world.h / REF_WORLD_H;
  }

  // Upload initial particle state from CPU arrays ({px, py, vx, vy}).
  seed(state) {
    const gl = this.gl;
    const n = this.count;
    const data = new Float32Array(TEX_WIDTH * this.texHeight * 4);
    for (let i = 0; i < n; i++) {
      data[i * 4] = state.px[i];
      data[i * 4 + 1] = state.py[i];
      data[i * 4 + 2] = state.vx[i];
      data[i * 4 + 3] = state.vy[i];
    }
    for (const t of this.posVel) {
      gl.bindTexture(gl.TEXTURE_2D, t.tex);
      gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, TEX_WIDTH, this.texHeight, gl.RGBA, gl.FLOAT, data);
    }
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  posVelTexture() {
    return this.posVel[this.cur].tex;
  }

  step(dt) {
    if (this.params.physics === "pbf") this.stepPBF(dt);
    else this.stepForce(dt);
  }

  stepForce(dt) {
    this.buildGrid(this.posVel[this.cur].tex);
    this.densityPass(this.posVel[this.cur].tex);
    this.forcePass(dt);
    this.cur = 1 - this.cur;
  }

  stepPBF(dt) {
    const gl = this.gl;
    const p = this.params;
    const A = this.posVel[this.cur];
    const B = this.posVel[1 - this.cur];
    const iters = Math.max(1, Math.round(p.pbfIterations) | 1); // force odd

    // predict: A -> B
    this.fragPass("predict", B, (u) => {
      this.bindTex(u.uPosVel, 0, A.tex);
      gl.uniform1i(u.uTexWidth, TEX_WIDTH);
      gl.uniform1i(u.uCount, this.count);
      this.setGravityUniforms(u);
      gl.uniform1f(u.uDt, dt);
      gl.uniform1f(u.uMaxV, p.maxSpeed);
      this.setPointerUniform(u);
    });

    this.buildGrid(B.tex);

    // constraint iterations ping-pong B <-> aux; odd count ends in aux
    let src = B, dst = this.aux;
    for (let it = 0; it < iters; it++) {
      this.fragPass("lambda", this.density, (u) => {
        this.setGatherUniforms(u, src.tex);
        gl.uniform1f(u.uMass, p.mass);
        gl.uniform1f(u.uPoly6, this.kern.poly6);
        gl.uniform1f(u.uSpikyGrad, this.kern.spikyGrad);
        gl.uniform1f(u.uRho0, p.restDensity);
        gl.uniform1f(u.uRelax, p.pbfRelax);
      });
      this.fragPass("deltaP", dst, (u) => {
        this.setGatherUniforms(u, src.tex);
        this.bindTex(u.uLambda, 3, this.density.tex);
        gl.uniform1f(u.uSpikyGrad, this.kern.spikyGrad);
        gl.uniform1f(u.uPoly6, this.kern.poly6);
        gl.uniform1f(u.uRho0, p.restDensity);
        gl.uniform1f(u.uSCorrK, p.pbfSCorrK);
        const dq = p.pbfSCorrDq * this.h;
        const wDq = this.kern.poly6 * Math.pow(this.h * this.h - dq * dq, 3);
        gl.uniform1f(u.uInvWDq, 1 / wDq);
        gl.uniform1f(u.uDt2, dt * dt);
        this.setCollideUniforms(u);
        // the nudge is applied once per iteration, which multiplies the
        // effective damping; half of wallDamping measures calmest here
        gl.uniform1f(u.uWallC, p.wallDamping * 0.5);
      });
      [src, dst] = [dst, src];
    }

    // finalize: (src = aux after odd iters, A) -> B
    this.fragPass("finalize", B, (u) => {
      this.setGatherUniforms(u, src.tex);
      this.bindTex(u.uOldPosVel, 3, A.tex);
      gl.uniform1f(u.uMass, p.mass);
      gl.uniform1f(u.uPoly6, this.kern.poly6);
      gl.uniform1f(u.uRho0, p.restDensity);
      gl.uniform1f(u.uDt, dt);
      gl.uniform1f(u.uMaxV, p.maxSpeed);
      gl.uniform1f(u.uXsph, p.pbfXsph * p.viscosity);
    });
    this.cur = 1 - this.cur;
  }

  // --- pass plumbing ------------------------------------------------------

  fragPass(name, target, setUniforms) {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
    gl.drawBuffers(name === "density"
      ? [gl.COLOR_ATTACHMENT0, gl.COLOR_ATTACHMENT1]
      : [gl.COLOR_ATTACHMENT0]);
    gl.viewport(0, 0, TEX_WIDTH, this.texHeight);
    gl.useProgram(this.prog[name]);
    setUniforms(this.u[name]);
    gl.bindVertexArray(this.quadVAO);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  bindTex(loc, unit, tex) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.uniform1i(loc, unit);
  }

  setGatherUniforms(u, posVelTex) {
    const gl = this.gl;
    this.bindTex(u.uPosVel, 0, posVelTex);
    this.bindTex(u.uGridA, 1, this.gridA.tex);
    this.bindTex(u.uGridB, 2, this.gridB.tex);
    gl.uniform1i(u.uTexWidth, TEX_WIDTH);
    gl.uniform1i(u.uCount, this.count);
    gl.uniform2i(u.uGridDims, this.gridCols, this.gridRows);
    gl.uniform1f(u.uCellSize, this.cellSize);
    gl.uniform1f(u.uH, this.h);
    gl.uniform1f(u.uH2, this.h * this.h);
  }

  setGravityUniforms(u) {
    const gl = this.gl;
    gl.uniform1f(u.uG, this.params.gravity / this.worldScale());
    gl.uniform1i(u.uGravMode, this.params.gravityCenter ? 1 : 0);
    gl.uniform2f(u.uGravCenter, this.world.w * 0.5, this.world.h * 0.5);
  }

  setCollideUniforms(u) {
    const gl = this.gl;
    const body = this.params.bodyCircles;
    const obstacles = body.length
      ? this.params.obstacles.concat(body)
      : this.params.obstacles;
    this.obstacleTex.upload(obstacles, 7);
    gl.uniform1i(u.uObstacleTex, 7);
    gl.uniform1i(u.uObstacleCount, obstacles.length);
    gl.uniform2f(u.uWorldSize, this.world.w, this.world.h);
    gl.uniform1f(u.uEps, this.h * 0.5);
    gl.uniform1f(u.uDamp, this.params.boundaryDamp);
    if (u.uWallK) gl.uniform1f(u.uWallK, this.params.wallStiffness);
    if (u.uWallC) gl.uniform1f(u.uWallC, this.params.wallDamping);
  }

  setPointerUniform(u) {
    const ptr = this.pointer;
    this.gl.uniform4f(u.uPointer,
      ptr.x, ptr.y,
      ptr.active ? ptr.mode * this.params.pointerStrength : 0,
      this.params.pointerRadius * this.worldScale());
  }

  buildGrid(posVelTex) {
    const gl = this.gl;
    gl.disable(gl.BLEND);
    gl.viewport(0, 0, this.gridCols, this.gridRows);

    gl.clearColor(-1, -1, -1, -1);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.gridA.fbo);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.gridB.fbo);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.prog.scatter);
    gl.bindVertexArray(this.emptyVAO);
    this.bindTex(this.u.scatter.uPosVel, 0, posVelTex);
    gl.uniform1i(this.u.scatter.uTexWidth, TEX_WIDTH);
    gl.uniform2f(this.u.scatter.uGridDims, this.gridCols, this.gridRows);
    gl.uniform1f(this.u.scatter.uCellSize, this.cellSize);

    // stencil routing (see header comment in shaders.js)
    gl.enable(gl.STENCIL_TEST);
    gl.stencilOp(gl.INCR, gl.INCR, gl.INCR);
    gl.clearStencil(0);
    for (let slot = 0; slot < 8; slot++) {
      const target = slot < 4 ? this.gridA : this.gridB;
      gl.bindFramebuffer(gl.FRAMEBUFFER, target.fbo);
      gl.clear(gl.STENCIL_BUFFER_BIT);
      const ch = slot % 4;
      gl.colorMask(ch === 0, ch === 1, ch === 2, ch === 3);
      gl.stencilFunc(gl.EQUAL, slot, 0xff);
      gl.drawArrays(gl.POINTS, 0, this.count);
    }
    gl.disable(gl.STENCIL_TEST);
    gl.colorMask(true, true, true, true);
    gl.bindVertexArray(null);
  }

  densityPass(posVelTex) {
    const gl = this.gl;
    const p = this.params;
    this.fragPass("density", this.density, (u) => {
      this.setGatherUniforms(u, posVelTex);
      gl.uniform1f(u.uMass, p.mass);
      gl.uniform1f(u.uPoly6, this.kern.poly6);
      gl.uniform1f(u.uSpiky3, this.kern.spiky3);
      gl.uniform1f(u.uSpikyGrad, this.kern.spikyGrad);
      gl.uniform1f(u.uK, p.stiffness);
      gl.uniform1f(u.uKNear, p.nearStiffness);
      gl.uniform1f(u.uRho0, p.restDensity);
    });
  }

  forcePass(dt) {
    const gl = this.gl;
    const p = this.params;
    this.fragPass("force", this.posVel[1 - this.cur], (u) => {
      this.setGatherUniforms(u, this.posVel[this.cur].tex);
      this.bindTex(u.uDensity, 3, this.density.tex);
      this.bindTex(u.uCurl, 4, this.curlTex);
      gl.uniform1f(u.uMass, p.mass);
      gl.uniform1f(u.uSpikyGrad, this.kern.spikyGrad);
      gl.uniform1f(u.uViscLap, this.kern.viscLap);
      gl.uniform1f(u.uMu, p.viscosity);
      gl.uniform1f(u.uVort, p.vorticity);
      this.setGravityUniforms(u);
      gl.uniform1f(u.uDt, dt);
      gl.uniform1f(u.uMaxV, p.maxSpeed);
      this.setPointerUniform(u);
      this.setCollideUniforms(u);
    });
  }

  // Sum the fluid's reaction to the body hull's boundary springs over all
  // particles (see bodyForceVS). One tiny sync readback per frame.
  sampleBodyForce(circles, cx, cy) {
    const gl = this.gl;
    const p = this.params;
    const outScale = 1 / 64; // headroom for the half-float fallback target
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bodyTarget.fbo);
    gl.viewport(0, 0, 1, 1);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this.prog.bodyForce);
    const u = this.u.bodyForce;
    this.bindTex(u.uPosVel, 0, this.posVel[this.cur].tex);
    gl.uniform1i(u.uTexWidth, TEX_WIDTH);
    gl.uniform1i(u.uCount, this.count);
    gl.uniform1f(u.uWallK, p.wallStiffness);
    // PBF applies the boundary nudge with damping halved (see stepPBF)
    gl.uniform1f(u.uWallC, p.wallDamping * (p.physics === "pbf" ? 0.5 : 1));
    gl.uniform1f(u.uMargin, this.h);
    gl.uniform1f(u.uMass, p.mass);
    gl.uniform1f(u.uOutScale, outScale);
    gl.uniform2f(u.uBodyCenter, cx, cy);
    const n = Math.min(circles.length, 8);
    gl.uniform1i(u.uBodyCount, n);
    this.bodyData.fill(0);
    for (let k = 0; k < n; k++) {
      this.bodyData[k * 4] = circles[k].x;
      this.bodyData[k * 4 + 1] = circles[k].y;
      this.bodyData[k * 4 + 2] = circles[k].r;
    }
    gl.uniform4fv(u.uBody, this.bodyData);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    gl.bindVertexArray(this.emptyVAO);
    gl.drawArrays(gl.POINTS, 0, this.count);
    gl.bindVertexArray(null);
    gl.disable(gl.BLEND);
    gl.readPixels(0, 0, 1, 1, gl.RGBA, gl.FLOAT, this.bodyPixel);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    // PBF applies the boundary nudge once per constraint iteration, so the
    // particles receive iterations x the impulse — mirror that in the
    // reaction or the body rides too deep in PBF
    const iters = p.physics === "pbf" ? Math.max(1, Math.round(p.pbfIterations) | 1) : 1;
    const s = iters / outScale;
    return {
      fx: this.bodyPixel[0] * s,
      fy: this.bodyPixel[1] * s,
      torque: this.bodyPixel[2] * s,
    };
  }

  // Read particle state back to CPU arrays (tests / debugging only).
  readback() {
    const gl = this.gl;
    const data = new Float32Array(TEX_WIDTH * this.texHeight * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.posVel[this.cur].fbo);
    gl.readPixels(0, 0, TEX_WIDTH, this.texHeight, gl.RGBA, gl.FLOAT, data);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    const n = this.count;
    const out = {
      px: new Float32Array(n), py: new Float32Array(n),
      vx: new Float32Array(n), vy: new Float32Array(n),
    };
    for (let i = 0; i < n; i++) {
      out.px[i] = data[i * 4];
      out.py[i] = data[i * 4 + 1];
      out.vx[i] = data[i * 4 + 2];
      out.vy[i] = data[i * 4 + 3];
    }
    return out;
  }
}
