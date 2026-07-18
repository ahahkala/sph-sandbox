// WebGPU renderer: instanced-quad particle sprites (WebGPU has no point
// size) into either the canvas directly (dots) or a half-res field texture
// that a composite pass shades into a liquid surface. The composite pass
// also draws the obstacle overlay.

import { particleQuadWGSL, compositeWGSL, obstacleOverlayWGSL, duckWGSL } from "./wgsl.js";
import { DUCK_MESH, DUCK_STRIDE } from "../sim/duck.js";

const FIELD_SCALE = 0.5;
const FIELD_AMP = 0.9;

export class WGPURenderer {
  constructor(device, canvas, params) {
    this.device = device;
    this.canvas = canvas;
    this.params = params;
    this.ctx = canvas.getContext("webgpu");
    if (!this.ctx) throw new Error("webgpu canvas context unavailable");
    this.format = navigator.gpu.getPreferredCanvasFormat();
    this.ctx.configure({ device, format: this.format, alphaMode: "opaque" });

    this.renderData = new Float32Array(12); // struct RenderParams (48 bytes)
    this.uDots = this.mkUniform();
    this.uField = this.mkUniform();
    this.uComposite = this.mkUniform();
    // obstacles: runtime-sized storage array, capacity doubles on demand
    this.obstacleCap = 64;
    this.obstacleData = new Float32Array(this.obstacleCap * 4);
    this.obstacleBuf = device.createBuffer({
      size: this.obstacleData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const particleModule = device.createShaderModule({ code: particleQuadWGSL });
    const compositeModule = device.createShaderModule({ code: compositeWGSL });

    const quadPipeline = (module, entry, format, blend) => device.createRenderPipeline({
      layout: "auto",
      vertex: { module, entryPoint: "vs" },
      fragment: { module, entryPoint: entry, targets: [{ format, blend }] },
      primitive: { topology: "triangle-strip" },
    });
    const additive = {
      color: { srcFactor: "one", dstFactor: "one" },
      alpha: { srcFactor: "one", dstFactor: "one" },
    };
    const alphaAdd = {
      color: { srcFactor: "src-alpha", dstFactor: "one" },
      alpha: { srcFactor: "zero", dstFactor: "one" },
    };
    this.pDots = quadPipeline(particleModule, "fsDots", this.format, alphaAdd);
    this.pField = quadPipeline(particleModule, "fsField", "rgba16float", additive);
    this.pComposite = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: compositeModule, entryPoint: "vs" },
      fragment: { module: compositeModule, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });
    this.sampler = device.createSampler({ magFilter: "linear", minFilter: "linear" });

    // obstacle overlay for dots mode (composite draws its own in liquid mode)
    const obstacleModule = device.createShaderModule({ code: obstacleOverlayWGSL });
    this.pObstacle = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: obstacleModule, entryPoint: "vs" },
      fragment: {
        module: obstacleModule, entryPoint: "fs",
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
            alpha: { srcFactor: "zero", dstFactor: "one" },
          },
        }],
      },
      primitive: { topology: "triangle-list" },
    });
    this.obstacleBind = null;

    // duck mesh: static vertex buffer, transform in a small uniform
    const duckModule = device.createShaderModule({ code: duckWGSL });
    this.pDuck = device.createRenderPipeline({
      layout: "auto",
      vertex: {
        module: duckModule, entryPoint: "vs",
        buffers: [{
          arrayStride: DUCK_STRIDE * 4,
          attributes: [
            { shaderLocation: 0, offset: 0, format: "float32x2" },
            { shaderLocation: 1, offset: 8, format: "float32x3" },
          ],
        }],
      },
      fragment: { module: duckModule, entryPoint: "fs", targets: [{ format: this.format }] },
      primitive: { topology: "triangle-list" },
    });
    this.duckVB = device.createBuffer({
      size: DUCK_MESH.byteLength,
      usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST,
    });
    device.queue.writeBuffer(this.duckVB, 0, DUCK_MESH);
    this.duckData = new Float32Array(8); // struct DuckDraw (32 bytes)
    // one uniform buffer per duck: the poses differ within a frame, and
    // queue.writeBuffer would otherwise leave every draw with the last pose
    this.duckSlots = [];

    this.fieldTex = null;
    this.bindCache = new WeakMap(); // per posVel buffer bind groups
  }

  mkDuckSlot() {
    const buf = this.device.createBuffer({
      size: this.duckData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    return {
      buf,
      bind: this.device.createBindGroup({
        layout: this.pDuck.getBindGroupLayout(0),
        entries: [{ binding: 0, resource: { buffer: buf } }],
      }),
    };
  }

  mkUniform() {
    return this.device.createBuffer({
      size: this.renderData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
  }

  resize(pixelW, pixelH) {
    this.canvas.width = pixelW;
    this.canvas.height = pixelH;
    this.fieldTex?.destroy();
    this.fieldTex = this.device.createTexture({
      size: [Math.max(2, Math.round(pixelW * FIELD_SCALE)), Math.max(2, Math.round(pixelH * FIELD_SCALE))],
      format: "rgba16float",
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    });
    this.fieldView = this.fieldTex.createView();
    this.compositeBind = null;
  }

  writeRenderParams(buf, world, halfSizePx) {
    const d = this.renderData;
    const interior = FIELD_AMP * 3.0;
    d[0] = world.w;
    d[1] = world.h;
    d[2] = this.canvas.width;
    d[3] = this.canvas.height;
    d[4] = halfSizePx;
    d[5] = 1 / this.params.speedRef;
    d[6] = FIELD_AMP;
    d[7] = interior * 0.18;  // threshold
    d[8] = 1 / interior;     // invScale
    d[9] = this.params.obstacles.length;
    this.device.queue.writeBuffer(buf, 0, d);
  }

  particleBind(pipeline, uniform, posVelBuf, key) {
    let cache = this.bindCache.get(posVelBuf);
    if (!cache) { cache = {}; this.bindCache.set(posVelBuf, cache); }
    if (!cache[key]) {
      cache[key] = this.device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniform } },
          { binding: 1, resource: { buffer: posVelBuf } },
        ],
      });
    }
    return cache[key];
  }

  render(posVelBuf, count, world) {
    const dev = this.device;
    const p = this.params;
    const ob = p.obstacles;
    if (ob.length > this.obstacleCap) {
      while (this.obstacleCap < ob.length) this.obstacleCap *= 2;
      this.obstacleData = new Float32Array(this.obstacleCap * 4);
      this.obstacleBuf.destroy();
      this.obstacleBuf = dev.createBuffer({
        size: this.obstacleData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.compositeBind = null; // both reference the old buffer
      this.obstacleBind = null;
    }
    for (let k = 0; k < ob.length; k++) {
      this.obstacleData[k * 4] = ob[k].x;
      this.obstacleData[k * 4 + 1] = ob[k].y;
      this.obstacleData[k * 4 + 2] = ob[k].r;
    }
    dev.queue.writeBuffer(this.obstacleBuf, 0, this.obstacleData);

    const poses = p.duckPoses;
    for (let i = 0; i < poses.length; i++) {
      if (!this.duckSlots[i]) this.duckSlots[i] = this.mkDuckSlot();
      const pose = poses[i];
      const dd = this.duckData;
      dd[0] = pose.x;
      dd[1] = pose.y;
      dd[2] = Math.cos(pose.angle);
      dd[3] = Math.sin(pose.angle);
      dd[4] = world.w;
      dd[5] = world.h;
      dd[6] = pose.scale;
      dev.queue.writeBuffer(this.duckSlots[i].buf, 0, dd);
    }
    const drawDuck = (pass) => {
      if (poses.length === 0) return;
      pass.setPipeline(this.pDuck);
      pass.setVertexBuffer(0, this.duckVB);
      for (let i = 0; i < poses.length; i++) {
        pass.setBindGroup(0, this.duckSlots[i].bind);
        pass.draw(DUCK_MESH.length / DUCK_STRIDE);
      }
    };

    const enc = dev.createCommandEncoder();
    const canvasView = this.ctx.getCurrentTexture().createView();

    if (p.renderMode === "liquid") {
      const fieldH = this.fieldTex.height;
      const half = Math.max(2.2, (p.h * 1.5) * (fieldH / world.h));
      this.writeRenderParams(this.uField, world, half);
      const fieldPass = enc.beginRenderPass({
        colorAttachments: [{
          view: this.fieldView, loadOp: "clear", storeOp: "store",
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
        }],
      });
      fieldPass.setPipeline(this.pField);
      fieldPass.setBindGroup(0, this.particleBind(this.pField, this.uField, posVelBuf, "field"));
      fieldPass.draw(4, count);
      fieldPass.end();

      this.writeRenderParams(this.uComposite, world, 0);
      if (!this.compositeBind) {
        this.compositeBind = dev.createBindGroup({
          layout: this.pComposite.getBindGroupLayout(0),
          entries: [
            { binding: 0, resource: { buffer: this.uComposite } },
            { binding: 1, resource: this.fieldView },
            { binding: 2, resource: this.sampler },
            { binding: 3, resource: { buffer: this.obstacleBuf } },
          ],
        });
      }
      const pass = enc.beginRenderPass({
        colorAttachments: [{ view: canvasView, loadOp: "clear", storeOp: "store" }],
      });
      pass.setPipeline(this.pComposite);
      pass.setBindGroup(0, this.compositeBind);
      pass.draw(3);
      drawDuck(pass);
      pass.end();
    } else {
      const half = Math.max(1, p.h * (this.canvas.height / world.h) * 0.65);
      this.writeRenderParams(this.uDots, world, half);
      const pass = enc.beginRenderPass({
        colorAttachments: [{
          view: canvasView, loadOp: "clear", storeOp: "store",
          clearValue: { r: 0.02, g: 0.028, b: 0.055, a: 1 },
        }],
      });
      pass.setPipeline(this.pDots);
      pass.setBindGroup(0, this.particleBind(this.pDots, this.uDots, posVelBuf, "dots"));
      pass.draw(4, count);
      if (ob.length > 0) {
        if (!this.obstacleBind) {
          this.obstacleBind = dev.createBindGroup({
            layout: this.pObstacle.getBindGroupLayout(0),
            entries: [
              { binding: 0, resource: { buffer: this.uDots } },
              { binding: 1, resource: { buffer: this.obstacleBuf } },
            ],
          });
        }
        pass.setPipeline(this.pObstacle);
        pass.setBindGroup(0, this.obstacleBind);
        pass.draw(3);
      }
      drawDuck(pass);
      pass.end();
    }
    dev.queue.submit([enc.finish()]);
  }
}
