// WebGPU renderer: instanced-quad particle sprites (WebGPU has no point
// size) into either the canvas directly (dots) or a half-res field texture
// that a composite pass shades into a liquid surface. The composite pass
// also draws the obstacle overlay.

import { particleQuadWGSL, compositeWGSL, obstacleOverlayWGSL, bodySpriteWGSL } from "./wgsl.js";
import { BODY_TYPES, bodyType } from "../sim/bodytypes.js";

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

    // rigid-body sprites: an attribute-less textured quad per body
    const bodyModule = device.createShaderModule({ code: bodySpriteWGSL });
    this.pBody = device.createRenderPipeline({
      layout: "auto",
      vertex: { module: bodyModule, entryPoint: "vs" },
      fragment: {
        module: bodyModule, entryPoint: "fs",
        targets: [{
          format: this.format,
          blend: {
            color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha" },
          },
        }],
      },
      primitive: { topology: "triangle-strip" },
    });
    this.bodyData = new Float32Array(12); // struct BodyDraw (48 bytes)
    // one uniform buffer per body: the poses differ within a frame, and
    // queue.writeBuffer would otherwise leave every draw with the last pose
    this.bodySlots = [];
    this.spriteViews = {}; // body type -> GPUTextureView, filled in on load
    this.loadSprites();

    this.fieldTex = null;
    this.bindCache = new WeakMap(); // per posVel buffer bind groups
  }

  // One texture per body type, fetched and decoded asynchronously; bodies
  // whose sprite hasn't arrived are skipped for a frame or two.
  async loadSprites() {
    for (const [name, type] of Object.entries(BODY_TYPES)) {
      const bitmap = await createImageBitmap(await (await fetch(type.sprite)).blob());
      const tex = this.device.createTexture({
        size: [bitmap.width, bitmap.height],
        format: "rgba8unorm",
        usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST |
          GPUTextureUsage.RENDER_ATTACHMENT,
      });
      this.device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: tex },
        [bitmap.width, bitmap.height]);
      this.spriteViews[name] = tex.createView();
    }
  }

  // A slot holds one body's uniform buffer plus its bind groups, one per
  // body type — the type in a given slot changes as bodies come and go.
  bodySlot(i, typeName) {
    let slot = this.bodySlots[i];
    if (!slot) {
      slot = {
        buf: this.device.createBuffer({
          size: this.bodyData.byteLength,
          usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        }),
        binds: {},
      };
      this.bodySlots[i] = slot;
    }
    if (!slot.binds[typeName]) {
      slot.binds[typeName] = this.device.createBindGroup({
        layout: this.pBody.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: slot.buf } },
          { binding: 1, resource: this.sampler },
          { binding: 2, resource: this.spriteViews[typeName] },
        ],
      });
    }
    return slot;
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

    const poses = p.bodyPoses;
    const drawable = [];
    for (let i = 0; i < poses.length; i++) {
      const pose = poses[i];
      if (!this.spriteViews[pose.type]) continue; // sprite still loading
      const slot = this.bodySlot(i, pose.type);
      const r = bodyType(pose.type).rect;
      const dd = this.bodyData;
      dd[0] = pose.x;
      dd[1] = pose.y;
      dd[2] = Math.cos(pose.angle);
      dd[3] = Math.sin(pose.angle);
      dd[4] = r.x0;
      dd[5] = r.y0;
      dd[6] = r.x1;
      dd[7] = r.y1;
      dd[8] = world.w;
      dd[9] = world.h;
      dd[10] = pose.scale;
      dev.queue.writeBuffer(slot.buf, 0, dd);
      drawable.push(slot.binds[pose.type]);
    }
    const drawBodies = (pass) => {
      if (drawable.length === 0) return;
      pass.setPipeline(this.pBody);
      for (const bind of drawable) {
        pass.setBindGroup(0, bind);
        pass.draw(4);
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
      drawBodies(pass);
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
      drawBodies(pass);
      pass.end();
    }
    dev.queue.submit([enc.finish()]);
  }
}
