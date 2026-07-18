// WebGPU compute SPH solver. Same force-SPH math as the WebGL2 backend but
// with an exact uniform grid (atomic counting sort + prefix sum), no per-cell
// slot cap, and storage-buffer state — scales to 1M+ particles.
//
// Per substep: clear counts -> count -> scan (3 dispatches) -> scatter ->
// density+curl -> force+integrate (ping-pong posVel buffer).

import {
  countWGSL, scanBlockWGSL, scanSerialWGSL, scanAddWGSL,
  scatterWGSL, densityWGSL, forceWGSL, duckForceWGSL,
  predictWGSL, pbfLambdaWGSL, pbfDeltaPWGSL, pbfFinalizeWGSL,
} from "./wgsl.js";
import { WORLD_HEIGHT as REF_WORLD_H } from "../config.js";

const WG = 256;

export class WGPUSolver {
  constructor(device, world, params) {
    this.device = device;
    this.world = world;
    this.params = params;
    this.maxCount = 1 << 20; // 1,048,576
    device.onuncapturederror = (e) => console.error("WebGPU error:", e.error.message);

    const h = params.h;
    this.h = h;
    this.kern = {
      poly6: 4 / (Math.PI * h ** 8),
      spiky3: 10 / (Math.PI * h ** 5),
      spikyGrad: 30 / (Math.PI * h ** 5),
      viscLap: 40 / (Math.PI * h ** 5),
    };

    this.pointer = { x: 0, y: 0, active: false, mode: 1 };
    this.paramsData = new Float32Array(36); // struct Params (144 bytes incl. padding)
    this.paramsBuf = device.createBuffer({
      size: this.paramsData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // obstacles: runtime-sized storage array, capacity doubles on demand
    this.obstacleCap = 64;
    this.obstacleData = new Float32Array(this.obstacleCap * 4);
    this.obstacleBuf = device.createBuffer({
      size: this.obstacleData.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    const mkPipeline = (code, label) => device.createComputePipeline({
      label,
      layout: "auto",
      compute: { module: device.createShaderModule({ label, code }), entryPoint: "main" },
    });
    this.pCount = mkPipeline(countWGSL, "count");
    this.pScanBlock = mkPipeline(scanBlockWGSL, "scanBlock");
    this.pScanSerial = mkPipeline(scanSerialWGSL, "scanSerial");
    this.pScanAdd = mkPipeline(scanAddWGSL, "scanAdd");
    this.pScatter = mkPipeline(scatterWGSL, "scatter");
    this.pDensity = mkPipeline(densityWGSL, "density");
    this.pForce = mkPipeline(forceWGSL, "force");
    this.pPredict = mkPipeline(predictWGSL, "pbfPredict");
    this.pPbfLambda = mkPipeline(pbfLambdaWGSL, "pbfLambda");
    this.pPbfDeltaP = mkPipeline(pbfDeltaPWGSL, "pbfDeltaP");
    this.pPbfFinalize = mkPipeline(pbfFinalizeWGSL, "pbfFinalize");
    this.pDuckForce = mkPipeline(duckForceWGSL, "duckForce");

    // duck reaction-force reduction: fixed-point atomic sums, read back
    // asynchronously (sampleDuckForce returns the latest resolved value)
    this.duckData = new Float32Array(36); // struct DuckParams (144 bytes)
    this.duckU = device.createBuffer({
      size: this.duckData.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // one slot per duck: each needs its own accumulator + staging buffer so
    // the in-flight readbacks don't collide
    this.duckSlots = [];
    this.duckForce = { fx: 0, fy: 0, torque: 0 }; // slot 0, for tests/compat

    this.count = 0;
    this.cur = 0;
    this.posVel = [null, null];
    this.nCells = 0;

    this.alloc(params.count);
    this.resizeGrid();
  }

  alloc(n) {
    const dev = this.device;
    if (n > this.maxCount) n = this.maxCount;
    if (n === this.count && this.posVel[0]) return;
    this.count = n;
    for (const b of this.posVel) b?.destroy();
    this.density?.destroy();
    this.curl?.destroy();
    this.sortedIdx?.destroy();
    this.predA?.destroy();
    this.predB?.destroy();
    this.lambdaBuf?.destroy();

    const mk = (size, extra = 0) => dev.createBuffer({
      size,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC | extra,
    });
    this.posVel = [mk(n * 16), mk(n * 16)];
    this.density = mk(n * 16);
    this.curl = mk(n * 8);
    this.sortedIdx = mk(n * 4);
    this.predA = mk(n * 16);       // PBF predicted positions ping-pong
    this.predB = mk(n * 16);
    this.lambdaBuf = mk(n * 8);    // PBF (lambda, rho)
    this.gridDirty = true;
    this.bindGroups = null;
    for (const s of this.duckSlots) s.binds = null; // reference the old posVel
  }

  resizeGrid() {
    const dev = this.device;
    const cols = Math.max(1, Math.ceil(this.world.w / this.h) + 1);
    const rows = Math.max(1, Math.ceil(this.world.h / this.h) + 1);
    if (cols === this.gridCols && rows === this.gridRows && !this.gridDirty) return;
    this.gridCols = cols;
    this.gridRows = rows;
    this.gridDirty = false;
    const nCells = cols * rows;
    this.nCells = nCells;
    this.cellCount?.destroy();
    this.cellStart?.destroy();
    this.cellFill?.destroy();
    this.blockSums?.destroy();
    const mk = (size) => dev.createBuffer({
      size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.cellCount = mk(nCells * 4);
    this.cellStart = mk(nCells * 4);
    this.cellFill = mk(nCells * 4);
    this.blockSums = mk(Math.ceil(nCells / WG) * 4);
    this.bindGroups = null; // depend on grid buffers
  }

  setWorld(w, h) {
    this.world.w = w;
    this.world.h = h;
    this.resizeGrid();
  }

  seed(state) {
    const n = this.count;
    const data = new Float32Array(n * 4);
    for (let i = 0; i < n; i++) {
      data[i * 4] = state.px[i];
      data[i * 4 + 1] = state.py[i];
      data[i * 4 + 2] = state.vx[i];
      data[i * 4 + 3] = state.vy[i];
    }
    for (const b of this.posVel) this.device.queue.writeBuffer(b, 0, data);
  }

  posVelTexture() { // interface parity: the renderer's particle source handle
    return this.posVel[this.cur];
  }

  worldScale() {
    return this.world.h / REF_WORLD_H;
  }

  writeParams(dt, pbf = false) {
    const p = this.params;
    const ptr = this.pointer;
    const d = this.paramsData;
    d[0] = ptr.x;
    d[1] = ptr.y;
    d[2] = ptr.active ? ptr.mode * p.pointerStrength : 0;
    d[3] = p.pointerRadius * this.worldScale();
    d[4] = this.world.w;
    d[5] = this.world.h;
    d[6] = this.gridCols;
    d[7] = this.gridRows;
    const ob = p.duckCircles.length
      ? p.obstacles.concat(p.duckCircles)
      : p.obstacles;
    d[8] = this.count;
    d[9] = ob.length; // drawn obstacles + duck hull circles
    d[10] = this.h;        // cellSize
    d[11] = this.h;
    d[12] = this.h * this.h;
    d[13] = this.kern.poly6;
    d[14] = this.kern.spiky3;
    d[15] = this.kern.spikyGrad;
    d[16] = this.kern.viscLap;
    d[17] = p.mass;
    d[18] = p.stiffness;
    d[19] = p.nearStiffness;
    d[20] = p.restDensity;
    d[21] = p.viscosity;
    d[22] = p.vorticity;
    d[23] = p.gravity / this.worldScale();
    d[24] = dt;
    d[25] = p.maxSpeed;
    d[26] = p.boundaryDamp;
    d[27] = this.h * 0.5;  // eps
    d[28] = p.wallStiffness;
    // per-iteration position-space damping: half of wallDamping measures
    // calmest (matches the WebGL2 deltaP pass)
    d[29] = p.wallDamping * (pbf ? 0.5 : 1);
    d[30] = p.pbfRelax;
    d[31] = p.pbfSCorrK;
    const dq = p.pbfSCorrDq * this.h;
    d[32] = 1 / (this.kern.poly6 * Math.pow(this.h * this.h - dq * dq, 3));
    d[33] = p.pbfXsph * p.viscosity;
    d[34] = p.gravityCenter ? 1 : 0;
    this.device.queue.writeBuffer(this.paramsBuf, 0, d);

    if (ob.length > this.obstacleCap) {
      while (this.obstacleCap < ob.length) this.obstacleCap *= 2;
      this.obstacleData = new Float32Array(this.obstacleCap * 4);
      this.obstacleBuf.destroy();
      this.obstacleBuf = this.device.createBuffer({
        size: this.obstacleData.byteLength,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
      this.bindGroups = null; // force/deltaP groups reference the old buffer
    }
    for (let k = 0; k < ob.length; k++) {
      this.obstacleData[k * 4] = ob[k].x;
      this.obstacleData[k * 4 + 1] = ob[k].y;
      this.obstacleData[k * 4 + 2] = ob[k].r;
    }
    this.device.queue.writeBuffer(this.obstacleBuf, 0, this.obstacleData);
  }

  makeBindGroups() {
    const dev = this.device;
    const bg = (pipeline, entries) => dev.createBindGroup({
      layout: pipeline.getBindGroupLayout(0),
      entries: entries.map((resource, i) => ({
        binding: i,
        resource: resource instanceof GPUBuffer ? { buffer: resource } : resource,
      })),
    });
    this.bindGroups = [0, 1].map((cur) => {
      const src = this.posVel[cur], dst = this.posVel[1 - cur];
      return {
        count: bg(this.pCount, [this.paramsBuf, src, this.cellCount]),
        scanBlock: bg(this.pScanBlock, [this.cellCount, this.cellStart, this.blockSums]),
        scanSerial: bg(this.pScanSerial, [this.blockSums]),
        scanAdd: bg(this.pScanAdd, [this.cellStart, this.blockSums]),
        scatter: bg(this.pScatter, [this.paramsBuf, src, this.cellStart, this.cellFill, this.sortedIdx]),
        density: bg(this.pDensity, [this.paramsBuf, src, this.cellStart, this.cellCount, this.sortedIdx, this.density, this.curl]),
        force: bg(this.pForce, [this.paramsBuf, src, this.cellStart, this.cellCount, this.sortedIdx, this.density, this.curl, dst, this.obstacleBuf]),
        // PBF: predict from src, iterate predA <-> predB (odd count ends in
        // predB), finalize (predB, src) -> dst
        predict: bg(this.pPredict, [this.paramsBuf, src, this.predA]),
        finalize: bg(this.pPbfFinalize, [this.paramsBuf, this.predB, this.cellStart, this.cellCount, this.sortedIdx, src, dst]),
      };
    });
    const gridArgs = [this.cellStart, this.cellCount, this.sortedIdx];
    this.pbfGroups = {
      count: bg(this.pCount, [this.paramsBuf, this.predA, this.cellCount]),
      scatter: bg(this.pScatter, [this.paramsBuf, this.predA, this.cellStart, this.cellFill, this.sortedIdx]),
      lambdaA: bg(this.pPbfLambda, [this.paramsBuf, this.predA, ...gridArgs, this.lambdaBuf]),
      lambdaB: bg(this.pPbfLambda, [this.paramsBuf, this.predB, ...gridArgs, this.lambdaBuf]),
      deltaAB: bg(this.pPbfDeltaP, [this.paramsBuf, this.predA, ...gridArgs, this.lambdaBuf, this.predB, this.obstacleBuf]),
      deltaBA: bg(this.pPbfDeltaP, [this.paramsBuf, this.predB, ...gridArgs, this.lambdaBuf, this.predA, this.obstacleBuf]),
    };
  }

  step(dt) {
    if (this.params.physics === "pbf") this.stepPBF(dt);
    else this.stepForce(dt);
  }

  beginStep(dt, pbf) {
    this.writeParams(dt, pbf); // may grow the obstacle buffer -> invalidates groups
    if (!this.bindGroups) this.makeBindGroups();
    const enc = this.device.createCommandEncoder();
    enc.clearBuffer(this.cellCount);
    enc.clearBuffer(this.cellFill);
    const pass = enc.beginComputePass();
    const dispatch = (pipeline, group, wgs) => {
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, group);
      pass.dispatchWorkgroups(wgs);
    };
    return { enc, pass, dispatch };
  }

  stepForce(dt) {
    const { enc, pass, dispatch } = this.beginStep(dt, false);
    const bgs = this.bindGroups[this.cur];
    const particleWGs = Math.ceil(this.count / WG);
    const cellWGs = Math.ceil(this.nCells / WG);
    dispatch(this.pCount, bgs.count, particleWGs);
    dispatch(this.pScanBlock, bgs.scanBlock, cellWGs);
    dispatch(this.pScanSerial, bgs.scanSerial, 1);
    dispatch(this.pScanAdd, bgs.scanAdd, cellWGs);
    dispatch(this.pScatter, bgs.scatter, particleWGs);
    dispatch(this.pDensity, bgs.density, particleWGs);
    dispatch(this.pForce, bgs.force, particleWGs);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    this.cur = 1 - this.cur;
  }

  stepPBF(dt) {
    const { enc, pass, dispatch } = this.beginStep(dt, true);
    const bgs = this.bindGroups[this.cur];
    const pg = this.pbfGroups;
    const particleWGs = Math.ceil(this.count / WG);
    const cellWGs = Math.ceil(this.nCells / WG);
    const iters = Math.max(1, Math.round(this.params.pbfIterations) | 1); // odd: ends in predB
    dispatch(this.pPredict, bgs.predict, particleWGs);
    dispatch(this.pCount, pg.count, particleWGs);          // grid from predicted
    dispatch(this.pScanBlock, bgs.scanBlock, cellWGs);
    dispatch(this.pScanSerial, bgs.scanSerial, 1);
    dispatch(this.pScanAdd, bgs.scanAdd, cellWGs);
    dispatch(this.pScatter, pg.scatter, particleWGs);
    for (let it = 0; it < iters; it++) {
      const fromA = it % 2 === 0;
      dispatch(this.pPbfLambda, fromA ? pg.lambdaA : pg.lambdaB, particleWGs);
      dispatch(this.pPbfDeltaP, fromA ? pg.deltaAB : pg.deltaBA, particleWGs);
    }
    dispatch(this.pPbfFinalize, bgs.finalize, particleWGs);
    pass.end();
    this.device.queue.submit([enc.finish()]);
    this.cur = 1 - this.cur;
  }

  duckSlot(i) {
    if (!this.duckSlots[i]) {
      const dev = this.device;
      this.duckSlots[i] = {
        out: dev.createBuffer({
          size: 16,
          usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        }),
        staging: dev.createBuffer({
          size: 16,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        }),
        binds: null,
        pending: false,
        force: { fx: 0, fy: 0, torque: 0 },
      };
    }
    return this.duckSlots[i];
  }

  // Latest resolved reaction force for duck `i` (zero until the first
  // readback lands).
  duckForceAt(i) {
    return this.duckSlots[i] ? this.duckSlots[i].force : { fx: 0, fy: 0, torque: 0 };
  }

  // Sum the fluid's reaction to one duck hull's boundary springs. The
  // readback is async: returns the most recently resolved value (a frame or
  // two stale), never stalls the pipeline. Uses P from the last substep.
  sampleDuckForce(circles, cx, cy, slotIndex = 0) {
    const dev = this.device;
    const slot = this.duckSlot(slotIndex);
    const FIX = 32; // fixed-point units per force unit
    const d = this.duckData;
    d.fill(0);
    d[0] = cx;
    d[1] = cy;
    d[2] = Math.min(circles.length, 8);
    d[3] = FIX;
    for (let k = 0; k < d[2]; k++) {
      d[4 + k * 4] = circles[k].x;
      d[5 + k * 4] = circles[k].y;
      d[6 + k * 4] = circles[k].r;
    }
    // the uniform is shared across slots: writes and submits execute in
    // queue order, so each dispatch still sees its own duck's data
    dev.queue.writeBuffer(this.duckU, 0, d);
    if (!slot.binds) {
      slot.binds = [0, 1].map((cur) => dev.createBindGroup({
        layout: this.pDuckForce.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.paramsBuf } },
          { binding: 1, resource: { buffer: this.posVel[cur] } },
          { binding: 2, resource: { buffer: this.duckU } },
          { binding: 3, resource: { buffer: slot.out } },
        ],
      }));
    }
    const enc = dev.createCommandEncoder();
    enc.clearBuffer(slot.out);
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pDuckForce);
    pass.setBindGroup(0, slot.binds[this.cur]);
    pass.dispatchWorkgroups(Math.ceil(this.count / WG));
    pass.end();
    const read = !slot.pending;
    if (read) enc.copyBufferToBuffer(slot.out, 0, slot.staging, 0, 16);
    dev.queue.submit([enc.finish()]);
    if (read) {
      slot.pending = true;
      // exposed so the ?t= fast-forward loop can await the fresh value
      this.duckReadPromise = slot.staging.mapAsync(GPUMapMode.READ).then(() => {
        const a = new Int32Array(slot.staging.getMappedRange().slice(0));
        // PBF applies the boundary nudge once per constraint iteration —
        // mirror that in the reaction (see the WebGL2 sampleDuckForce)
        const p = this.params;
        const iters = p.physics === "pbf"
          ? Math.max(1, Math.round(p.pbfIterations) | 1) : 1;
        const s = iters / FIX;
        slot.force = { fx: a[0] * s, fy: a[1] * s, torque: a[2] * s };
        if (slotIndex === 0) this.duckForce = slot.force; // tests read this
        slot.staging.unmap();
        slot.pending = false;
      }).catch(() => { slot.pending = false; });
    }
    return slot.force;
  }

  // Async readback for tests/debugging.
  async readback() {
    const dev = this.device;
    const n = this.count;
    const staging = dev.createBuffer({
      size: n * 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const enc = dev.createCommandEncoder();
    enc.copyBufferToBuffer(this.posVel[this.cur], 0, staging, 0, n * 16);
    dev.queue.submit([enc.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const data = new Float32Array(staging.getMappedRange().slice(0));
    staging.unmap();
    staging.destroy();
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
