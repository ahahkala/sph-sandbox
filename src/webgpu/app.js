// WebGPU backend bootstrap: adapter/device setup, returns a solver+renderer
// pair with the same interface main.js uses for the WebGL2 backend.

import { WGPUSolver } from "./wgpusolver.js";
import { WGPURenderer } from "./wgpurenderer.js";

export async function startWebGPU(canvas, world, params) {
  if (!navigator.gpu) throw new Error("navigator.gpu is undefined (WebGPU unsupported)");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("no WebGPU adapter");
  const device = await adapter.requestDevice();
  device.lost.then((info) => {
    if (info.reason !== "destroyed") console.error("WebGPU device lost:", info.message);
  });
  const renderer = new WGPURenderer(device, canvas, params);
  const solver = new WGPUSolver(device, world, params);
  return { solver, renderer };
}
