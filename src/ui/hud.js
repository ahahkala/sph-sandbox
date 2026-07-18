// Bottom-left stats readout, updated a couple of times per second.

export class Hud {
  constructor(el) {
    this.el = el;
    this.frames = 0;
    this.simMs = 0;
    this.lastReport = performance.now();
  }

  frame(simMs) {
    this.frames++;
    this.simMs += simMs;
    const now = performance.now();
    const elapsed = now - this.lastReport;
    if (elapsed > 500) {
      const fps = Math.round((this.frames * 1000) / elapsed);
      const avgSim = (this.simMs / this.frames).toFixed(1);
      this.el.textContent =
        `${this.count ?? "?"} particles · ${this.backend ?? ""}/${this.physics ?? ""} · ${fps} fps · sim ${avgSim} ms`;
      this.frames = 0;
      this.simMs = 0;
      this.lastReport = now;
    }
  }

  setCount(n, backend, physics) {
    this.count = n;
    this.backend = backend;
    this.physics = physics;
  }
}
