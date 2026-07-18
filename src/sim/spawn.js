// Initial particle configurations. Each fills the solver's position/velocity
// arrays for solver.count particles inside solver.world.

function lattice(solver, x0, y0, w, hgt, startIndex, maxCount) {
  const spacing = solver.h * 0.55;
  const jitter = spacing * 0.3;
  const cols = Math.max(1, Math.floor(w / spacing));
  let i = startIndex;
  outer:
  for (let row = 0; ; row++) {
    const y = y0 + row * spacing;
    if (hgt > 0 && y > y0 + hgt) break;
    for (let col = 0; col < cols; col++) {
      if (i >= maxCount) break outer;
      solver.px[i] = x0 + col * spacing + (Math.random() - 0.5) * jitter;
      solver.py[i] = y + (Math.random() - 0.5) * jitter;
      solver.vx[i] = 0;
      solver.vy[i] = 0;
      i++;
    }
  }
  return i;
}

export const spawners = {
  // Classic: block of fluid on the left, released against the floor.
  damBreak(solver) {
    const { w, h } = solver.world;
    lattice(solver, 2, 2, w * 0.42, -1, 0, solver.count);
    // lattice() grows downward from y0; flip so the block sits on the floor
    for (let i = 0; i < solver.count; i++) solver.py[i] = h - 2 - (solver.py[i] - 2);
  },

  // Two blocks colliding in the middle.
  doubleDam(solver) {
    const { w, h } = solver.world;
    const half = Math.floor(solver.count / 2);
    lattice(solver, 2, 2, w * 0.24, -1, 0, half);
    lattice(solver, w * 0.76 - 2, 2, w * 0.24, -1, half, solver.count);
    for (let i = 0; i < solver.count; i++) solver.py[i] = h - 2 - (solver.py[i] - 2);
  },

  // Resting pool with a round blob dropped into it.
  drop(solver) {
    const { w, h } = solver.world;
    const poolCount = Math.floor(solver.count * 0.7);
    lattice(solver, 2, 2, w - 4, -1, 0, poolCount);
    for (let i = 0; i < poolCount; i++) solver.py[i] = h - 2 - (solver.py[i] - 2);

    // blob: filled disc above the pool
    const spacing = solver.h * 0.55;
    const cx = w * 0.5, cy = h * 0.22;
    const blobCount = solver.count - poolCount;
    const R = Math.sqrt(blobCount / Math.PI) * spacing * 1.05;
    let i = poolCount;
    for (let ring = 0; i < solver.count; ring++) {
      const r = ring * spacing;
      if (r > R * 3) break; // safety
      const n = ring === 0 ? 1 : Math.floor((2 * Math.PI * r) / spacing);
      for (let s = 0; s < n && i < solver.count; s++) {
        const a = (s / n) * Math.PI * 2;
        solver.px[i] = cx + Math.cos(a) * r;
        solver.py[i] = cy + Math.sin(a) * r;
        solver.vx[i] = 0;
        solver.vy[i] = 0;
        i++;
      }
    }
    // leftovers (if the disc estimate ran short) join the pool
    for (; i < solver.count; i++) {
      solver.px[i] = 2 + Math.random() * (w - 4);
      solver.py[i] = h - 2 - Math.random() * 5;
      solver.vx[i] = 0;
      solver.vy[i] = 0;
    }
  },

  // A single round blob at the world center (pairs well with center gravity).
  sphere(solver) {
    const { w, h } = solver.world;
    const spacing = solver.h * 0.55;
    const cx = w * 0.5, cy = h * 0.5;
    let i = 0;
    for (let ring = 0; i < solver.count; ring++) {
      const r = ring * spacing;
      const n = ring === 0 ? 1 : Math.floor((2 * Math.PI * r) / spacing);
      for (let s = 0; s < n && i < solver.count; s++) {
        const a = (s / n) * Math.PI * 2;
        solver.px[i] = cx + Math.cos(a) * r;
        solver.py[i] = cy + Math.sin(a) * r;
        solver.vx[i] = 0;
        solver.vy[i] = 0;
        i++;
      }
    }
  },
};

export const presetNames = Object.keys(spawners);
