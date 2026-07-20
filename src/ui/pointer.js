// Pointer interaction. Normal mode: left-drag pushes fluid, right-drag
// pulls it; a drag starting on a rigid body grabs it instead (release to
// throw). Obstacle-draw mode (toggle "O"): left-drag stamps circular
// walls, right-drag erases them. Eraser mode (toggle "E"): any drag
// erases obstacles near the cursor. Object add/remove modes: a click spawns
// the object picked in the panel, or removes whichever one is under the
// cursor.

import { WORLD_HEIGHT } from "../config.js";

export function setupPointer(canvas, solver, params, bodies, world) {
  const toWorld = (e) => {
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((e.clientX - rect.left) / rect.width) * solver.world.w,
      y: ((e.clientY - rect.top) / rect.height) * solver.world.h,
    };
  };

  let drawing = 0; // 0 = off, 1 = stamping, -1 = erasing
  let draggingBody = false;

  const stamp = (w) => {
    const r = params.obstacleRadius * (solver.world.h / WORLD_HEIGHT);
    if (drawing === 1) {
      const last = params.obstacles[params.obstacles.length - 1];
      if (last && Math.hypot(last.x - w.x, last.y - w.y) < r * 0.6) return;
      params.obstacles.push({ x: w.x, y: w.y, r });
    } else {
      for (let i = params.obstacles.length - 1; i >= 0; i--) {
        const o = params.obstacles[i];
        if (Math.hypot(o.x - w.x, o.y - w.y) < o.r + r * 0.5) {
          params.obstacles.splice(i, 1);
        }
      }
    }
  };

  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    const w = toWorld(e);
    if (params.addBodies) {
      bodies.spawn(world, w.x, w.y); // picked type, one per click, not per drag
    } else if (params.removeBodies) {
      bodies.removeAt(w.x, w.y);
    } else if (params.drawObstacles || params.eraseObstacles) {
      drawing = (params.eraseObstacles || e.button === 2) ? -1 : 1;
      stamp(w);
    } else if (bodies && bodies.startDrag(w.x, w.y)) {
      draggingBody = true; // grab the object instead of pushing fluid
    } else {
      solver.pointer.x = w.x;
      solver.pointer.y = w.y;
      solver.pointer.mode = e.button === 2 ? -1 : 1;
      solver.pointer.active = true;
    }
    e.preventDefault();
  });
  canvas.addEventListener("pointermove", (e) => {
    const w = toWorld(e);
    if (drawing !== 0) {
      stamp(w);
    } else if (draggingBody) {
      bodies.dragTo(w.x, w.y);
    } else if (solver.pointer.active) {
      solver.pointer.x = w.x;
      solver.pointer.y = w.y;
    }
  });
  const release = () => {
    solver.pointer.active = false;
    drawing = 0;
    if (draggingBody) {
      bodies.endDrag(); // the body keeps its velocity: release mid-swing to throw
      draggingBody = false;
    }
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
}
