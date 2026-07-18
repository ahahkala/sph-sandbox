// Headless check for duck interaction wiring (middle-click spawn/remove,
// keyboard thrust, drag). Live rAF physics doesn't advance reliably headless
// (see CLAUDE.md), so real keyboard/mouse events verify the input plumbing and
// duck.step is driven directly for dynamics.
// Run: node tools/test-duck-input.mjs
import puppeteer from "puppeteer-core";
import { chromePath, ensureServer, BASE } from "./chrome.mjs";

await ensureServer();
const browser = await puppeteer.launch({
  executablePath: chromePath(),
  headless: true,
  args: ["--hide-scrollbars"],
});
let failures = 0;
const check = (name, ok, detail = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${detail ? "  — " + detail : ""}`);
  if (!ok) failures++;
};
try {
  const tab = await browser.newPage();
  await tab.setViewport({ width: 1280, height: 800 });
  tab.on("pageerror", (err) => console.error("page error:", err.message));
  await tab.goto(BASE + "/?n=25000&duck=1&t=6", { waitUntil: "load", timeout: 120000 });
  await tab.waitForFunction(() => (window.__frames || 0) >= 10, { timeout: 300000, polling: 200 });

  const duckState = () => tab.evaluate(() => {
    const f = window.__ducks;
    const d = f.ducks[0];
    return {
      n: f.ducks.length,
      pose: d ? { ...d.pose } : null,
      grab: d && d.grab ? { tx: d.grab.tx } : null,
      circles: window.__params.duckCircles.length,
      poses: window.__params.duckPoses.length,
    };
  });

  let st = await duckState();
  check("?duck=1 spawns one duck", st.n === 1 && st.poses === 1, `n ${st.n}`);
  check("hull circles published to params", st.circles === 3, `${st.circles} circles`);

  // --- keyboard wiring: real key events set/clear params.duckThrust -------
  await tab.keyboard.down("ArrowUp");
  await tab.keyboard.down("ArrowRight");
  let th = await tab.evaluate(() => ({ ...window.__params.duckThrust }));
  check("arrows set thrust", th.x === 1 && th.y === -1, JSON.stringify(th));
  await tab.keyboard.up("ArrowUp");
  await tab.keyboard.up("ArrowRight");
  th = await tab.evaluate(() => ({ ...window.__params.duckThrust }));
  check("keyup clears thrust", th.x === 0 && th.y === 0, JSON.stringify(th));

  // --- thrust dynamics: drive duck.step directly with gravity off ---------
  const dvy = await tab.evaluate(() => {
    const d = window.__ducks.ducks[0], p = window.__params;
    const world = { w: d.pose.x * 2, h: d.pose.y * 2 }; // roomy box
    const g0 = p.gravity;
    p.gravity = 0;
    p.duckThrust.y = -1;
    const vy0 = d.vy;
    for (let i = 0; i < 30; i++) d.step(1 / 60, world, null);
    p.duckThrust.y = 0;
    p.gravity = g0;
    return d.vy - vy0;
  });
  check("ArrowUp thrust accelerates the duck upward", dvy < -1, `dvy ${dvy.toFixed(2)}`);

  // --- pointer wiring: real mouse drag grabs, moves, releases -------------
  const dims = await tab.evaluate(() => {
    const c = document.getElementById("glcanvas").getBoundingClientRect();
    return { cw: c.width, ch: c.height };
  });
  const wh = 70 * Math.sqrt(25000 / 3000);
  const ww = wh * (dims.cw / dims.ch);
  const toScreen = (p) => ({ x: (p.x / ww) * dims.cw, y: (p.y / wh) * dims.ch });

  st = await duckState();
  let s = toScreen(st.pose);
  await tab.mouse.move(s.x, s.y);
  await tab.mouse.down();
  let grab = (await duckState()).grab;
  check("mousedown on duck grabs it", !!grab, JSON.stringify(grab));
  await tab.mouse.move(s.x + 150, s.y - 60);
  const grab2 = (await duckState()).grab;
  check("mousemove updates grab target", grab2 && grab2.tx > grab.tx,
    `tx ${grab?.tx?.toFixed(1)} -> ${grab2?.tx?.toFixed(1)}`);
  // grabbed drag spring pulls the duck toward the cursor when stepped
  const dvx = await tab.evaluate(() => {
    const d = window.__ducks.ducks[0];
    const world = { w: d.pose.x * 4, h: d.pose.y * 4 };
    const vx0 = d.vx;
    for (let i = 0; i < 10; i++) d.step(1 / 60, world, null);
    return d.vx - vx0;
  });
  check("drag spring pulls duck toward cursor", dvx > 0.5, `dvx ${dvx.toFixed(2)}`);
  await tab.mouse.up();
  check("mouseup releases the grab", (await duckState()).grab === null);

  // pointer down away from the duck should still push fluid, not grab
  await tab.mouse.move(s.x - 300, s.y - 300);
  await tab.mouse.down();
  check("mousedown off-duck doesn't grab", (await duckState()).grab === null);
  await tab.mouse.up();

  // --- Add tool: a click spawns a duck at the cursor ----------------------
  // paused, so the ducks stay where they were clicked while we test the tools
  await tab.keyboard.press("Space");
  const tools = () => tab.evaluate(() => {
    const p = window.__params;
    return {
      addDucks: p.addDucks, removeDucks: p.removeDucks,
      drawObstacles: p.drawObstacles, eraseObstacles: p.eraseObstacles,
    };
  });
  await tab.click("#duck-add");
  let t = await tools();
  check("Add activates only the add tool",
    Object.entries(t).every(([k, v]) => v === (k === "addDucks")), JSON.stringify(t));

  // the duck has drifted with the flow, so keep every click on open canvas:
  // clear of the control panel on the left and the hint line at the bottom
  const spot = (x, y) => ({
    x: Math.min(Math.max(x, 300), dims.cw - 20),
    y: Math.min(Math.max(y, 20), dims.ch - 60),
  });
  const p1 = spot(s.x - 300, s.y - 200);
  await tab.mouse.click(p1.x, p1.y);
  st = await duckState();
  check("click with Add spawns a duck",
    st.n === 2 && st.circles === 6 && st.poses === 2, `n ${st.n}`);
  const spawned = await tab.evaluate(() => ({ ...window.__ducks.ducks[1].pose }));
  const want = { x: (p1.x / dims.cw) * ww, y: (p1.y / dims.ch) * wh };
  check("spawns at the cursor",
    Math.hypot(spawned.x - want.x, spawned.y - want.y) < 1,
    `${spawned.x.toFixed(1)},${spawned.y.toFixed(1)} vs ${want.x.toFixed(1)},${want.y.toFixed(1)}`);

  // dragging must not spew one duck per pointermove
  const p2 = spot(p1.x - 200, p1.y - 150);
  await tab.mouse.move(p2.x, p2.y);
  await tab.mouse.down();
  await tab.mouse.move(p2.x + 60, p2.y + 40);
  await tab.mouse.move(p2.x + 120, p2.y + 80);
  await tab.mouse.up();
  st = await duckState();
  check("dragging with Add spawns only one duck", st.n === 3, `n ${st.n}`);

  // --- Remove tool: a click deletes the duck under the cursor -------------
  await tab.click("#duck-remove");
  t = await tools();
  check("Remove replaces Add as the active tool",
    t.removeDucks && !t.addDucks, JSON.stringify(t));
  await tab.mouse.click(p1.x, p1.y);
  st = await duckState();
  check("click with Remove deletes that duck", st.n === 2 && st.poses === 2, `n ${st.n}`);
  await tab.mouse.click(dims.cw - 20, 20); // top-right: no duck there
  check("click with Remove on empty water is a no-op", (await duckState()).n === 2);

  // --- tools stay mutually exclusive, and Clear drops all the ducks -------
  await tab.click("#obstacle-draw");
  t = await tools();
  check("obstacle Draw turns the duck tool off",
    t.drawObstacles && !t.removeDucks, JSON.stringify(t));
  await tab.click("#duck-clear");
  st = await duckState();
  check("Clear removes every duck and its published hull",
    st.n === 0 && st.circles === 0 && st.poses === 0,
    `n ${st.n}, ${st.circles} circles`);

  // --- duck-duck collision: drive the flock directly, gravity off ---------
  const coll = await tab.evaluate(() => {
    const f = window.__ducks, p = window.__params;
    const world = { w: 200, h: 200 };
    const g0 = p.gravity;
    p.gravity = 0;
    // two ducks overlapping by about half a body radius, level with each other
    const a = f.spawn(world, 100, 100);
    const b = f.spawn(world, 100 + a.circles[0].r, 100);
    const gap0 = b.pose.x - a.pose.x;
    for (let i = 0; i < 240; i++) f.step(1 / 240, world);
    const gap1 = b.pose.x - a.pose.x;
    // momentum must stay balanced: equal and opposite contact forces
    const p_x = a.mass * a.vx + b.mass * b.vx;
    // a far-apart pair must not interact at all
    f.clear();
    const c = f.spawn(world, 40, 100);
    f.spawn(world, 160, 100);
    for (let i = 0; i < 120; i++) f.step(1 / 240, world);
    const drift = Math.abs(c.pose.x - 40);
    p.gravity = g0;
    f.clear();
    return { gap0, gap1, p_x, drift, avx: a.vx, bvx: b.vx };
  });
  check("overlapping ducks push apart",
    coll.gap1 > coll.gap0 * 1.5, `gap ${coll.gap0.toFixed(2)} -> ${coll.gap1.toFixed(2)}`);
  check("both ducks move, in opposite directions",
    coll.avx < 0 && coll.bvx > 0, `vx ${coll.avx.toFixed(2)} / ${coll.bvx.toFixed(2)}`);
  check("contact conserves momentum",
    Math.abs(coll.p_x) < 1e-6 * Math.max(1, Math.abs(coll.avx)), `p ${coll.p_x.toExponential(1)}`);
  check("separated ducks don't interact", coll.drift < 1e-9, `drift ${coll.drift.toExponential(1)}`);
} finally {
  await browser.close();
}
console.log(failures ? `${failures} FAILED` : "ALL INPUT CHECKS PASSED");
process.exit(failures ? 1 : 0);
