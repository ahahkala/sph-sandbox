// Headless check for duck interaction wiring (keyboard thrust + drag). Live rAF physics doesn't
// advance reliably headless (see CLAUDE.md), so real keyboard/mouse events
// verify the input plumbing and duck.step is driven directly for dynamics.
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
  await tab.goto(BASE + "/?n=25000&t=6", { waitUntil: "load", timeout: 120000 });
  await tab.waitForFunction(() => (window.__frames || 0) >= 10, { timeout: 300000, polling: 200 });

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
    const d = window.__duck, p = window.__params;
    const world = { w: p.duckPose.x * 2, h: p.duckPose.y * 2 }; // roomy box
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
    const p = window.__params.duckPose;
    return { cw: c.width, ch: c.height, px: p.x, py: p.y };
  });
  const wh = 70 * Math.sqrt(25000 / 3000);
  const ww = wh * (dims.cw / dims.ch);
  const sx = (dims.px / ww) * dims.cw, sy = (dims.py / wh) * dims.ch;
  await tab.mouse.move(sx, sy);
  await tab.mouse.down();
  let grab = await tab.evaluate(() => window.__duck.grab && { tx: window.__duck.grab.tx });
  check("mousedown on duck grabs it", !!grab, JSON.stringify(grab));
  await tab.mouse.move(sx + 150, sy - 60);
  const grab2 = await tab.evaluate(() => window.__duck.grab && { tx: window.__duck.grab.tx });
  check("mousemove updates grab target", grab2 && grab2.tx > grab.tx,
    `tx ${grab?.tx?.toFixed(1)} -> ${grab2?.tx?.toFixed(1)}`);
  // grabbed drag spring pulls the duck toward the cursor when stepped
  const dvx = await tab.evaluate(() => {
    const d = window.__duck, p = window.__params;
    const world = { w: p.duckPose.x * 4, h: p.duckPose.y * 4 };
    const vx0 = d.vx;
    for (let i = 0; i < 10; i++) d.step(1 / 60, world, null);
    return d.vx - vx0;
  });
  check("drag spring pulls duck toward cursor", dvx > 0.5, `dvx ${dvx.toFixed(2)}`);
  await tab.mouse.up();
  grab = await tab.evaluate(() => window.__duck.grab);
  check("mouseup releases the grab", grab === null);

  // pointer down away from the duck should still push fluid, not grab
  await tab.mouse.move(sx - 300, sy - 300);
  await tab.mouse.down();
  const fluidPtr = await tab.evaluate(() => !window.__duck.grab);
  check("mousedown off-duck doesn't grab", fluidPtr);
  await tab.mouse.up();
} finally {
  await browser.close();
}
console.log(failures ? `${failures} FAILED` : "ALL INPUT CHECKS PASSED");
process.exit(failures ? 1 : 0);
