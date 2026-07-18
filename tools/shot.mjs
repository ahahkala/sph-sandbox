// Headless screenshot of the sim for visual validation. Uses puppeteer-core
// so async backends (WebGPU) boot fully before the capture.
// usage: node tools/shot.mjs "?n=25000&t=1.5" [outfile] [sw]
//   query   — path + query appended to http://localhost:8123/
//   outfile — default shots/last.png (overwritten each run)
//   sw      — pass "sw" to force SwiftShader (deterministic, slow)
import puppeteer from "puppeteer-core";
import { mkdirSync } from "node:fs";
import { resolve, dirname } from "node:path";
import { chromePath, ensureServer, BASE } from "./chrome.mjs";

const query = process.argv[2] ?? "";
const outArg = process.argv.find((a, i) => i >= 3 && a !== "sw");
const out = resolve(outArg ?? "shots/last.png");
const software = process.argv.includes("sw");

await ensureServer();
mkdirSync(dirname(out), { recursive: true });

const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan", "--hide-scrollbars"];
if (software) args.push("--disable-gpu");

const browser = await puppeteer.launch({
  executablePath: chromePath(),
  headless: true,
  args,
  protocolTimeout: 660000,
});
try {
  const tab = await browser.newPage();
  await tab.setViewport({ width: 1280, height: 800 });
  tab.on("pageerror", (err) => console.error("page error:", err.message));
  tab.on("console", (msg) => {
    if (msg.type() === "error") console.error("[page error]", msg.text().slice(0, 800));
  });
  await tab.goto(BASE + "/" + query, { waitUntil: "load", timeout: 120000 });
  // wait for the app to render a few frames (covers ?t= fast-forward too)
  await tab.waitForFunction(() => (window.__frames || 0) >= 5, { timeout: 600000, polling: 200 });
  await tab.screenshot({ path: out });
  console.log(`screenshot -> ${out}`);
} finally {
  await browser.close();
}
