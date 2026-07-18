// Runs a browser test page headlessly and reports its <pre id="out"> text.
// Test pages signal completion by setting document.title to GPU-PASS,
// GPU-FAIL, or GPU-SKIP.
//
// usage: node tools/test-headless.mjs [pagePath] [timeoutMs] [sw]
//   pagePath  — default test/gpu.html
//   timeoutMs — wait for completion, default 480000
//   sw        — pass "sw" to force SwiftShader (default: hardware GPU,
//               with WebGPU enabled)
import puppeteer from "puppeteer-core";
import { chromePath, ensureServer, BASE } from "./chrome.mjs";

const page = process.argv[2] ?? "test/gpu.html";
const timeout = Number(process.argv[3]) || 480000;
const software = process.argv.includes("sw");

await ensureServer();

const args = ["--enable-unsafe-webgpu", "--enable-features=Vulkan"];
if (software) args.push("--disable-gpu");

const browser = await puppeteer.launch({
  executablePath: chromePath(),
  headless: true,
  args,
  protocolTimeout: timeout + 60000,
});
let ok = false;
try {
  const tab = await browser.newPage();
  tab.on("pageerror", (err) => console.error("page error:", err.message));
  tab.on("console", (msg) => {
    if (msg.type() === "error" || msg.type() === "warn") {
      console.error(`[page ${msg.type()}]`, msg.text().slice(0, 1500));
    }
  });
  await tab.goto(`${BASE}/${page}`, { waitUntil: "load", timeout: 120000 });
  await tab.waitForFunction(
    () => /^GPU-(PASS|FAIL|SKIP)$/.test(document.title),
    { timeout, polling: 500 },
  );
  const text = await tab.$eval("#out", (el) => el.textContent);
  console.log(text);
  const title = await tab.title();
  ok = title !== "GPU-FAIL" && !/FAIL/.test(text);
  if (title === "GPU-SKIP") console.log("(skipped — treated as pass for CI)");
} catch (err) {
  console.error("test run failed:", err.message);
} finally {
  await browser.close();
}
process.exit(ok ? 0 : 1);
