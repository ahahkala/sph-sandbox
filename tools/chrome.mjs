// Shared helpers for headless-Chrome validation scripts: locate Chrome,
// make sure the dev server is up (auto-starting it if needed).
import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
import { fileURLToPath } from "node:url";

export const PORT = 8123;
export const BASE = `http://localhost:${PORT}`;

export function chromePath() {
  const candidates = [
    process.env.CHROME,
    "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe",
    "C:\\Program Files (x86)\\Google\\Chrome\\Application\\chrome.exe",
    "/usr/bin/google-chrome",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
  ];
  for (const p of candidates) if (p && existsSync(p)) return p;
  throw new Error("Chrome not found; set CHROME env var");
}

export async function ensureServer() {
  const alive = async () => {
    try {
      const res = await fetch(BASE + "/index.html", { signal: AbortSignal.timeout(1500) });
      return res.ok;
    } catch {
      return false;
    }
  };
  if (await alive()) return;
  const serveScript = fileURLToPath(new URL("./serve.mjs", import.meta.url));
  spawn(process.execPath, [serveScript], { detached: true, stdio: "ignore" }).unref();
  for (let i = 0; i < 20; i++) {
    await new Promise((r) => setTimeout(r, 250));
    if (await alive()) return;
  }
  throw new Error("could not start dev server");
}
