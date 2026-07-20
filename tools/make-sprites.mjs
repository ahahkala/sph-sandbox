// Generates the rigid-body sprites in assets/ as transparent PNGs.
// Everything is drawn analytically in the body's own local frame (the same
// frame as its hull circles), so the artwork lines up with the physics by
// construction: the painter samples BODY_TYPES[name].rect and nothing else
// needs to agree on a pixel size.
//
// Run: node tools/make-sprites.mjs   (only needed when the shapes change;
// the PNGs are committed)

import { deflateSync } from "node:zlib";
import { writeFileSync, mkdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { BODY_TYPES } from "../src/sim/bodytypes.js";

const PX_PER_UNIT = 128;
const SS = 4; // supersampling per axis

// --- PNG encoding ----------------------------------------------------------

const CRC_TABLE = (() => {
  const t = new Int32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c;
  }
  return t;
})();

const crc32 = (buf) => {
  let c = -1;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ -1) >>> 0;
};

const chunk = (type, data) => {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const body = Buffer.concat([Buffer.from(type, "ascii"), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(body));
  return Buffer.concat([len, body, crc]);
};

// rgba: Uint8Array of w*h*4, straight (non-premultiplied) alpha
function encodePNG(rgba, w, h) {
  const raw = Buffer.alloc(h * (w * 4 + 1));
  for (let y = 0; y < h; y++) {
    raw[y * (w * 4 + 1)] = 0; // filter: none
    Buffer.from(rgba.buffer, y * w * 4, w * 4).copy(raw, y * (w * 4 + 1) + 1);
  }
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(w, 0);
  ihdr.writeUInt32BE(h, 4);
  ihdr[8] = 8; // bit depth
  ihdr[9] = 6; // color type: RGBA
  return Buffer.concat([
    Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]),
    chunk("IHDR", ihdr),
    chunk("IDAT", deflateSync(raw, { level: 9 })),
    chunk("IEND", Buffer.alloc(0)),
  ]);
}

// --- painter ---------------------------------------------------------------

// Layers are painted back to front; each is {hit(x, y) -> color | null} in
// local units with y pointing UP (the mirror of the y-down runtime frame,
// which is what the shape math below reads naturally).
function paint(rect, layers) {
  const w = Math.round((rect.x1 - rect.x0) * PX_PER_UNIT);
  const h = Math.round((rect.y1 - rect.y0) * PX_PER_UNIT);
  const out = new Uint8Array(w * h * 4);
  for (let py = 0; py < h; py++) {
    for (let px = 0; px < w; px++) {
      let r = 0, g = 0, b = 0, a = 0; // premultiplied accumulation
      for (let sy = 0; sy < SS; sy++) {
        for (let sx = 0; sx < SS; sx++) {
          const x = rect.x0 + ((px + (sx + 0.5) / SS) / PX_PER_UNIT);
          const yDown = rect.y0 + ((py + (sy + 0.5) / SS) / PX_PER_UNIT);
          let col = null;
          for (let i = layers.length - 1; i >= 0 && !col; i--) col = layers[i](x, -yDown);
          if (col) {
            r += col[0];
            g += col[1];
            b += col[2];
            a += 1;
          }
        }
      }
      const n = SS * SS;
      const i = (py * w + px) * 4;
      // un-premultiply: averaging premultiplied keeps edges from fringing
      out[i] = a ? Math.round(255 * Math.min(r / a, 1)) : 0;
      out[i + 1] = a ? Math.round(255 * Math.min(g / a, 1)) : 0;
      out[i + 2] = a ? Math.round(255 * Math.min(b / a, 1)) : 0;
      out[i + 3] = Math.round((255 * a) / n);
    }
  }
  return { data: out, w, h };
}

const circle = (cx, cy, r) => (x, y) => (x - cx) ** 2 + (y - cy) ** 2 <= r * r;

const ellipse = (cx, cy, rx, ry, rot) => (x, y) => {
  const dx = x - cx, dy = y - cy;
  const c = Math.cos(-rot), s = Math.sin(-rot);
  return ((dx * c - dy * s) / rx) ** 2 + ((dx * s + dy * c) / ry) ** 2 <= 1;
};

const polygon = (pts) => (x, y) => {
  let inside = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
};

// deterministic value noise, for a bit of surface grain
const hash = (x, y) => {
  const s = Math.sin(x * 127.1 + y * 311.7) * 43758.5453;
  return s - Math.floor(s);
};

const mixc = (a, b, t) => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];

const layer = (test, color) => (x, y) => (test(x, y) ? (typeof color === "function" ? color(x, y) : color) : null);

// --- duck ------------------------------------------------------------------
// Same silhouette the old polygon mesh had: a squashed ellipse with a tail
// bump swept up-back, plus head, wing, beak and eye.

function duckBody(grow = 0) {
  return (x, y) => {
    const t = Math.atan2(y, x);
    const ct = Math.cos(t), st = Math.sin(t);
    let r = 1 / Math.sqrt((ct / 1.18) ** 2 + (st / 0.88) ** 2);
    let d = t - 2.65; // tail direction
    if (d > Math.PI) d -= 2 * Math.PI;
    if (d < -Math.PI) d += 2 * Math.PI;
    r += 0.5 * Math.exp(-((d / 0.38) ** 2));
    return Math.hypot(x, y) <= r + grow;
  };
}

const DUCK_BEAK = [
  [1.04, 1.24],
  [1.52, 1.12],
  [1.5, 1.0],
  [1.04, 0.98],
];
const grownPoly = (pts, e) => {
  const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length;
  const cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
  return pts.map(([x, y]) => {
    const len = Math.hypot(x - cx, y - cy) || 1;
    return [x + ((x - cx) / len) * e, y + ((y - cy) / len) * e];
  });
};

function duckLayers() {
  const OUT = [0.35, 0.2, 0.03];
  const YELLOW = [1.0, 0.8, 0.12];
  const WING = [0.9, 0.64, 0.1];
  const BEAK = [0.96, 0.47, 0.1];
  const EYE = [0.1, 0.09, 0.08];
  const E = 0.05; // outline width
  // vertical shade, the same ramp the old vertex colors used
  const shade = (col) => (x, y) => mixc(
    col.map((c) => c * 0.78),
    col,
    Math.min(Math.max((y + 0.7) / 2.1, 0), 1),
  );
  const body = duckBody(), head = circle(0.62, 1.12, 0.5), beak = polygon(DUCK_BEAK);
  const beakEdge = polygon(grownPoly(DUCK_BEAK, E));
  // one outline around the whole silhouette (head and body merge into a
  // single blob, the way a moulded rubber duck does) plus one around the beak
  const inside = (x, y) => body(x, y) || head(x, y) || beak(x, y);
  const grown = (x, y) => duckBody(E)(x, y) || circle(0.62, 1.12, 0.5 + E)(x, y) || beakEdge(x, y);
  return [
    layer((x, y) => grown(x, y) && !inside(x, y), OUT),
    layer(body, shade(YELLOW)),
    layer(head, shade(YELLOW)),
    layer(ellipse(-0.22, 0.28, 0.52, 0.31, -0.44), shade(WING)),
    layer((x, y) => beakEdge(x, y) && !beak(x, y), OUT),
    layer(beak, BEAK),
    layer(circle(0.74, 1.28, 0.085), EYE),
    layer(circle(0.715, 1.305, 0.031), [1, 1, 1]), // catchlight
    layer(ellipse(-0.3, 0.62, 0.3, 0.12, 0.3), [1, 0.95, 0.62]), // sheen
  ];
}

// --- brick -----------------------------------------------------------------

function brickLayers() {
  const HW = 1.0, HH = 0.5, R = 0.09; // half extents, corner radius
  const rounded = (grow) => (x, y) => {
    const dx = Math.abs(x) - (HW + grow - R), dy = Math.abs(y) - (HH + grow - R);
    const ox = Math.max(dx, 0), oy = Math.max(dy, 0);
    return Math.hypot(ox, oy) + Math.min(Math.max(dx, dy), 0) <= R;
  };
  const CLAY = [0.66, 0.26, 0.19];
  const DARK = [0.3, 0.11, 0.08];
  const face = (x, y) => {
    // top-lit gradient plus a little grain, so it doesn't read as flat plastic
    const grain = hash(Math.floor(x * 46), Math.floor(y * 46));
    const lit = mixc(CLAY, [0.82, 0.42, 0.31], Math.min(Math.max((y + HH) / (2 * HH), 0), 1) * 0.55);
    return mixc(lit, DARK, grain * 0.16);
  };
  return [
    layer(rounded(0.05), DARK),
    layer(rounded(0), face),
    // chipped highlight along the top edge and shadow along the bottom
    layer((x, y) => rounded(0)(x, y) && y > HH - 0.07, [0.88, 0.5, 0.38]),
    layer((x, y) => rounded(0)(x, y) && y < -HH + 0.06, [0.4, 0.15, 0.11]),
  ];
}

// --- emit ------------------------------------------------------------------

const here = dirname(fileURLToPath(import.meta.url));
const assets = join(here, "..", "assets");
mkdirSync(assets, { recursive: true });

for (const [name, layers] of Object.entries({ duck: duckLayers(), brick: brickLayers() })) {
  const { rect } = BODY_TYPES[name];
  const img = paint(rect, layers);
  const file = join(assets, name + ".png");
  writeFileSync(file, encodePNG(img.data, img.w, img.h));
  console.log(`${name}.png  ${img.w}x${img.h}`);
}
