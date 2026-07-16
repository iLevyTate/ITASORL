/* ItaSoRL outreach film player.
   Deterministic: every frame is a pure function of virtual time t (ms).
   ?t=<ms> renders one static frame; ?play=1 free-runs; window.__seek(t) is
   the capture hook. No Date.now(), no unseeded randomness in the scene. */

"use strict";

// ---------------------------------------------------------------- utilities

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const clamp01 = (x) => Math.max(0, Math.min(1, x));
const lerp = (a, b, p) => a + (b - a) * p;
const easeInOut = (p) => (p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2);
const easeOut = (p) => 1 - Math.pow(1 - p, 3);

function ramp(t, a, b) { return b <= a ? 1 : clamp01((t - a) / (b - a)); }

// Headline text uses ((...)) to mark the one gradient phrase per beat.
// Built with DOM nodes (no innerHTML) so content stays inert.
function setHeadline(el, text) {
  el.replaceChildren();
  const parts = String(text).split(/\(\(|\)\)/);
  parts.forEach((part, i) => {
    if (!part) return;
    if (i % 2 === 1) {
      const span = document.createElement("span");
      span.className = "grad";
      span.textContent = part;
      el.appendChild(span);
    } else {
      el.appendChild(document.createTextNode(part));
    }
  });
}

// ------------------------------------------------------------- scene data

// Fallback scene: deterministic synthesized terrain + wandering trajectory,
// used until viz/collect.py writes ../data/scene.json from the real world.
function placeholderScene() {
  const n = 160;
  const rng = mulberry32(7);
  const bumps = [];
  for (let i = 0; i < 10; i++) {
    bumps.push({ x: rng(), y: rng(), s: 0.08 + 0.22 * rng(), a: rng() * 2 - 1 });
  }
  const height = new Float32Array(n * n);
  let mn = 1e9, mx = -1e9;
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) {
      const x = i / (n - 1), y = j / (n - 1);
      let h = 0;
      for (const b of bumps) {
        const d2 = (x - b.x) ** 2 + (y - b.y) ** 2;
        h += b.a * Math.exp(-d2 / (2 * b.s * b.s));
      }
      height[j * n + i] = h;
      if (h < mn) mn = h; if (h > mx) mx = h;
    }
  }
  const wet = new Float32Array(n * n);
  for (let k = 0; k < n * n; k++) {
    height[k] = (height[k] - mn) / (mx - mn);
    wet[k] = height[k] < 0.30 ? 1 : 0;
  }
  const pellets = [];
  const prng = mulberry32(21);
  while (pellets.length < 26) {
    const x = prng(), y = prng();
    const gi = Math.min(n - 1, Math.round(x * (n - 1)));
    const gj = Math.min(n - 1, Math.round(y * (n - 1)));
    if (wet[gj * n + gi] < 0.5) pellets.push([x, y]);
  }
  function walk(seed) {
    const r = mulberry32(seed);
    const pts = [];
    let x = 0.5, y = 0.55, vx = 0.004, vy = 0.0;
    let energy = 0.8;
    for (let s = 0; s < 900; s++) {
      vx += (r() - 0.5) * 0.0026; vy += (r() - 0.5) * 0.0026;
      const sp = Math.hypot(vx, vy), cap = 0.0052;
      if (sp > cap) { vx *= cap / sp; vy *= cap / sp; }
      x += vx; y += vy;
      if (x < 0.06 || x > 0.94) vx *= -1;
      if (y < 0.06 || y > 0.94) vy *= -1;
      x = Math.max(0.05, Math.min(0.95, x));
      y = Math.max(0.05, Math.min(0.95, y));
      energy = Math.max(0.15, Math.min(1, energy + (r() - 0.52) * 0.01));
      pts.push([x, y, Math.atan2(vy, vx), energy]);
    }
    return pts;
  }
  return {
    meta: { source: "placeholder", note: "synthesized; replaced by viz/collect.py output" },
    grid_n: n,
    height: Array.from(height),
    wet: Array.from(wet),
    pellets,
    // surr seed chosen so the stand-in creature stays inside the split-panel
    // strip (u 0.25..0.75) for the whole trick beat; walk(11) drifted out.
    trajs: { auth: walk(3), surr: walk(356) },
    step_ms: 100
  };
}

// --------------------------------------------------------------- terrain

// Isometric pseudo-3D world (Super Mario / Monument Valley direction). The
// recorded height grid is quantized into chunky terrace levels and extruded
// into two-tone iso blocks (bright lavender tops, dark dirt sides, a mint
// grass lip and a lit top edge); water sits on a low bench. The diamond is
// wider than the 960 viewport, so it is baked once into an oversized, sky-
// filled canvas and the player pans a 960 window across it to follow the
// creature. surfaceAt(u,v) returns the on-surface pixel (in baked-canvas
// coordinates) for a normalized world coordinate, so food and the creature
// sit on the terraces.
// PAD sizes the ring of clamped-edge tiles baked beyond the play area. The
// follow-camera window is ~890px and the creature sits at its centre, so the
// pad must supply ~445px of ground past any world edge: 445 / TH(5.3) per
// (i+j) step downward means ~44 tiles.
const ISO = { N: 72, PAD: 44, LEVELS: 7, TW: 10.6, TH: 5.3, STEP: 13, WATER_LVL: 0.6, OX: 480, OY: 132 };
const TOP_LO = [120, 108, 184], TOP_HI = [214, 205, 242];   // low -> high ground
const GRASS = [150, 205, 176], DIRT = [96, 74, 126];        // mint lip, dirt sides
const WATER_HI = [120, 176, 232], WATER_LO = [70, 116, 190];
const EDGE_LIT = [248, 245, 255];

const lerp3 = (a, b, p) => [a[0] + (b[0] - a[0]) * p, a[1] + (b[1] - a[1]) * p, a[2] + (b[2] - a[2]) * p];
const rgb = (c) => "rgb(" + (c[0] | 0) + "," + (c[1] | 0) + "," + (c[2] | 0) + ")";
const shadeC = (c, f) => "rgb(" + (c[0] * f | 0) + "," + (c[1] * f | 0) + "," + (c[2] * f | 0) + ")";

// Area-average (height) / any-wet (water) downsample of a flat grid to n1xn1.
function downsampleGrid(flat, n0, n1, mode) {
  const out = new Float64Array(n1 * n1);
  const s = n0 / n1;
  for (let j = 0; j < n1; j++) {
    for (let i = 0; i < n1; i++) {
      let acc = 0, cnt = 0;
      for (let dj = 0; dj < s; dj++) {
        for (let di = 0; di < s; di++) {
          const fi = Math.min(n0 - 1, (i * s + di) | 0);
          const fj = Math.min(n0 - 1, (j * s + dj) | 0);
          acc += flat[fj * n0 + fi]; cnt++;
        }
      }
      out[j * n1 + i] = mode === "max" ? (acc / cnt >= 0.5 ? 1 : 0) : acc / cnt;
    }
  }
  return out;
}

// Bake the iso terrain and hand back a surfaceAt() mapper. Runs at load.
function makeIsoWorld(scene) {
  const N = ISO.N;
  const coarseH = downsampleGrid(scene.height, scene.grid_n, N, "avg");
  const coarseWet = downsampleGrid(scene.wet, scene.grid_n, N, "max");
  // Grid lookups clamp, so PAD rings of tiles beyond the play area repeat the
  // edge terrain: the follow-camera always has ground in frame, never dead sky.
  const cl = (k) => Math.max(0, Math.min(N - 1, k));
  const cell = (i, j) => cl(j) * N + cl(i);
  const levelAt = (i, j) =>
    coarseWet[cell(i, j)] > 0.5 ? ISO.WATER_LVL : Math.round(coarseH[cell(i, j)] * (ISO.LEVELS - 1));
  const base = (i, j, lvl) => [ISO.OX + (i - j) * ISO.TW, ISO.OY + (i + j) * ISO.TH - lvl * ISO.STEP];

  // Size an oversized canvas that holds the whole diamond (plus margin), then
  // shift the projection so every tile lands in-bounds. The window the player
  // pans is 960 wide, so pad to at least that in each dimension.
  let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
  for (let j = -ISO.PAD; j <= N + ISO.PAD; j++) for (let i = -ISO.PAD; i <= N + ISO.PAD; i++) {
    const [x, y] = base(i, j, 0);
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  const M = 90;
  const W = Math.max(Math.ceil(maxX - minX) + 2 * M, 1000);
  const H = Math.max(Math.ceil(maxY - minY) + 2 * M, 1000);
  const offX = (W - (maxX - minX)) / 2 - minX, offY = (H - (maxY - minY)) / 2 - minY;
  const project = (i, j, lvl) => { const p = base(i, j, lvl); return [p[0] + offX, p[1] + offY]; };

  // Play-area (unpadded) bbox in baked coordinates, for whole-world fits.
  let cx0 = 1e9, cx1 = -1e9, cy0 = 1e9, cy1 = -1e9;
  for (let j = 0; j <= N; j++) for (let i = 0; i <= N; i++) {
    const [x, y] = project(i, j, 0);
    if (x < cx0) cx0 = x; if (x > cx1) cx1 = x;
    if (y < cy0) cy0 = y; if (y > cy1) cy1 = y;
  }
  cy0 -= ISO.LEVELS * ISO.STEP;   // terraces rise above ground level
  const core = { x: cx0 - 20, y: cy0 - 10, w: cx1 - cx0 + 40, h: cy1 - cy0 + 30 };

  const cnv = document.createElement("canvas");
  cnv.width = W; cnv.height = H;
  const c = cnv.getContext("2d");
  const sky = c.createLinearGradient(0, 0, 0, H);
  sky.addColorStop(0, "#E9E4F5"); sky.addColorStop(1, "#CFC6E8");
  c.fillStyle = sky; c.fillRect(0, 0, W, H);

  const bakeTiles = (c2, proj, lo, hi) => {
    const poly = (pts, fill, stroke) => {
      c2.beginPath(); c2.moveTo(pts[0][0], pts[0][1]);
      for (let k = 1; k < pts.length; k++) c2.lineTo(pts[k][0], pts[k][1]);
      c2.closePath(); c2.fillStyle = fill; c2.fill();
      if (stroke) { c2.strokeStyle = stroke; c2.lineWidth = 1.2; c2.stroke(); }
    };
    const order = [];
    for (let j = lo; j < hi; j++) for (let i = lo; i < hi; i++) order.push([i, j]);
    order.sort((p, q) => (p[0] + p[1]) - (q[0] + q[1]));   // painter: far -> near
    for (const [i, j] of order) {
      const lvl = levelAt(i, j);
      const wet = coarseWet[cell(i, j)] > 0.5;
      const hn = coarseH[cell(i, j)];
      const A = proj(i, j, lvl), B = proj(i + 1, j, lvl);
      const C = proj(i + 1, j + 1, lvl), D = proj(i, j + 1, lvl);
      const gB = proj(i + 1, j, 0), gC = proj(i + 1, j + 1, 0), gD = proj(i, j + 1, 0);
      if (wet) {
        const wc = lerp3(WATER_LO, WATER_HI, hn);
        poly([B, C, gC, gB], shadeC(wc, 0.60));
        poly([D, C, gC, gD], shadeC(wc, 0.78));
        poly([A, B, C, D], rgb(wc));
        poly([A, B, C, D], "rgba(255,255,255,0)", "rgba(180,214,250,0.7)");
      } else {
        const top = lerp3(TOP_LO, TOP_HI, hn);
        const dirt = lerp3(top, DIRT, 0.55);
        poly([B, C, gC, gB], shadeC(dirt, 0.72));   // right (+i) face
        poly([D, C, gC, gD], shadeC(dirt, 0.88));   // left  (+j) face
        poly([A, B, C, D], rgb(top));               // lit top face
        c2.strokeStyle = rgb(lerp3(top, GRASS, 0.5)); c2.lineWidth = 2;
        c2.beginPath(); c2.moveTo(D[0], D[1]); c2.lineTo(A[0], A[1]); c2.lineTo(B[0], B[1]); c2.stroke();
        c2.strokeStyle = rgb(EDGE_LIT); c2.lineWidth = 1;
        c2.beginPath(); c2.moveTo(D[0], D[1] - 1); c2.lineTo(A[0], A[1] - 1); c2.lineTo(B[0], B[1] - 1); c2.stroke();
      }
    }
  };
  bakeTiles(c, project, -ISO.PAD, N + ISO.PAD);

  // Second bake: just the play-area diamond on a transparent background, for
  // whole-world fits (the scan beat). The padded canvas cannot serve there,
  // because its bounding rectangle is full of pad terrain, not sky.
  const coreCnv = document.createElement("canvas");
  coreCnv.width = Math.ceil(core.w); coreCnv.height = Math.ceil(core.h);
  const projCore = (i, j, lvl) => {
    const p = project(i, j, lvl);
    return [p[0] - core.x, p[1] - core.y];
  };
  bakeTiles(coreCnv.getContext("2d"), projCore, 0, N);

  const surfaceAt = (u, v) => {
    const gi = Math.max(0, Math.min(N - 1, Math.round(u * (N - 1))));
    const gj = Math.max(0, Math.min(N - 1, Math.round(v * (N - 1))));
    return project(gi + 0.5, gj + 0.5, levelAt(gi, gj));
  };
  return { terrain: cnv, surfaceAt, w: W, h: H, core, coreView: coreCnv };
}

// --------------------------------------------------------------- glyphs

// The creature is the Blob pixel sprite (blob-spritesheet.png). Growth stage
// comes from beats.json (world.stage); the morph is teal, which contrasts the
// purple world so the creature reads against the terraces. Pixel art is
// blitted at an integer scale with smoothing off, and idles with a whole-pixel
// bob rather than a fractional scale so the pixels never resample.
const CREATURE_MORPH = "teal";

function shadowEllipse(ctx, x, y, r) {
  ctx.save();
  ctx.globalAlpha = 0.26; ctx.fillStyle = "#241b38";
  ctx.beginPath(); ctx.ellipse(x, y + 4, r, r * 0.48, 0, 0, 2 * Math.PI); ctx.fill();
  ctx.restore();
}

function drawCreature(ctx, creature, stage, x, y, t, sc) {
  const f = creature.byStage[Math.max(0, Math.min(creature.byStage.length - 1, stage))];
  const ax = f.anchor.x - f.x, ay = f.anchor.y - f.y;
  const bob = Math.round(Math.sin((2 * Math.PI * t) / 3200)) * sc;
  shadowEllipse(ctx, x, y, f.w * sc * 0.28);
  ctx.save();
  ctx.imageSmoothingEnabled = false;
  ctx.drawImage(
    creature.img, f.x, f.y, f.w, f.h,
    Math.round(x - ax * sc), Math.round(y - ay * sc) + bob, f.w * sc, f.h * sc
  );
  ctx.restore();
}

function drawTrail(ctx, pts, upto, view) {
  const TAIL = 90;
  const start = Math.max(0, upto - TAIL);
  ctx.lineWidth = 3;
  ctx.lineCap = "round";
  for (let k = start + 1; k <= upto; k++) {
    const a = (k - start) / TAIL;
    const p0 = view.pt(pts[k - 1][0], pts[k - 1][1]);
    const p1 = view.pt(pts[k][0], pts[k][1]);
    ctx.beginPath();
    ctx.moveTo(p0[0], p0[1]); ctx.lineTo(p1[0], p1[1]);
    ctx.strokeStyle = `rgba(52,168,150,${(0.34 * a * a).toFixed(3)})`;
    ctx.stroke();
  }
}

// --------------------------------------------------------------- sensing

// Illustrative sensor fan driven by the creature's *recorded* heading: rays
// probe ahead in world space and every sample is re-projected onto the
// terraces, so "what the creature sees" hugs the real terrain along the real
// trajectory. Calm = teal; alert = hot flash (survival beat, glitch on screen).
const FAN_SPREAD = 1.15, FAN_REACH = 0.115, FAN_RAYS = 5;

function angDiff(a, b) {
  let d = a - b;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return d;
}

function inFan(qx, qy, sense) {
  const dx = qx - sense.u, dy = qy - sense.v;
  return Math.hypot(dx, dy) <= FAN_REACH + 0.015 &&
    Math.abs(angDiff(Math.atan2(dy, dx), sense.heading)) <= FAN_SPREAD / 2 + 0.25;
}

function drawFan(ctx, view, origin, u, v, heading, t, scale, alert) {
  const STEPS = 4;
  const col = alert ? [244, 92, 64] : [22, 158, 132];
  const breathe = 0.8 + 0.2 * Math.sin((2 * Math.PI * t) / 1400);
  const flash = alert ? 0.65 + 0.35 * Math.sin(t / 70) : 1;
  const a = breathe * flash;
  const rays = [];
  for (let k = 0; k < FAN_RAYS; k++) {
    const ang = heading - FAN_SPREAD / 2 + (FAN_SPREAD * k) / (FAN_RAYS - 1);
    const pts = [];
    for (let s = 1; s <= STEPS; s++) {
      const r = (FAN_REACH * s) / STEPS;
      const uu = Math.max(0, Math.min(1, u + Math.cos(ang) * r));
      const vv = Math.max(0, Math.min(1, v + Math.sin(ang) * r));
      pts.push(view.pt(uu, vv));
    }
    rays.push(pts);
  }
  ctx.save();
  ctx.beginPath();
  ctx.moveTo(origin[0], origin[1]);
  for (const pts of rays) ctx.lineTo(pts[STEPS - 1][0], pts[STEPS - 1][1]);
  ctx.closePath();
  ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${(0.17 * a).toFixed(3)})`;
  ctx.fill();
  for (const pts of rays) {
    ctx.beginPath();
    ctx.moveTo(origin[0], origin[1]);
    for (const q of pts) ctx.lineTo(q[0], q[1]);
    ctx.strokeStyle = `rgba(${col[0]},${col[1]},${col[2]},${(0.55 * a).toFixed(3)})`;
    ctx.lineWidth = 2.4 * scale;
    ctx.stroke();
    pts.forEach((q, si) => {
      ctx.beginPath();
      ctx.arc(q[0], q[1], (3.4 - 0.5 * si) * scale, 0, 2 * Math.PI);
      ctx.fillStyle = `rgba(${col[0]},${col[1]},${col[2]},${(0.8 * (1 - si / (STEPS + 1)) * a).toFixed(3)})`;
      ctx.fill();
    });
  }
  ctx.restore();
}

// Gold coins (food pellets) sitting on the terraces, gently bobbing. When a
// sense target is given, coins inside the fan get a pinging teal ring.
function drawCoins(ctx, coins, view, scale, t, sense) {
  for (let k = 0; k < coins.length; k++) {
    const s = view.pt(coins[k][0], coins[k][1]);
    const bob = 3 * scale * Math.sin(t / 420 + k * 1.7);
    const cy = s[1] - 18 * scale - bob;
    shadowEllipse(ctx, s[0], s[1], 6 * scale);
    ctx.beginPath(); ctx.ellipse(s[0], cy, 6 * scale, 8 * scale, 0, 0, 2 * Math.PI);
    ctx.fillStyle = "#B0781C"; ctx.fill();
    ctx.beginPath(); ctx.ellipse(s[0], cy, 4.4 * scale, 6.6 * scale, 0, 0, 2 * Math.PI);
    ctx.fillStyle = "#FFCE46"; ctx.fill();
    ctx.beginPath(); ctx.ellipse(s[0] - 1.4 * scale, cy - 1.6 * scale, 1.4 * scale, 2.4 * scale, 0, 0, 2 * Math.PI);
    ctx.fillStyle = "#FFF0B4"; ctx.fill();
    if (sense && inFan(coins[k][0], coins[k][1], sense)) {
      const rr = (9.5 + 1.6 * Math.sin(t / 130)) * scale;
      ctx.beginPath(); ctx.ellipse(s[0], cy, rr, rr * 1.2, 0, 0, 2 * Math.PI);
      ctx.strokeStyle = "rgba(52,186,160,0.85)";
      ctx.lineWidth = 2.2;
      ctx.stroke();
    }
  }
}

// A pellet present at step k-1 and gone at step k was eaten at step k. That is
// recorded data (pellets_t), not a guessed effect. The world respawns a new
// pellet the same step, so the count stays flat: diff the SETS, not lengths.
// Pop rings linger ~700 ms.
function drawEats(ctx, pelletFrames, idx, view, scale, t, stepMs) {
  if (!pelletFrames) return;
  const POP_MS = 700;
  const back = Math.ceil(POP_MS / stepMs) + 1;
  const hi = Math.min(idx, pelletFrames.length - 1);
  for (let k = Math.max(1, idx - back); k <= hi; k++) {
    const prev = pelletFrames[k - 1], cur = pelletFrames[k];
    if (!prev || !cur) continue;
    const have = new Set(cur.map((q) => q[0] + "," + q[1]));
    for (const q of prev) {
      if (have.has(q[0] + "," + q[1])) continue;
      const age = t - k * stepMs;
      if (age < 0 || age > POP_MS) continue;
      const p01 = age / POP_MS;
      const s = view.pt(q[0], q[1]);
      const cy = s[1] - 18 * scale;
      ctx.beginPath();
      ctx.ellipse(s[0], cy, (8 + 22 * p01) * scale, (6 + 17 * p01) * scale, 0, 0, 2 * Math.PI);
      ctx.strokeStyle = `rgba(255,190,60,${(0.85 * (1 - p01)).toFixed(3)})`;
      ctx.lineWidth = 3;
      ctx.stroke();
      ctx.font = `600 ${Math.round(17 * scale)}px 'IBM Plex Mono', monospace`;
      ctx.fillStyle = `rgba(42,140,120,${(0.9 * (1 - p01)).toFixed(3)})`;
      ctx.fillText("+1", s[0] + 10 * scale, cy - (14 + 20 * p01) * scale);
    }
  }
}

// Deterministic world glitches: from GLITCH_START into the beat, every
// GLITCH_EVERY ms a patch near the creature flickers for GLITCH_MS. Seeded by
// beat and glitch index, so every seek reproduces the same flicker.
const GLITCH_EVERY = 3000, GLITCH_MS = 650, GLITCH_START = 2000;

function drawGlitch(ctx, g, view, scale, t) {
  if (Math.sin(t / 33) <= -0.6) return;   // strobe off-phase
  const r = mulberry32(g.seed * 31 + 5);
  const c = view.pt(g.u, g.v);
  const a = 0.85 * (1 - Math.abs(g.age * 2 - 1));
  for (let k = 0; k < 10; k++) {
    const ox = (r() - 0.5) * 116 * scale, oy = (r() - 0.5) * 64 * scale;
    const w = (14 + r() * 28) * scale, h = (4 + r() * 8) * scale;
    ctx.fillStyle = k % 2
      ? `rgba(236,84,164,${a.toFixed(3)})`
      : `rgba(112,224,255,${a.toFixed(3)})`;
    ctx.fillRect(c[0] + ox, c[1] + oy - 16 * scale, w, h);
  }
}

// Counterpart ghost (split beat): a dashed ring at the OTHER world's creature
// position, in that world's accent colour. Same start, different physics: the
// ring walks away from the creature as the flaw compounds. Recorded data.
function drawGhost(ctx, gs, cp, scale, t, color) {
  ctx.save();
  if (Math.hypot(gs[0] - cp[0], gs[1] - cp[1]) > 60) {
    ctx.setLineDash([7, 9]);
    ctx.strokeStyle = color.replace("ALPHA", "0.5");
    ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.moveTo(cp[0], cp[1]); ctx.lineTo(gs[0], gs[1]); ctx.stroke();
  }
  const r = Math.max(15, 22 * scale) + 2 * Math.sin(t / 300);
  const gy = gs[1] - 8 * scale;
  ctx.setLineDash([6, 6]);
  ctx.strokeStyle = color.replace("ALPHA", "0.95");
  ctx.lineWidth = 3.5;
  ctx.beginPath(); ctx.ellipse(gs[0], gy, r, r * 0.8, 0, 0, 2 * Math.PI); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = color.replace("ALPHA", "0.9");
  ctx.beginPath(); ctx.arc(gs[0], gy, 4.5, 0, 2 * Math.PI); ctx.fill();
  ctx.restore();
}

// Game-style energy pill above the creature (survival beat). The level is the
// recorded per-step energy, so the on-screen drain is the real starvation arc.
function drawEnergyPill(ctx, x, y, e, scale, t) {
  const w = 64 * scale, h = 9 * scale;
  const px = x - w / 2, py = y - 78 * scale;
  const col = e > 0.5 ? "#35B5A2" : e > 0.25 ? "#E8A13C" : "#E0526E";
  ctx.save();
  ctx.fillStyle = "rgba(42,37,71,0.35)";
  ctx.beginPath(); ctx.roundRect(px - 2, py - 2, w + 4, h + 4, 999); ctx.fill();
  ctx.fillStyle = "rgba(251,250,255,0.75)";
  ctx.beginPath(); ctx.roundRect(px, py, w, h, 999); ctx.fill();
  ctx.fillStyle = col;
  ctx.beginPath(); ctx.roundRect(px, py, Math.max(h, w * e), h, 999); ctx.fill();
  if (e <= 0.25) {
    ctx.strokeStyle = `rgba(224,82,110,${(0.5 + 0.4 * Math.sin(t / 120)).toFixed(3)})`;
    ctx.lineWidth = 2.5;
    ctx.beginPath(); ctx.roundRect(px - 4, py - 4, w + 8, h + 8, 999); ctx.stroke();
  }
  ctx.restore();
}

// Materialization wipe over the FLAWED COPY panel at the start of the split
// beat: a scanline sweeps down, digital slices trail it and decay, then the
// panel settles into a pixel-identical copy.
function drawMaterialize(ctx, vx, vw, tl) {
  const P = clamp01(tl / 1400);
  const edge = P * 960;
  ctx.save();
  ctx.beginPath(); ctx.rect(vx, 0, vw, 960); ctx.clip();
  ctx.fillStyle = "rgba(233,228,245,0.92)";
  ctx.fillRect(vx, edge, vw, 960 - edge);
  const r = mulberry32(77);
  const n = Math.round(8 * (1 - P));
  for (let k = 0; k < n; k++) {
    const y = edge - r() * 120 - 6;
    const h = 3 + r() * 8;
    if (y < 0) continue;
    ctx.fillStyle = k % 2 ? "rgba(236,84,164,0.35)" : "rgba(112,224,255,0.35)";
    ctx.fillRect(vx, y, vw, h);
  }
  const g = ctx.createLinearGradient(0, edge - 26, 0, edge + 4);
  g.addColorStop(0, "rgba(95,130,198,0)");
  g.addColorStop(1, "rgba(95,130,198,0.55)");
  ctx.fillStyle = g;
  ctx.fillRect(vx, edge - 26, vw, 30);
  ctx.fillStyle = "rgba(63,94,178,0.9)";
  ctx.fillRect(vx, edge, vw, 3);
  ctx.restore();
}

// Timed documentary-style callout: a leader line from the element to a small
// mono label pill, so first-time viewers can read the picture. One at a time.
function drawCallout(ctx, x, y, text, dir, a) {
  if (a <= 0) return;
  const lx = x + dir[0], ly = y + dir[1];
  ctx.save();
  ctx.globalAlpha = a;
  ctx.strokeStyle = "rgba(42,37,71,0.6)";
  ctx.lineWidth = 1.6;
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(lx, ly); ctx.stroke();
  ctx.font = "500 16px 'IBM Plex Mono', monospace";
  const wtx = ctx.measureText(text).width;
  const pad = 9;
  const bx = dir[0] >= 0 ? lx + 5 : lx - 5 - wtx - 2 * pad;
  ctx.fillStyle = "rgba(251,250,255,0.93)";
  ctx.beginPath(); ctx.roundRect(bx, ly - 15, wtx + 2 * pad, 30, 9); ctx.fill();
  ctx.strokeStyle = "rgba(42,37,71,0.22)";
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.roundRect(bx, ly - 15, wtx + 2 * pad, 30, 9); ctx.stroke();
  ctx.fillStyle = "#2A2547";
  ctx.textBaseline = "middle";
  ctx.fillText(text, bx + pad, ly + 1);
  ctx.restore();
}

// 400 ms fade in/out inside a [t0, t1] window (beat-relative ms).
function calloutAlpha(tl, t0, t1) {
  if (tl < t0 || tl > t1) return 0;
  return Math.min(1, (tl - t0) / 400, (t1 - tl) / 400);
}

function drawAlertPing(ctx, x, y, scale, age) {
  const pop = Math.min(1, age * 3);
  if (pop <= 0) return;
  const px = x + 44 * scale, py = y - 64 * scale;
  ctx.save();
  ctx.globalAlpha = 0.95 * pop;
  ctx.beginPath(); ctx.arc(px, py, 15 * scale * pop, 0, 2 * Math.PI);
  ctx.fillStyle = "#E0526E"; ctx.fill();
  ctx.fillStyle = "#FFF7F2";
  ctx.font = `700 ${Math.max(8, Math.round(20 * scale * pop))}px 'Hanken Grotesk', sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText("!", px, py + 1);
  ctx.restore();
}

// ---------------------------------------------------------------- player

const $ = (id) => document.getElementById(id);

class Player {
  constructor(beats, scene, creature) {
    this.beats = beats;
    this.scene = scene;
    this.creature = creature;
    this.canvas = $("world");
    this.ctx = this.canvas.getContext("2d");
    const iso = makeIsoWorld(scene);
    this.terrain = iso.terrain;
    this.iso = iso;
    this.scanFlags = null;   // lazily built, deterministic (seeded)
  }

  beatAt(t) {
    const bs = this.beats.beats;
    for (const b of bs) if (t >= b.t0 && t < b.t1) return b;
    return bs[bs.length - 1];
  }

  paintWorld(t, beat) {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, 960, 960);
    const mode = beat.world ? beat.world.mode : "none";
    if (mode === "none") return;

    const tl = t - beat.t0;
    // worldT0 lets a beat replay a chosen window of the recorded run (film
    // time keeps flowing; world time starts at worldT0). The survival beat
    // uses this to show the hunt-and-decline stretch without the death.
    const wt = beat.world.worldT0 != null ? beat.world.worldT0 + tl : t;
    const zoom = beat.world.zoom
      ? lerp(beat.world.zoom[0], beat.world.zoom[1], easeInOut(ramp(t, beat.t0, beat.t1)))
      : 1;

    // Follow-camera. One target: creature centred. Two targets (split): the
    // SAME window frames both worlds so the panels show the same place, and
    // the camera zooms out just enough to keep both creatures in their strip.
    const camFor = (targets) => {
      let cx = 0, cy = 0;
      for (const q of targets) { cx += q[0]; cy += q[1]; }
      cx /= targets.length; cy /= targets.length;
      let sw = 960 / zoom;
      if (targets.length > 1) {
        const dx = Math.abs(targets[0][0] - targets[1][0]);
        const dy = Math.abs(targets[0][1] - targets[1][1]);
        sw = Math.max(sw, dx * 2.6, dy * 1.7);
      }
      sw = Math.min(sw, this.iso.w, this.iso.h);
      return {
        sw,
        sx: Math.max(0, Math.min(this.iso.w - sw, cx - sw / 2)),
        sy: Math.max(0, Math.min(this.iso.h - sw, cy - sw / 2)),
      };
    };

    // Per-frame interpolation between recorded steps (a screen-space lerp of
    // the terrace-anchored points): the creature, camera, fan, ghost and pill
    // glide instead of jumping once per step. Still a pure function of t.
    const stepMs = this.scene.step_ms || 100;
    const posAt = (traj, wtX) => {
      const f = wtX / stepMs;
      const k = Math.max(0, Math.min(traj.length - 1, Math.floor(f)));
      const k2 = Math.min(traj.length - 1, k + 1);
      const fr = Math.max(0, Math.min(1, f - k));
      const A = this.iso.surfaceAt(traj[k][0], traj[k][1]);
      const B = this.iso.surfaceAt(traj[k2][0], traj[k2][1]);
      let dh = traj[k2][2] - traj[k][2];
      while (dh > Math.PI) dh -= 2 * Math.PI;
      while (dh < -Math.PI) dh += 2 * Math.PI;
      return {
        scr: [A[0] + (B[0] - A[0]) * fr, A[1] + (B[1] - A[1]) * fr],
        u: traj[k][0] + (traj[k2][0] - traj[k][0]) * fr,
        v: traj[k][1] + (traj[k2][1] - traj[k][1]) * fr,
        heading: traj[k][2] + dh * fr,
        energy: traj[k][3] + (traj[k2][3] - traj[k][3]) * fr,
        idx: k,
      };
    };
    const posOf = (traj) => posAt(traj, wt);

    const drawPatch = (vx, vw, key, cam, pos, ghost) => {
      const traj = this.scene.trajs[key];
      const idx = pos.idx;
      const { sx, sy, sw } = cam;
      const dx0 = vx - (960 - vw) / 2;
      ctx.save();
      ctx.beginPath(); ctx.rect(vx, 0, vw, 960); ctx.clip();
      ctx.drawImage(this.terrain, sx, sy, sw, sw, dx0, 0, 960, 960);
      const map = (tx, ty) => [dx0 + ((tx - sx) / sw) * 960, ((ty - sy) / sw) * 960];
      const view = { pt: (u, v) => map(...this.iso.surfaceAt(u, v)) };
      const worldScale = 960 / sw;
      const sc = Math.max(2, Math.round(4 * worldScale));
      // Glitches anchor where the creature WAS when the glitch began, so the
      // flicker and its label stay put instead of trailing the creature.
      const glitchAt = (giMin, giMax, windowMs) => {
        if (!beat.world.glitch || tl < GLITCH_START) return null;
        const gi = Math.floor((tl - GLITCH_START) / GLITCH_EVERY);
        if (gi < giMin || gi > giMax) return null;
        const phase = (tl - GLITCH_START) % GLITCH_EVERY;
        if (phase > windowMs) return null;
        const g0 = posAt(traj, wt - phase);
        const r = mulberry32((beat.t0 / 1000) * 97 + gi * 7919 + 13);
        return {
          u: Math.max(0.06, Math.min(0.94, g0.u + (r() - 0.5) * 0.22)),
          v: Math.max(0.06, Math.min(0.94, g0.v + (r() - 0.5) * 0.22)),
          age: phase / GLITCH_MS,
          phase,
          seed: gi,
        };
      };
      const glitch = glitchAt(0, 99, GLITCH_MS);
      if (glitch) drawGlitch(ctx, glitch, view, worldScale, t);
      const fanMode = beat.world.fan;
      const alert = fanMode === "alert" && !!glitch;
      const pt = this.scene.pellets_t ? this.scene.pellets_t[key] : null;
      const pellets = pt ? pt[Math.min(pt.length - 1, idx)] : this.scene.pellets;
      const sense = fanMode ? { u: pos.u, v: pos.v, heading: pos.heading } : null;
      drawCoins(ctx, pellets, view, worldScale, t, sense);
      drawEats(ctx, pt, idx, view, worldScale, wt, stepMs);
      drawTrail(ctx, traj, idx, view);
      const cp = map(pos.scr[0], pos.scr[1]);
      if (fanMode) drawFan(ctx, view, cp, pos.u, pos.v, pos.heading, t, worldScale, alert);
      drawCreature(ctx, this.creature, beat.world.stage || 0, cp[0], cp[1], t, sc);
      if (beat.world.energy) drawEnergyPill(ctx, cp[0], cp[1], pos.energy, worldScale, t);
      if (ghost) drawGhost(ctx, map(ghost.scr[0], ghost.scr[1]), cp, worldScale, t, ghost.color);
      if (alert) drawAlertPing(ctx, cp[0], cp[1], worldScale, glitch.age);

      // Sequenced callouts (one visible at a time per panel). Anchors are
      // stabilised: screen-smooth or frozen points so labels never hop with
      // tile quantisation, and creature-anchored labels keep a fixed side.
      const pcx2 = vx + vw / 2;
      const label = (x, y, text, a, side) => {
        const s2 = side || (x < pcx2 ? 1 : -1);
        drawCallout(ctx, x, y, text, s2 > 0 ? [46, -36] : [-46, -36], a);
      };
      if (beat.id === "creature") {
        // flat-iso transform of the interpolated heading: smooth every frame
        const du = Math.cos(pos.heading), dv = Math.sin(pos.heading);
        const hx = (du - dv) * ISO.TW, hy = (du + dv) * ISO.TH;
        const hl = Math.hypot(hx, hy) || 1;
        label(cp[0] + (hx / hl) * 150 * worldScale, cp[1] + (hy / hl) * 150 * worldScale,
          "WHAT IT SENSES", calloutAlpha(tl, 2200, 5600));
        // coin chosen once (nearest at the window's start), so no switching
        const selWt = (beat.world.worldT0 != null ? beat.world.worldT0 : beat.t0) + 6400;
        const selIdx = Math.min(traj.length - 1, Math.floor(selWt / stepMs));
        const p6 = traj[selIdx];
        const pel6 = pt ? pt[Math.min(pt.length - 1, selIdx)] : this.scene.pellets;
        let best = null, bd = 1e9;
        for (const q of pel6) {
          const d = Math.hypot(q[0] - p6[0], q[1] - p6[1]);
          if (d < bd) { bd = d; best = q; }
        }
        if (best) {
          const s = view.pt(best[0], best[1]);
          label(s[0], s[1] - 18 * worldScale, "FOOD", calloutAlpha(tl, 6400, 9300));
        }
      } else if (beat.id === "trick" && ghost && ghost.label) {
        const gs2 = map(ghost.scr[0], ghost.scr[1]);
        label(gs2[0], gs2[1] - 14, ghost.label, calloutAlpha(tl, 2200, 6200));
      } else if (beat.id === "nocare") {
        const gl = glitchAt(0, 1, 1900);
        if (gl) {
          const s = view.pt(gl.u, gl.v);
          label(s[0], s[1] - 12, "THE COPY'S FLAW",
            Math.min(1, gl.phase / 300, (1900 - gl.phase) / 400));
        }
        label(cp[0], cp[1] - 66 * worldScale, "IT DOESN'T REACT",
          calloutAlpha(tl, 8600, 12000), 1);
      } else if (beat.id === "survival") {
        label(cp[0] - 34 * worldScale, cp[1] - 82 * worldScale, "ENERGY",
          calloutAlpha(tl, 1200, 3800), -1);
        const gl = glitchAt(1, 1, 1900);
        if (gl) {
          label(cp[0] + 44 * worldScale, cp[1] - 64 * worldScale, "NOW IT NOTICES",
            Math.min(1, gl.phase / 300, (1900 - gl.phase) / 400), 1);
        }
        if (pos.energy < 0.24) {
          label(cp[0] - 34 * worldScale, cp[1] - 82 * worldScale, "STARVING",
            Math.min(1, (0.24 - pos.energy) / 0.05), -1);
        }
      }
      ctx.restore();
    };

    if (mode === "single") {
      const pos = posOf(this.scene.trajs.auth);
      drawPatch(0, 960, "auth", camFor([pos.scr]), pos, null);
      if (beat.world.energy) this.paintEnergy(t, wt, beat);
    } else if (mode === "split") {
      const pa = posOf(this.scene.trajs.auth);
      const pr = posOf(this.scene.trajs.surr);
      const cam = camFor([pa.scr, pr.scr]);
      drawPatch(0, 477, "auth", cam, pa,
        { scr: pr.scr, color: "rgba(224,82,110,ALPHA)", label: "ITS TWIN IN THE COPY" });
      ctx.fillStyle = "#E8E5F1";
      ctx.fillRect(477, 0, 6, 960);
      drawPatch(483, 477, "surr", cam, pr, { scr: pa.scr, color: "rgba(38,166,140,ALPHA)" });
      if (tl < 1400) drawMaterialize(ctx, 483, 477, tl);
    } else if (mode === "physics") {
      // Explainer: a frozen recorded moment. The teal arrow is the creature's
      // actual next-step direction under true physics (recorded heading); the
      // rose dashed arrow is the imitation's answer, rotated a hair to make
      // the compounding-error idea visible. Diagram, labelled as such.
      const traj = this.scene.trajs.auth;
      const p = traj[84];
      const [cx, cy] = this.iso.surfaceAt(p[0], p[1]);
      const sw2 = 960 / 1.15;
      const sx2 = Math.max(0, Math.min(this.iso.w - sw2, cx - sw2 / 2));
      const sy2 = Math.max(0, Math.min(this.iso.h - sw2, cy - sw2 / 2));
      ctx.drawImage(this.terrain, sx2, sy2, sw2, sw2, 0, 0, 960, 960);
      const mapP = (tx, ty) => [((tx - sx2) / sw2) * 960, ((ty - sy2) / sw2) * 960];
      const cp = mapP(cx, cy);
      drawCreature(ctx, this.creature, beat.world.stage || 0, cp[0], cp[1], t, 5);
      const q = this.iso.surfaceAt(
        Math.max(0, Math.min(1, p[0] + Math.cos(p[2]) * 0.08)),
        Math.max(0, Math.min(1, p[1] + Math.sin(p[2]) * 0.08)));
      const qp = mapP(q[0], q[1]);
      const ang = Math.atan2(qp[1] - cp[1], qp[0] - cp[0]);
      const ext = easeOut(ramp(tl, 700, 2000));
      const arrow = (a2, color, dash) => {
        if (ext <= 0) return null;
        const x0 = cp[0] + Math.cos(a2) * 62, y0 = cp[1] + Math.sin(a2) * 46;
        const ex = cp[0] + Math.cos(a2) * (62 + 180 * ext);
        const ey = cp[1] + Math.sin(a2) * (46 + 150 * ext);
        ctx.save();
        ctx.strokeStyle = color; ctx.fillStyle = color;
        ctx.lineWidth = 5;
        if (dash) ctx.setLineDash([11, 9]);
        ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(ex, ey); ctx.stroke();
        ctx.setLineDash([]);
        const ha = Math.atan2(ey - y0, ex - x0);
        ctx.beginPath();
        ctx.moveTo(ex + Math.cos(ha) * 17, ey + Math.sin(ha) * 17);
        ctx.lineTo(ex + Math.cos(ha + 2.5) * 13, ey + Math.sin(ha + 2.5) * 13);
        ctx.lineTo(ex + Math.cos(ha - 2.5) * 13, ey + Math.sin(ha - 2.5) * 13);
        ctx.closePath(); ctx.fill();
        ctx.restore();
        return [ex, ey];
      };
      const tip1 = arrow(ang, "rgba(38,166,140,0.95)", false);
      const tip2 = arrow(ang + 0.45, "rgba(224,82,110,0.95)", true);
      if (tip1) {
        drawCallout(ctx, tip1[0], tip1[1], "TRUE PHYSICS: ITS NEXT STEP",
          tip1[0] < 480 ? [46, -36] : [-46, -36], calloutAlpha(tl, 2100, 5400));
      }
      if (tip2) {
        drawCallout(ctx, tip2[0], tip2[1], "THE IMITATION: A HAIR OFF",
          tip2[0] < 480 ? [46, -36] : [-46, -36], calloutAlpha(tl, 5800, 9400));
      }
    } else if (mode === "scan") {
      // Faded copy world with a scanner band sweeping it; anomaly flags pop
      // where the band has passed. The band uses the same eased ramp that
      // fills the gauge, so 0.99 lands exactly as the sweep completes.
      const core = this.iso.core;
      const fs = Math.min(900 / core.w, 900 / core.h);
      const ox = (960 - core.w * fs) / 2;
      const oy = (960 - core.h * fs) / 2;
      ctx.save();
      ctx.globalAlpha = 0.22;
      ctx.drawImage(this.iso.coreView, ox, oy, core.w * fs, core.h * fs);
      ctx.restore();
      if (!this.scanFlags) {
        // Flags sit ON the copy creature's recorded path: each marks a moment
        // of motion the observer checked against the true physics. The faint
        // line is that replayed path, so the dots point at movement, not land.
        const traj = this.scene.trajs.surr;
        const fit = (u, v) => {
          const s = this.iso.surfaceAt(u, v);
          return [(s[0] - core.x) * fs + ox, (s[1] - core.y) * fs + oy];
        };
        this.scanPath = [];
        for (let k = 0; k < traj.length; k += 2) this.scanPath.push(fit(traj[k][0], traj[k][1]));
        this.scanFlags = [];
        const nf = 14;
        for (let i = 0; i < nf; i++) {
          const k = Math.round(((i + 0.5) / nf) * (traj.length - 1));
          this.scanFlags.push(fit(traj[k][0], traj[k][1]));
        }
      }
      ctx.save();
      ctx.strokeStyle = "rgba(224,82,110,0.30)";
      ctx.lineWidth = 2;
      ctx.beginPath();
      this.scanPath.forEach((q, i) => (i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1])));
      ctx.stroke();
      ctx.restore();
      const swp = beat.gauge ? beat.gauge.sweep : [0, 1];
      const p = easeInOut(ramp(tl, swp[0], swp[1]));
      const xBand = lerp(60, 900, p);
      for (const [fx, fy] of this.scanFlags) {
        if (xBand < fx) continue;
        const pop = Math.min(1, (xBand - fx) / 46);
        ctx.beginPath(); ctx.arc(fx, fy, 9 * pop, 0, 2 * Math.PI);
        ctx.fillStyle = "rgba(224,82,110,0.92)"; ctx.fill();
        ctx.beginPath(); ctx.arc(fx, fy, 12 + 18 * (1 - pop), 0, 2 * Math.PI);
        ctx.strokeStyle = `rgba(224,82,110,${(0.5 * (1 - pop)).toFixed(3)})`;
        ctx.lineWidth = 2; ctx.stroke();
      }
      if (p < 1) {
        const g = ctx.createLinearGradient(xBand - 70, 0, xBand + 14, 0);
        g.addColorStop(0, "rgba(95,130,198,0)");
        g.addColorStop(0.85, "rgba(95,130,198,0.20)");
        g.addColorStop(1, "rgba(95,130,198,0.42)");
        ctx.fillStyle = g;
        ctx.fillRect(xBand - 70, 0, 84, 960);
        ctx.fillStyle = "rgba(63,94,178,0.85)";
        ctx.fillRect(xBand + 12, 0, 2.5, 960);
      }
      // Name the path first, then the first mismatch the scanner catches.
      const midP = this.scanPath[Math.floor(this.scanPath.length / 2)];
      drawCallout(ctx, midP[0], midP[1], "ITS RECORDED PATH",
        midP[0] < 480 ? [46, -36] : [-46, -36], calloutAlpha(tl, 1300, 3200));
      let fmin = null;
      for (const f of this.scanFlags) if (!fmin || f[0] < fmin[0]) fmin = f;
      if (fmin && xBand > fmin[0] + 46) {
        drawCallout(ctx, fmin[0], fmin[1], "MISMATCH FOUND",
          fmin[0] < 480 ? [46, -36] : [-46, -36], calloutAlpha(tl, 3600, 7200));
      }
    }
  }

  paintEnergy(t, wt, beat) {
    const ctx = this.ctx;
    const traj = this.scene.trajs.auth;
    const f = wt / (this.scene.step_ms || 100);
    const k = Math.max(0, Math.min(traj.length - 1, Math.floor(f)));
    const k2 = Math.min(traj.length - 1, k + 1);
    const e = lerp(traj[k][3], traj[k2][3], Math.max(0, Math.min(1, f - k)));
    const x = 36, y = 34, w = 240, h = 12;
    const alpha = ramp(t, beat.t0, beat.t0 + 700);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = "rgba(200,195,221,0.55)";
    ctx.beginPath(); ctx.roundRect(x, y, w, h, 999); ctx.fill();
    const grad = ctx.createLinearGradient(x, y, x + w, y);
    grad.addColorStop(0, "#35B5A2"); grad.addColorStop(1, "#4A9AC8");
    ctx.fillStyle = grad;
    ctx.beginPath(); ctx.roundRect(x, y, Math.max(h, w * e), h, 999); ctx.fill();
    ctx.font = "500 13px 'IBM Plex Mono', monospace";
    ctx.fillStyle = "#6B6588";
    ctx.fillText("ENERGY", x, y - 10);
    ctx.restore();
  }

  chrome(t, beat) {
    const FADE_IN = 550, FADE_OUT = 350;
    const inP = ramp(t, beat.t0, beat.t0 + FADE_IN);
    const outP = 1 - ramp(t, beat.t1 - FADE_OUT, beat.t1);
    const last = beat === this.beats.beats[this.beats.beats.length - 1];
    const vis = Math.min(inP, last ? 1 : outP);

    const cap = beat.caption;
    if (cap) {
      // With the gauge on screen its label already carries the context; the
      // kicker would double it up and crowd the lower third.
      $("kicker").textContent = beat.gauge ? "" : (cap.kicker || "");
      setHeadline($("headline"), cap.headline || "");
      $("subline").textContent = cap.subline || "";
      const el = $("caption");
      el.style.opacity = vis.toFixed(3);
      el.style.transform = `translateY(${(12 * (1 - easeOut(inP))).toFixed(2)}px)`;
    } else {
      $("caption").style.opacity = "0";
    }

    const g = beat.gauge;
    if (g) {
      const el = $("gauge");
      el.style.opacity = vis.toFixed(3);
      $("gauge-label").textContent = g.label;
      const tl = t - beat.t0;
      const p = easeInOut(ramp(tl, g.sweep[0], g.sweep[1]));
      const v = lerp(g.from, g.to, p);
      const pct = clamp01((v - 0.45) / (1.0 - 0.45));
      $("gauge-fill").style.width = (pct * 100).toFixed(2) + "%";
      $("gauge-value").textContent = p >= 1 ? g.display : v.toFixed(2);
      const barPct = clamp01((0.65 - 0.45) / (1.0 - 0.45));
      $("gauge-tick").style.left = "calc(" + (barPct * 100).toFixed(2) + "% - 1px)";
      $("caption").style.top = "1208px";
    } else {
      $("gauge").style.opacity = "0";
      $("caption").style.top = "1108px";
    }

    const chips = beat.world && beat.world.mode === "split" ? vis : 0;
    $("chip-a").style.opacity = (chips * 0.95).toFixed(3);
    $("chip-b").style.opacity = (chips * 0.95).toFixed(3);

    const worldVis = beat.world && beat.world.mode !== "none" ? 1 : 0;
    $("world-card").style.opacity = worldVis ? vis.toFixed(3) : (1 - inP).toFixed(3);

    if (beat.endcard) {
      $("end-headline").textContent = beat.endcard.headline;
      $("end-url").textContent = beat.endcard.url;
      $("end-foot").textContent = beat.endcard.foot;
      $("endcard").style.opacity = inP.toFixed(3);
    } else {
      $("endcard").style.opacity = "0";
    }
  }

  render(t) {
    t = Math.max(0, Math.min(this.beats.duration_ms - 1, t));
    const beat = this.beatAt(t);
    this.paintWorld(t, beat);
    this.chrome(t, beat);
  }
}

// ---------------------------------------------------------------- boot

function fitStage() {
  const s = Math.min(window.innerWidth / 1080, window.innerHeight / 1350);
  $("stage").style.transform = `translate(-50%, -50%) scale(${s})`;
}

async function loadJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: ${res.status}`);
  return res.json();
}

async function loadCreature() {
  const meta = await loadJSON("blob-frames.json");
  const img = new Image();
  img.src = meta.image;
  await img.decode();
  const byStage = meta.frames
    .filter((f) => f.morph === CREATURE_MORPH)
    .sort((a, b) => a.stageIndex - b.stageIndex);
  return { img, byStage };
}

async function main() {
  fitStage();
  window.addEventListener("resize", fitStage);
  const params = new URLSearchParams(location.search);

  const beats = await loadJSON("beats.json");
  let scene;
  try {
    scene = await loadJSON("../data/scene.json");
  } catch {
    scene = placeholderScene();
  }

  const creature = await loadCreature();
  await document.fonts.ready;
  const player = new Player(beats, scene, creature);

  window.__seek = (t) => { player.render(t); return true; };
  window.__duration = beats.duration_ms;
  window.__sceneSource = scene.meta ? scene.meta.source : "unknown";

  const t0 = parseFloat(params.get("t") || "0");
  player.render(t0);

  if (params.get("play") === "1") {
    let start = null;
    const loop = (ts) => {
      if (start === null) start = ts - t0;
      const t = ts - start;
      player.render(t % beats.duration_ms);
      requestAnimationFrame(loop);
    };
    requestAnimationFrame(loop);
  }
  window.__ready = true;
}

main().catch((e) => {
  const pre = document.createElement("pre");
  pre.style.cssText = "padding:2rem;color:#b00";
  pre.textContent = String(e.stack || e);
  document.body.replaceChildren(pre);
  window.__ready = "error";
});
