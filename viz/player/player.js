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
const ISO = { N: 72, LEVELS: 7, TW: 10.6, TH: 5.3, STEP: 13, WATER_LVL: 0.6, OX: 480, OY: 132 };
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
  const levelAt = (i, j) =>
    coarseWet[j * N + i] > 0.5 ? ISO.WATER_LVL : Math.round(coarseH[j * N + i] * (ISO.LEVELS - 1));
  const base = (i, j, lvl) => [ISO.OX + (i - j) * ISO.TW, ISO.OY + (i + j) * ISO.TH - lvl * ISO.STEP];

  // Size an oversized canvas that holds the whole diamond (plus margin), then
  // shift the projection so every tile lands in-bounds. The window the player
  // pans is 960 wide, so pad to at least that in each dimension.
  let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
  for (let j = 0; j <= N; j++) for (let i = 0; i <= N; i++) {
    const [x, y] = base(i, j, 0);
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
  }
  const M = 90;
  const W = Math.max(Math.ceil(maxX - minX) + 2 * M, 1000);
  const H = Math.max(Math.ceil(maxY - minY) + 2 * M, 1000);
  const offX = (W - (maxX - minX)) / 2 - minX, offY = (H - (maxY - minY)) / 2 - minY;
  const project = (i, j, lvl) => { const p = base(i, j, lvl); return [p[0] + offX, p[1] + offY]; };

  const cnv = document.createElement("canvas");
  cnv.width = W; cnv.height = H;
  const c = cnv.getContext("2d");
  const sky = c.createLinearGradient(0, 0, 0, H);
  sky.addColorStop(0, "#E9E4F5"); sky.addColorStop(1, "#CFC6E8");
  c.fillStyle = sky; c.fillRect(0, 0, W, H);

  const poly = (pts, fill, stroke) => {
    c.beginPath(); c.moveTo(pts[0][0], pts[0][1]);
    for (let k = 1; k < pts.length; k++) c.lineTo(pts[k][0], pts[k][1]);
    c.closePath(); c.fillStyle = fill; c.fill();
    if (stroke) { c.strokeStyle = stroke; c.lineWidth = 1.2; c.stroke(); }
  };

  const order = [];
  for (let j = 0; j < N; j++) for (let i = 0; i < N; i++) order.push([i, j]);
  order.sort((p, q) => (p[0] + p[1]) - (q[0] + q[1]));   // painter: far -> near
  for (const [i, j] of order) {
    const lvl = levelAt(i, j);
    const wet = coarseWet[j * N + i] > 0.5;
    const hn = coarseH[j * N + i];
    const A = project(i, j, lvl), B = project(i + 1, j, lvl);
    const C = project(i + 1, j + 1, lvl), D = project(i, j + 1, lvl);
    const gB = project(i + 1, j, 0), gC = project(i + 1, j + 1, 0), gD = project(i, j + 1, 0);
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
      c.strokeStyle = rgb(lerp3(top, GRASS, 0.5)); c.lineWidth = 2;
      c.beginPath(); c.moveTo(D[0], D[1]); c.lineTo(A[0], A[1]); c.lineTo(B[0], B[1]); c.stroke();
      c.strokeStyle = rgb(EDGE_LIT); c.lineWidth = 1;
      c.beginPath(); c.moveTo(D[0], D[1] - 1); c.lineTo(A[0], A[1] - 1); c.lineTo(B[0], B[1] - 1); c.stroke();
    }
  }

  const surfaceAt = (u, v) => {
    const gi = Math.max(0, Math.min(N - 1, Math.round(u * (N - 1))));
    const gj = Math.max(0, Math.min(N - 1, Math.round(v * (N - 1))));
    return project(gi + 0.5, gj + 0.5, levelAt(gi, gj));
  };
  return { terrain: cnv, surfaceAt, w: W, h: H };
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
    ctx.strokeStyle = `rgba(138,114,192,${(0.34 * a * a).toFixed(3)})`;
    ctx.stroke();
  }
}

// Gold coins (food pellets) sitting on the terraces, gently bobbing.
function drawCoins(ctx, coins, view, scale, t) {
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
  }
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
    this.cloudPts = this.makeClouds();
  }

  makeClouds() {
    const r = mulberry32(99);
    const pts = [];
    for (let i = 0; i < 130; i++) {
      pts.push({
        x0: 0.5 + (r() - 0.5) * 0.55, y0: 0.5 + (r() - 0.5) * 0.5,
        jx: (r() - 0.5) * 0.16, jy: (r() - 0.5) * 0.16,
        side: i % 2, r: 3.4 + r() * 3.2
      });
    }
    return pts;
  }

  beatAt(t) {
    const bs = this.beats.beats;
    for (const b of bs) if (t >= b.t0 && t < b.t1) return b;
    return bs[bs.length - 1];
  }

  trajIndex(t, traj) {
    const step = this.scene.step_ms || 100;
    return Math.min(traj.length - 1, Math.floor(t / step));
  }

  paintWorld(t, beat) {
    const ctx = this.ctx;
    ctx.clearRect(0, 0, 960, 960);
    const mode = beat.world ? beat.world.mode : "none";
    if (mode === "none") return;

    const tl = t - beat.t0;
    const zoom = beat.world.zoom
      ? lerp(beat.world.zoom[0], beat.world.zoom[1], easeInOut(ramp(t, beat.t0, beat.t1)))
      : 1;

    const drawPatch = (vx, vw, key) => {
      const traj = this.scene.trajs[key];
      const idx = this.trajIndex(t, traj);
      const p = traj[idx];
      // Follow-cam: pan a sw-wide window over the oversized baked canvas so the
      // creature lands at this panel's centre (vx+vw/2, ~mid), clamped at edges.
      const sw = 960 / zoom;
      const dx0 = vx - (960 - vw) / 2;
      const pcx = vx + vw / 2, pcy = 480;
      const [cx, cy] = this.iso.surfaceAt(p[0], p[1]);
      let sx = cx - (sw * (pcx - dx0)) / 960;
      let sy = cy - (sw * pcy) / 960;
      sx = Math.max(0, Math.min(this.iso.w - sw, sx));
      sy = Math.max(0, Math.min(this.iso.h - sw, sy));
      ctx.save();
      ctx.beginPath(); ctx.rect(vx, 0, vw, 960); ctx.clip();
      ctx.drawImage(this.terrain, sx, sy, sw, sw, dx0, 0, 960, 960);
      const map = (tx, ty) => [dx0 + ((tx - sx) / sw) * 960, ((ty - sy) / sw) * 960];
      const view = { pt: (u, v) => map(...this.iso.surfaceAt(u, v)) };
      const worldScale = 960 / sw;
      const sc = Math.max(2, Math.round(4 * worldScale));
      const pt = this.scene.pellets_t ? this.scene.pellets_t[key] : null;
      const pellets = pt ? pt[Math.min(pt.length - 1, idx)] : this.scene.pellets;
      drawCoins(ctx, pellets, view, worldScale, t);
      drawTrail(ctx, traj, idx, view);
      const cp = view.pt(p[0], p[1]);
      drawCreature(ctx, this.creature, beat.world.stage || 0, cp[0], cp[1], t, sc);
      ctx.restore();
    };

    if (mode === "single") {
      drawPatch(0, 960, "auth");
      if (beat.world.energy) this.paintEnergy(t, beat);
    } else if (mode === "split") {
      drawPatch(0, 477, "auth");
      ctx.fillStyle = "#E8E5F1";
      ctx.fillRect(477, 0, 6, 960);
      drawPatch(483, 477, "surr");
    } else if (mode === "cloud") {
      ctx.save();
      ctx.globalAlpha = 0.16;
      const fs = 960 / this.iso.w;
      ctx.drawImage(this.terrain, 0, 0, this.iso.w, this.iso.h,
        0, (960 - this.iso.h * fs) / 2, 960, this.iso.h * fs);
      ctx.restore();
      const sw = beat.gauge ? beat.gauge.sweep : [0, 1];
      const p = easeInOut(ramp(tl, sw[0], sw[1]));
      for (const q of this.cloudPts) {
        const dx = q.side === 0 ? -0.17 : 0.17;
        const x = (q.x0 + q.jx * (1 - p) + dx * p) * 960;
        const y = (q.y0 + q.jy * (1 - p) + (q.side === 0 ? -0.05 : 0.05) * p) * 960;
        ctx.beginPath();
        ctx.arc(x, y, q.r, 0, 2 * Math.PI);
        ctx.fillStyle = q.side === 0 ? "rgba(138,114,192,0.75)" : "rgba(95,130,198,0.75)";
        ctx.fill();
      }
    }
  }

  paintEnergy(t, beat) {
    const ctx = this.ctx;
    const traj = this.scene.trajs.auth;
    const e = traj[this.trajIndex(t, traj)][3];
    const x = 36, y = 34, w = 240, h = 12;
    const alpha = ramp(t, beat.t0, beat.t0 + 700);
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = "rgba(200,195,221,0.55)";
    ctx.beginPath(); ctx.roundRect(x, y, w, h, 999); ctx.fill();
    const grad = ctx.createLinearGradient(x, y, x + w, y);
    grad.addColorStop(0, "#9079C8"); grad.addColorStop(1, "#5F82C6");
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
      $("kicker").textContent = cap.kicker || "";
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
      $("caption").style.top = "1160px";
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
