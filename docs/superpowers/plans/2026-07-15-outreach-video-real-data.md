# Outreach Video Real-Data Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the outreach film's placeholder motion with real recorded runs: a saved survival agent rolled in the authentic and L3-surrogate worlds, exported to `viz/data/scene.json`, consumed by the existing player, and re-rendered to MP4 with number-honesty and determinism checks.

**Architecture:** Record-then-render (spec `docs/superpowers/specs/2026-07-14-outreach-video-design.md`). A new `viz/collect.py` builds the frozen `PatchOfEarthV0` world, rolls `agent_d0.45_s0_survival.pt` (from the completed h8 run) 900 steps in paired auth/surrogate worlds on identical seeds, samples the terrain the player bakes, asserts the film's three numbers against committed artifacts, and writes one committed JSON. The player already tries `fetch('../data/scene.json')` with a placeholder fallback (player.js:419-423); we extend it to per-step pellet state and make the capture refuse to render the placeholder silently.

**Tech Stack:** Python 3.13 (system `python`, no venv), numpy + torch CPU, existing `itasorl` package; vanilla JS canvas player; Playwright + ffmpeg capture (already installed in `viz/player/capture/node_modules`).

---

## Context for the engineer (read first)

- **Work ONLY in the worktree** `C:\Users\benja\Documents\GitHub\ITASORL\.worktrees\outreach-video` (branch `feat/outreach-video`). All relative paths below are from this worktree root unless stated.
- The main checkout at `C:\Users\benja\Documents\GitHub\ITASORL` has a **live GPU run in progress** (`fullruns/l3_h7_heldout`). Consequences: (a) everything here runs on **CPU** (`device="cpu"`, modest thread counts), (b) `fullruns/` in the main checkout is **strictly read-only** — we only read `.pt` files from the *completed* `fullruns/l3_h8_heldout/agents/`, never write there.
- The worktree has no `fullruns/` — the agents dir default resolves through `<worktree>/../../fullruns/...`, which lands in the main checkout. Verified present: `agent_d0.45_s0_survival.pt` ... `s9`.
- `artifacts/expB2/behavior_audit_l3_h8_traces.json` and `docs/FINDINGS.md` are committed, so they exist inside the worktree.
- Verified ground truth (2026-07-15): pooled over the 10 drift=0.45 cells of the audit JSON, `survival.resid_trace = 0.7259` (film shows "0.73") and `untrained.target = 0.4878` (film shows "0.50"); `0.993` appears in `docs/FINDINGS.md` (film shows "0.99"). NOTE: `drift` is a **string** in that JSON — always `float(c["drift"])`.
- Terrain is seed-dependent: `PatchOfEarthV0._init_static_fields` (itasorl/patch_of_earth.py:120-124) draws Fourier coefficients from the `world` RNG during `reset()`. Sample fields ONLY from a world that has been `reset(seeds)` with the same `SeedBundle` as the rollouts.
- Pellets respawn when depleted (patch_of_earth.py:205-207), so 900-step rollouts are sustainable.
- `make_world` (itasorl/experiment_b2.py:96-103) applies `SURVIVAL_METAB`/`SURVIVAL_FOOD` overrides (E0=1.0, basal_E=0.4, n_pellets=24, reach=0.08) and installs the L3 surrogate only when `b2.DRIFT_MODE == "l3"` and `drift_sigma > 0`.
- Spec deviation, intentional: the spec mentions sampling the ambient field; the player never consumes it, so we do not export it (YAGNI).
- ASCII punctuation everywhere, including strings and comments (binding, per spec). CI runs `ruff check .` — no f-strings without placeholders (F541), no assigned lambdas (E731).
- Do NOT push or open a PR without an explicit ask from the user.

## File structure

- Create: `viz/collect.py` — exporter; pure helpers (`sample_fields`, `rollout`, `in_split_window`, `verify_numbers`, `build_scene`) + `main()` CLI.
- Create: `tests/test_viz_collect.py` — fast tests for the pure helpers (no bundle load, no surrogate training).
- Create: `viz/data/scene.json` — committed artifact written by the collector.
- Modify: `viz/player/player.js:261-285` — key-based `drawPatch`, optional per-step pellets.
- Modify: `viz/player/capture/capture.js:53` — refuse full render of placeholder scene.
- Modify (maybe): `.gitignore` — ensure `viz/out/` is ignored.

---

### Task 1: Collector pure helpers with tests

**Files:**
- Create: `viz/collect.py`
- Test: `tests/test_viz_collect.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_viz_collect.py`:

```python
"""Tests for viz/collect.py pure helpers (no agent bundle, no surrogate training)."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("viz_collect", ROOT / "viz" / "collect.py")
vc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(vc)


def _beats(surv="0.73", untr="0.50", orac="0.99"):
    return {
        "numbers": {
            "oracle_l2": {"display": orac},
            "probe_chance": {"display": untr},
            "probe_survival": {"display": surv},
        },
        "beats": [
            {"id": "observer", "gauge": {"display": orac}},
            {"id": "nocare", "gauge": {"display": untr}},
            {"id": "survival", "gauge": {"display": surv}},
        ],
    }


def _cells(surv=0.726, untr=0.488):
    out = []
    for s in range(3):
        out.append({"agent": "survival", "drift": "0.45", "seed": s,
                    "resid_trace": surv, "target": 0.75})
        out.append({"agent": "untrained", "drift": "0.45", "seed": s,
                    "resid_trace": 0.5, "target": untr})
        out.append({"agent": "survival", "drift": "0.00", "seed": s,
                    "resid_trace": 0.5, "target": 0.5})
    return out


def test_verify_numbers_passes_on_published_values():
    got = vc.verify_numbers(_beats(), _cells(), "oracle AUROC 0.993 etc")
    assert got["survival_resid_trace_pooled"] == 0.726
    assert got["untrained_target_pooled"] == 0.488


def test_verify_numbers_fails_on_drifted_survival():
    with pytest.raises(SystemExit):
        vc.verify_numbers(_beats(), _cells(surv=0.61), "0.993")


def test_verify_numbers_fails_when_untrained_leaves_chance():
    with pytest.raises(SystemExit):
        vc.verify_numbers(_beats(), _cells(untr=0.60), "0.993")


def test_verify_numbers_fails_without_findings_oracle():
    with pytest.raises(SystemExit):
        vc.verify_numbers(_beats(), _cells(), "no oracle number here")


def test_sample_fields_normalized_grid():
    from itasorl.experiment_b2 import _seeds, make_world
    w = make_world(vc.P, 0.0, vc.RAY_STEPS)
    w.reset(_seeds(1234))
    height, wet = vc.sample_fields(w, 8)
    assert height.shape == (64,) and wet.shape == (64,)
    assert height.min() == 0.0 and height.max() == 1.0
    assert set(np.unique(wet)) <= {0.0, 1.0}


def test_in_split_window():
    inside = [[0.5, 0.5, 0.0, 1.0]] * 300
    assert vc.in_split_window(inside)
    outside = [list(p) for p in inside]
    outside[150][0] = 0.9  # wanders out of the split-panel strip mid-beat
    assert not vc.in_split_window(outside)


def test_build_scene_schema_and_coverage():
    r = {"pts": [[0.5, 0.5, 0.0, 0.8]] * 900, "pellets_t": [[[0.1, 0.2]]] * 900}
    scene = vc.build_scene({"source": "collect.py"}, 8, np.zeros(64), np.zeros(64), r, r)
    for key in ("meta", "grid_n", "height", "wet", "pellets", "trajs", "pellets_t", "step_ms"):
        assert key in scene
    json.dumps(scene)  # fully JSON-serializable
    assert len(scene["trajs"]["auth"]) * scene["step_ms"] >= vc.LAST_WORLD_MS


def test_build_scene_rejects_short_traj():
    r = {"pts": [[0.5, 0.5, 0.0, 0.8]] * 100, "pellets_t": [[[0.1, 0.2]]] * 100}
    with pytest.raises(AssertionError):
        vc.build_scene({"source": "collect.py"}, 8, np.zeros(64), np.zeros(64), r, r)
```

- [ ] **Step 2: Run tests to verify they fail**

Run from worktree root: `python -m pytest tests/test_viz_collect.py -q`
Expected: FAIL at import time (`viz/collect.py` does not exist).

- [ ] **Step 3: Write the helpers**

Create `viz/collect.py`:

```python
"""Exporter: real recorded runs -> viz/data/scene.json for the outreach film player.

Record-then-render (spec docs/superpowers/specs/2026-07-14-outreach-video-design.md):
rolls a saved survival agent in the authentic world and the L3 learned-surrogate
world on identical seeds, samples the terrain fields the player bakes, and fails
loudly if the film's caption numbers drift from the committed artifacts.
CPU only: fullruns/ is read-only and the GPU may be busy with a live run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

WT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT_ROOT))

from itasorl.world import WorldParams  # noqa: E402

# Frozen world params, identical to scripts/run_expB2.py:57.
P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
RAY_STEPS = 5           # run_expB2.py --ray_steps default; the eval standard
ENERGY_FULL = 1.0       # SURVIVAL_METAB E0; player energy bar expects [0, 1]
# Beat 2 (trick, 10s-25s at step_ms=100) shows a 477px center strip of the
# 960px world; x outside [0.25, 0.75] is clipped. 0.28/0.72 leaves a margin.
TRICK_I0, TRICK_I1 = 100, 250
SPLIT_LO, SPLIT_HI = 0.28, 0.72
LAST_WORLD_MS = 75_000  # the world is on screen through beat 5 (55s-75s)
STEP_MS = 100


def sample_fields(world, n):
    """Sample the reset world's analytic height/wetness on an n*n row-major grid
    (index j*n+i, x=i/(n-1), y=j/(n-1)) to mirror player.js placeholderScene."""
    height = np.empty(n * n)
    wet = np.empty(n * n)
    for j in range(n):
        y = j / (n - 1)
        for i in range(n):
            x = i / (n - 1)
            height[j * n + i] = world._H(x, y)
            wet[j * n + i] = 1.0 if world._wetness(x, y) > 0.5 else 0.0
    height = (height - height.min()) / (height.max() - height.min())
    return height, wet


def rollout(agent, norm, world, steps, device="cpu"):
    """Deterministic policy rollout; per-step [x, y, heading, energy] plus active
    pellet positions. Returns None if the creature dies (caller tries new seeds).
    Mirrors common_garden_rollout (itasorl/experiment_b2.py:869-887)."""
    import torch
    h = agent.initial_state(1, device)
    prev = torch.zeros(1, agent.act_dim, device=device)
    obs = world.observe().astype(np.float64)
    pts, pellets_t = [], []
    for _ in range(steps):
        obs_t = torch.as_tensor(norm(obs)[None], dtype=torch.float32, device=device)
        _, env_act, _, _, h = agent.act(obs_t, prev, h, deterministic=True)
        r = world.step(env_act[0].detach().cpu().numpy().astype(np.float32))
        obs = r.obs.astype(np.float64)
        prev = env_act
        pts.append([float(world.pos[0]), float(world.pos[1]), float(world.heading),
                    min(1.0, max(0.0, float(world.E) / ENERGY_FULL))])
        active = world.pellet_amt > 1e-9
        pellets_t.append([[round(float(x), 4), round(float(y), 4)]
                          for x, y in world.pellets[active]])
        if r.terminated:
            return None
    return {"pts": pts, "pellets_t": pellets_t}


def in_split_window(pts, i0=TRICK_I0, i1=TRICK_I1, lo=SPLIT_LO, hi=SPLIT_HI):
    """True if the creature stays visible in the split-beat center strip."""
    return all(lo <= p[0] <= hi for p in pts[i0:i1 + 1])


def _pooled(cells, agent, key):
    vals = [float(c[key]) for c in cells
            if c["agent"] == agent and abs(float(c["drift"]) - 0.45) < 1e-9]
    if not vals:
        raise SystemExit(f"number honesty: no drift=0.45 cells for agent '{agent}'")
    return sum(vals) / len(vals)


def verify_numbers(beats, cells, findings_text):
    """Binding spec rule: the film's three numbers must match the committed
    artifacts. Fails loudly; returns the verified values for scene meta."""
    disp = {b["id"]: b["gauge"]["display"] for b in beats["beats"] if "gauge" in b}
    nums = beats["numbers"]

    surv = _pooled(cells, "survival", "resid_trace")
    if f"{surv:.2f}" != disp["survival"] or disp["survival"] != nums["probe_survival"]["display"]:
        raise SystemExit(f"number honesty: survival resid_trace pools to {surv:.4f}, "
                         f"film shows {disp['survival']}")

    untr = _pooled(cells, "untrained", "target")
    if (abs(untr - 0.50) > 0.03 or disp["nocare"] != "0.50"
            or nums["probe_chance"]["display"] != "0.50"):
        raise SystemExit(f"number honesty: untrained target pools to {untr:.4f}, "
                         f"film shows {disp['nocare']} as chance")

    if ("0.993" not in findings_text or disp["observer"] != "0.99"
            or nums["oracle_l2"]["display"] != "0.99"):
        raise SystemExit("number honesty: 0.993 missing from FINDINGS or gauge is not 0.99")

    return {"survival_resid_trace_pooled": round(surv, 4),
            "untrained_target_pooled": round(untr, 4),
            "oracle_l2": 0.993}


def _round_traj(pts):
    return [[round(x, 4), round(y, 4), round(a, 3), round(e, 3)] for x, y, a, e in pts]


def build_scene(meta, grid_n, height, wet, ra, rs, step_ms=STEP_MS):
    scene = {
        "meta": meta,
        "grid_n": grid_n,
        "height": [round(float(h), 4) for h in height],
        "wet": [float(w) for w in wet],
        "pellets": ra["pellets_t"][0],
        "trajs": {"auth": _round_traj(ra["pts"]), "surr": _round_traj(rs["pts"])},
        "pellets_t": {"auth": ra["pellets_t"], "surr": rs["pellets_t"]},
        "step_ms": step_ms,
    }
    assert len(scene["trajs"]["auth"]) * step_ms >= LAST_WORLD_MS, \
        "trajectory too short: the world is on screen through beat 5"
    assert len(scene["trajs"]["surr"]) * step_ms >= LAST_WORLD_MS
    return scene
```

(`main()` comes in Task 2; the module must import cleanly without torch at top level — torch is imported inside `rollout` and `main`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_viz_collect.py -q`
Expected: 8 passed.

- [ ] **Step 5: Lint and commit**

```bash
ruff check viz/collect.py tests/test_viz_collect.py
git add viz/collect.py tests/test_viz_collect.py
git commit -m "feat(viz): collector pure helpers - field sampling, framing check, number honesty"
```

---

### Task 2: Collector CLI (surrogate refit, paired rollouts, scene export)

**Files:**
- Modify: `viz/collect.py` (append `main()`)

- [ ] **Step 1: Append `main()` to `viz/collect.py`**

```python
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents",
                    default=str((WT_ROOT / ".." / ".." / "fullruns" / "l3_h8_heldout"
                                 / "agents").resolve()),
                    help="READ-ONLY dir of saved bundles (main checkout)")
    ap.add_argument("--bundle", default="agent_d0.45_s0_survival.pt")
    ap.add_argument("--drift", type=float, default=0.45)
    ap.add_argument("--l3-hidden", type=int, default=8)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--grid", type=int, default=160)
    ap.add_argument("--seed-base", type=int, default=424_242)
    ap.add_argument("--max-candidates", type=int, default=30)
    ap.add_argument("--out", default=str(WT_ROOT / "viz" / "data" / "scene.json"))
    a = ap.parse_args(argv)

    import torch
    torch.set_num_threads(2)  # a GPU training run may be live on this machine
    import itasorl.experiment_b2 as b2
    from itasorl.experiment_b2 import _seeds, load_agent_bundle, make_world

    beats = json.loads((WT_ROOT / "viz" / "player" / "beats.json").read_text(encoding="utf-8"))
    audit = WT_ROOT / "artifacts" / "expB2" / "behavior_audit_l3_h8_traces.json"
    cells = json.loads(audit.read_text(encoding="utf-8"))["cells"]
    findings = (WT_ROOT / "docs" / "FINDINGS.md").read_text(encoding="utf-8")
    numbers = verify_numbers(beats, cells, findings)
    print(f"numbers verified against artifacts: {numbers}", flush=True)

    b2.DRIFT_MODE = "l3"
    print(f"training L3 surrogate on cpu (hidden={a.l3_hidden}, frozen recipe seed=0)...",
          flush=True)
    b2.setup_l3_surrogate(hidden=a.l3_hidden, device="cpu", seed=0, params=P)

    agent, norm = load_agent_bundle(str(Path(a.agents) / a.bundle), device="cpu")

    ra = rs = None
    base = a.seed_base
    for i in range(a.max_candidates):
        base = a.seed_base + 1000 * i
        seeds = _seeds(base)
        wa = make_world(P, 0.0, RAY_STEPS)
        wa.reset(seeds)
        ws = make_world(P, a.drift, RAY_STEPS)
        ws.reset(seeds)
        ra = rollout(agent, norm, wa, a.steps)
        rs = rollout(agent, norm, ws, a.steps)
        ok = (ra is not None and rs is not None
              and in_split_window(ra["pts"]) and in_split_window(rs["pts"]))
        print(f"candidate seed base {base}: {'selected' if ok else 'rejected'}", flush=True)
        if ok:
            break
    else:
        raise SystemExit("no candidate stayed alive 900 steps inside the split-beat "
                         "frame; raise --max-candidates")

    height, wet = sample_fields(wa, a.grid)
    e5 = [p[3] for p in ra["pts"][550:750]]  # beat 5 window on the auth trajectory
    meta = {"source": "collect.py",
            "bundle": a.bundle,
            "agents_dir": str(Path(a.agents).resolve()),
            "drift": a.drift, "l3_hidden": a.l3_hidden,
            "surrogate_refit_device": "cpu",
            "seed_base": base, "steps": a.steps,
            "energy_range_beat5": [round(min(e5), 3), round(max(e5), 3)],
            "numbers_verified": numbers}
    scene = build_scene(meta, a.grid, height, wet, ra, rs)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scene, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB), seed_base={base}")


if __name__ == "__main__":
    main()
```

Notes:
- The surrogate is refit with the frozen recipe (hidden=8, seed=0, same params) but on CPU; the run trained it on CUDA, so weights are not bit-identical. That is why `meta.surrogate_refit_device` is recorded — the film is still a real rollout in a real L3 surrogate, flagged honestly.
- `ENERGY_FULL = 1.0` clamps the exported energy; `meta.energy_range_beat5` records the raw span seen during beat 5. If the visual pass (Task 5) shows the bar pinned flat, the constant is the one knob to revisit — ask the user before redefining it.

- [ ] **Step 2: Sanity checks**

```bash
ruff check viz/collect.py
python -m pytest tests/test_viz_collect.py -q
python viz/collect.py --help
```
Expected: lint clean, 8 passed, help text prints (module imports without running anything heavy).

- [ ] **Step 3: Commit**

```bash
git add viz/collect.py
git commit -m "feat(viz): real-data collector - survival agent rollouts in auth and L3 worlds"
```

---

### Task 3: Generate and commit scene.json

- [ ] **Step 1: Preflight**

```bash
ls artifacts/expB2/behavior_audit_l3_h8_traces.json docs/FINDINGS.md
ls ../../fullruns/l3_h8_heldout/agents/agent_d0.45_s0_survival.pt
```
Expected: all three exist. (If the audit JSON is missing from the worktree, stop and report — do not copy files from the main checkout.)

- [ ] **Step 2: Run the collector**

Run from worktree root: `python viz/collect.py`
Expected output lines, in order: `numbers verified against artifacts: {...}`, `training L3 surrogate on cpu...` (this step takes a few minutes), one or more `candidate seed base ...` lines ending in `selected`, then `wrote ...viz/data/scene.json (X.X MB), seed_base=NNNN`.
If every candidate is `rejected`, rerun with `--max-candidates 60`. If the failure is deaths (not framing), report to the user rather than switching bundles silently.

- [ ] **Step 3: Inspect the artifact**

```bash
python -c "
import json
s = json.load(open('viz/data/scene.json'))
m = s['meta']
print(m['source'], m['seed_base'], m['energy_range_beat5'])
print('traj lens', len(s['trajs']['auth']), len(s['trajs']['surr']))
print('grid', s['grid_n'], 'height', len(s['height']), 'wet', len(s['wet']))
"
```
Expected: `collect.py <base> [lo, hi]`, traj lens 900 900, grid 160 with 25600-length fields. If `energy_range_beat5` is a degenerate span (for example `[1.0, 1.0]`), flag it to the user at the Task 5 visual pass.

- [ ] **Step 4: Commit**

```bash
git add viz/data/scene.json
git commit -m "feat(viz): committed real-run scene - h8 survival agent d0.45 s0, paired auth/L3 rollouts"
```

---

### Task 4: Player consumes recorded data; capture refuses placeholder

**Files:**
- Modify: `viz/player/player.js:261-285`
- Modify: `viz/player/capture/capture.js:53`
- Modify (conditional): `.gitignore`

- [ ] **Step 1: Rework `drawPatch` to key-based lookup with per-step pellets**

In `viz/player/player.js`, replace the `drawPatch` closure and its call sites (lines 261-285) with:

```js
    const drawPatch = (vx, vw, key) => {
      const traj = this.scene.trajs[key];
      const idx = this.trajIndex(t, traj);
      const sw = 960 / zoom, sx = (960 - sw) / 2, sy = (960 - sw) / 2;
      ctx.save();
      ctx.beginPath(); ctx.rect(vx, 0, vw, 960); ctx.clip();
      ctx.drawImage(this.terrain, sx, sy, sw, sw, vx - (960 - vw) / 2, 0, 960, 960);
      const view = {
        x: (u) => vx - (960 - vw) / 2 + ((u * 960 - sx) / sw) * 960,
        y: (v) => ((v * 960 - sy) / sw) * 960
      };
      const pt = this.scene.pellets_t ? this.scene.pellets_t[key] : null;
      const pellets = pt ? pt[Math.min(pt.length - 1, idx)] : this.scene.pellets;
      drawPellets(ctx, pellets, view);
      drawTrail(ctx, traj, idx, view);
      const p = traj[idx];
      drawCreature(ctx, this.creature, beat.world.stage || 0, view.x(p[0]), view.y(p[1]), t);
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
```

(The `cloud` branch and everything after it are unchanged. The placeholder scene has no `pellets_t`, so the fallback path still works.)

- [ ] **Step 2: Guard the capture against the silent placeholder fallback**

The player fetches `../data/scene.json` and silently falls back to the placeholder (player.js:419-423) — for example when the HTTP server root is wrong. In `viz/player/capture/capture.js`, insert after the `console.log(...)` on line 53:

```js
  if (MODE === "full" && src !== "collect.py" && process.env.CAP_ALLOW_PLACEHOLDER !== "1") {
    throw new Error("scene source is '" + src + "', not collect.py - refusing full render "
      + "(serve viz/ as the http root; CAP_ALLOW_PLACEHOLDER=1 overrides)");
  }
```

- [ ] **Step 3: Ensure `viz/out/` is gitignored**

```bash
git check-ignore -q viz/out/x.mp4 && echo ignored || echo NOT-ignored
```
If `NOT-ignored`: append a `viz/out/` line to `.gitignore` and include it in the commit below.

- [ ] **Step 4: Commit**

```bash
git add viz/player/player.js viz/player/capture/capture.js
git commit -m "feat(viz): player consumes recorded scene; capture refuses placeholder full render"
```

---

### Task 5: Verification and final render

The capture URL must serve `viz/` as the root so `../data/scene.json` resolves (a server rooted at `viz/player` clamps `..` and reproduces the placeholder film).

- [ ] **Step 1: Start the file server (background)**

From worktree root: `python -m http.server 8765 --bind 127.0.0.1 --directory viz`

- [ ] **Step 2: Locate the capture Chromium**

```bash
ls "$LOCALAPPDATA"/ms-playwright/chromium-*/chrome-win*/chrome.exe 2>/dev/null | head -2
```
Export the newest as `CHROME_EXE`. If none: `cd viz/player/capture && npx playwright install chromium`.

- [ ] **Step 3: Proof frames + visual pass**

```bash
cd viz/player/capture
CAP_MODE=proof CAP_URL=http://127.0.0.1:8765/player/index.html CHROME_EXE="<path>" node capture.js
```
Expected log: `scene=collect.py` (anything else means the fetch fell back — fix the server root before continuing), then 6 proof PNGs. View all six and check: (5000) creature on land, trail behind it; (17000) split view, creature visible in BOTH panels inside the strips; (32000) point clouds over faded terrain; (47000) gauge at 0.50; (65000) energy bar visible and not pinned at either end, pellet count visibly changing as it eats; (80000) end card. Report the frames to the user before the full render — the energy-bar readability question (ENERGY_FULL) is theirs to call if it looks flat.

- [ ] **Step 4: Determinism byte-compare**

```bash
mv proof proofA
CAP_MODE=proof CAP_URL=http://127.0.0.1:8765/player/index.html CHROME_EXE="<path>" node capture.js
cmp proofA/frame_032000.png proof/frame_032000.png && cmp proofA/frame_065000.png proof/frame_065000.png && echo DETERMINISTIC
rm -rf proofA
```
Expected: `DETERMINISTIC`.

- [ ] **Step 5: Full render**

```bash
mkdir -p ../../out
CAP_MODE=full CAP_URL=http://127.0.0.1:8765/player/index.html CHROME_EXE="<path>" CAP_OUT=../../out/itasorl_v1.mp4 node capture.js
```
Expected: progress lines, then `FULL_DONE -> ../../out/itasorl_v1.mp4` (roughly 10-20 minutes at capture speed; run in background and monitor).

- [ ] **Step 6: Output checks**

```bash
ffprobe -v error -show_entries format=duration,size -of default=nw=1 ../../out/itasorl_v1.mp4
```
Expected: duration within 84.9-85.1 s; size well under 60 MB.

- [ ] **Step 7: Repo green + wrap up**

From worktree root:

```bash
ruff check .
python -m pytest -q
```
Expected: both clean (ruff has failed CI before on F541 — check it, do not skip). Stop the file server. Report to the user with the MP4 path and proof frames; do NOT push or open a PR without an explicit ask.

---

## Self-review (done at write time)

- Spec coverage: collector + committed data (T1-T3), real-data player (T4), number honesty (T1/T3), Playwright capture, determinism, 1080x1350/duration/size checks (T5), ruff/pytest green (T5). Ambient field intentionally not exported (player never consumes it; noted above).
- Types consistent: `rollout` returns `{"pts", "pellets_t"}` consumed by `build_scene` and `main`; `scene.pellets_t[key]` matches the player lookup; `verify_numbers(beats, cells, findings_text)` used identically in tests and `main`.
- No placeholders: every code step carries the actual code; commands carry expected output.
