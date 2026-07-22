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
# Beat 2 (trick, 10s-25s) shows a 477px center strip of the 960px world; x
# outside [0.25, 0.75] is clipped. 0.28/0.72 leaves a margin. TRICK_I0/I1 are
# the step indices for the 100 ms/step case (the unit-test default); main()
# recomputes them from the chosen step_ms.
TRICK_T0, TRICK_T1 = 10_000, 25_000  # split beat: auth AND surr both on screen
TRICK_I0, TRICK_I1 = 100, 250
SPLIT_LO, SPLIT_HI = 0.28, 0.72
LAST_WORLD_MS = 75_000  # the authentic world is on screen through beat 5 (75s)
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
    """Deterministic policy rollout up to `steps` or death, whichever comes first.
    Records per-step [x, y, heading, energy] plus active pellet positions. Returns
    {"pts", "pellets_t", "n", "died"}; n == len(pts) is the survival length (< steps
    if the creature starved). Mirrors common_garden_rollout (experiment_b2.py:869)."""
    import torch
    h = agent.initial_state(1, device)
    prev = torch.zeros(1, agent.act_dim, device=device)
    obs = world.observe().astype(np.float64)
    pts, pellets_t = [], []
    died = False
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
            died = True
            break
    return {"pts": pts, "pellets_t": pellets_t, "n": len(pts), "died": died}


def in_split_window(pts, i0=TRICK_I0, i1=TRICK_I1, lo=SPLIT_LO, hi=SPLIT_HI):
    """True if the creature stays visible in the split-beat center strip."""
    return all(lo <= p[0] <= hi for p in pts[i0:i1 + 1])


def _pooled(cells, agent, key):
    vals = [float(c[key]) for c in cells
            if c["agent"] == agent and abs(float(c["drift"]) - 0.45) < 1e-9]
    if not vals:
        raise SystemExit("number honesty: no drift=0.45 cells for agent '" + agent + "'")
    return sum(vals) / len(vals)


def verify_numbers(beats, cells, findings_text):
    """Binding spec rule: the film's three numbers must match the committed
    artifacts. Fails loudly; returns the verified values for scene meta."""
    disp = {b["id"]: b["gauge"]["display"] for b in beats["beats"] if "gauge" in b}
    nums = beats["numbers"]

    # The film shows whole percents (the lay reading of AUROC on a balanced
    # one-real-one-fake comparison); each must still match the artifacts.
    surv = _pooled(cells, "survival", "resid_trace")
    if (f"{surv * 100:.0f}%" != disp["survival"]
            or disp["survival"] != nums["probe_survival"]["display"]):
        raise SystemExit("number honesty: survival resid_trace pools to "
                         + f"{surv:.4f}, film shows " + disp["survival"])

    untr = _pooled(cells, "untrained", "target")
    if (abs(untr - 0.50) > 0.03 or disp["nocare"] != "50%"
            or nums["probe_chance"]["display"] != "50%"):
        raise SystemExit("number honesty: untrained target pools to "
                         + f"{untr:.4f}, film shows " + disp["nocare"] + " as chance")

    if ("0.993" not in findings_text or disp["observer"] != "99%"
            or nums["oracle_l2"]["display"] != "99%"):
        raise SystemExit("number honesty: 0.993 missing from FINDINGS or gauge is not 99%")

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
        "authentic trajectory too short: it is on screen through beat 5 (75 s)"
    assert len(scene["trajs"]["surr"]) * step_ms >= TRICK_T1, \
        "surrogate trajectory too short: it must outlast the split beat (25 s)"
    return scene


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--agents",
                    default=str((WT_ROOT / ".." / ".." / "fullruns" / "l3_h8_heldout"
                                 / "agents").resolve()),
                    help="READ-ONLY dir of saved bundles (main checkout)")
    ap.add_argument("--bundle", default="agent_d0.45_s0_survival.pt")
    ap.add_argument("--drift", type=float, default=0.45)
    ap.add_argument("--l3-hidden", type=int, default=8)
    ap.add_argument("--steps", type=int, default=900,
                    help="rollout cap; the agent usually starves well before this")
    ap.add_argument("--grid", type=int, default=160)
    ap.add_argument("--seed-base", type=int, default=424_242)
    ap.add_argument("--max-candidates", type=int, default=240,
                    help="seeds ranked by authentic survival length")
    ap.add_argument("--top-k", type=int, default=32,
                    help="longest survivors that get the surrogate + framing check")
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
    print("numbers verified against artifacts: " + str(numbers), flush=True)

    b2.DRIFT_MODE = "l3"
    print("training L3 surrogate on cpu (hidden="
          + f"{a.l3_hidden}, frozen recipe seed=0)...", flush=True)
    b2.setup_l3_surrogate(hidden=a.l3_hidden, device="cpu", seed=0, params=P)

    agent, norm = load_agent_bundle(str(Path(a.agents) / a.bundle), device="cpu")

    # Option 1 (slow the world clock): the saved short-horizon survival agent
    # (trained at max_steps<=80) starves long before 900 steps, so no single seed
    # survives the film at 100 ms/step. Rank seeds by AUTHENTIC survival (auth is on
    # screen through beat 5 / 75 s), then stretch the longest real episode across
    # those 75 s by picking step_ms = ceil(75000 / n_auth). The SURROGATE creature
    # only appears in the split beat (10-25 s), so it just has to outlast that.
    ranked = []
    for i in range(a.max_candidates):
        base = a.seed_base + 1000 * i
        wa = make_world(P, 0.0, RAY_STEPS)
        wa.reset(_seeds(base))
        ra = rollout(agent, norm, wa, a.steps)
        ranked.append((ra["n"], base))
    ranked.sort(reverse=True)
    print("authentic survival: best "
          + f"{ranked[0][0]} median {ranked[len(ranked) // 2][0]} "
          + f"worst {ranked[-1][0]} steps (n={len(ranked)})", flush=True)

    chosen = None
    for n_auth, base in ranked[:a.top_k]:
        step_ms = (LAST_WORLD_MS + n_auth - 1) // n_auth   # ceil
        i0, i1 = round(TRICK_T0 / step_ms), round(TRICK_T1 / step_ms)
        seeds = _seeds(base)
        wa = make_world(P, 0.0, RAY_STEPS)
        wa.reset(seeds)
        ws = make_world(P, a.drift, RAY_STEPS)
        ws.reset(seeds)
        ra = rollout(agent, norm, wa, a.steps)
        rs = rollout(agent, norm, ws, a.steps)
        surr_ok = rs["n"] > i1  # surrogate creature alive through the split beat
        frame_ok = (surr_ok and in_split_window(ra["pts"], i0, i1)
                    and in_split_window(rs["pts"], i0, i1))
        print("seed " + f"{base}: n_auth={n_auth} n_surr={rs['n']} step_ms={step_ms} "
              + f"surr_ok={surr_ok} frame_ok={frame_ok}", flush=True)
        if frame_ok:
            chosen = (base, step_ms, ra, rs, wa, n_auth, rs["n"], i0, i1)
            break
    if chosen is None:
        raise SystemExit("no ranked seed kept both creatures framed through the split "
                         "beat; raise --max-candidates/--top-k or loosen SPLIT_LO/HI")
    base, step_ms, ra, rs, wa, n_auth, n_surr, i0, i1 = chosen

    height, wet = sample_fields(wa, a.grid)
    b5_lo, b5_hi = round(55_000 / step_ms), round(75_000 / step_ms)
    e5 = [p[3] for p in ra["pts"][b5_lo:b5_hi]]  # beat 5 (survival) energy span
    meta = {"source": "collect.py",
            "bundle": a.bundle,
            "agents_dir": Path(a.agents).as_posix(),
            "drift": a.drift, "l3_hidden": a.l3_hidden,
            "surrogate_refit_device": "cpu",
            "seed_base": base, "steps": a.steps,
            "step_ms": step_ms, "n_auth": n_auth, "n_surr": n_surr,
            "clock_note": "step_ms stretches the real episode over the 75 s the world "
                          "is on screen; agent is short-horizon (starves ~150 steps)",
            "split_frame_idx": [i0, i1],
            "energy_range_beat5": [round(min(e5), 3), round(max(e5), 3)],
            "numbers_verified": numbers}
    scene = build_scene(meta, a.grid, height, wet, ra, rs, step_ms=step_ms)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(scene, separators=(",", ":")), encoding="utf-8")
    print("wrote " + str(out) + f" ({out.stat().st_size / 1e6:.1f} MB), "
          + f"seed_base={base} step_ms={step_ms}")


if __name__ == "__main__":
    main()
