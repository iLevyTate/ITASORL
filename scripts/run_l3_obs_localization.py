"""L3 observation-channel localization (A2 ablation), READOUT-ONLY.

Loads saved L3 agents and re-collects drift-0.45 pools with selected observation
channels masked (zeroed before the agent's running norm). A fresh pooled_readout on
the masked pools tells us which observation channels the world-identity signal
rides on. No training.

Usage:
    python scripts/run_l3_obs_localization.py \\
        --agents-dir fullruns/l3_h8_heldout/agents \\
        --out-dir fullruns/l3_h8_obs_localization \\
        --masks vision intero all --device cuda
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import re

import numpy as np

import itasorl.experiment_b2 as b2
from itasorl.experiment_b2 import default_device, load_agent_bundle, pooled_readout, setup_l3_surrogate
from itasorl.patch_of_earth import first_config_obs_spec
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
AGENT_RE = re.compile(r"agent_d(\d+\.\d+)_s(\d+)_(untrained|predictor|survival)\.pt$")


def parse_agent_filename(name: str) -> tuple[float, int, str]:
    m = AGENT_RE.search(name)
    if not m:
        raise ValueError(f"unrecognized agent filename: {name}")
    return float(m.group(1)), int(m.group(2)), m.group(3)


def build_mask(mask_type: str, obs_dim: int) -> np.ndarray:
    """Return a float mask of length obs_dim. 1 = channel present, 0 = masked."""
    spec = first_config_obs_spec()
    if spec.size != obs_dim:
        raise ValueError(f"obs_dim mismatch: spec={spec.size} vs agent={obs_dim}")
    m = np.ones(obs_dim, dtype=np.float64)
    if mask_type == "none":
        return m
    if mask_type == "all":
        return np.zeros(obs_dim, dtype=np.float64)
    sl = spec.slices()
    if mask_type == "vision":
        m[sl["vision"]] = 0.0
    elif mask_type == "intero":
        m[sl["intero"]] = 0.0
    elif mask_type == "smell":
        m[sl["smell"]] = 0.0
    else:
        raise ValueError(f"unknown mask type: {mask_type}")
    return m


def cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--masks", nargs="+", default=["none", "vision", "intero", "all"],
                    choices=("none", "vision", "intero", "smell", "all"))
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--n-eps", type=int, default=110)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--quick", action="store_true",
                    help="smoke: seed 0 only, tiny pools, no full aggregate")
    return ap.parse_args()


def main():
    a = cfg()
    if a.hidden not in (7, 8):
        raise SystemExit("--hidden must be 7 or 8")
    dev = default_device() if a.device == "auto" else a.device
    if a.device == "cuda" and dev != "cuda":
        raise SystemExit("--device cuda requested but CUDA unavailable")
    os.makedirs(a.out_dir, exist_ok=True)
    b2.DRIFT_MODE = "l3"
    setup_l3_surrogate(hidden=a.hidden, seed=0, params=P, device=dev)
    n_eps, steps = (12, 8) if a.quick else (a.n_eps, a.steps)

    cells = sorted(f for f in os.listdir(a.agents_dir) if AGENT_RE.search(f))
    cells = [c for c in cells if parse_agent_filename(c)[0] == 0.45]
    if a.quick:
        cells = [c for c in cells if "_s0_" in c]
    if not cells:
        raise SystemExit(f"no drift-0.45 agents in {a.agents_dir}")

    # Load one agent to infer obs_dim
    _, _, arm0 = parse_agent_filename(cells[0])
    agent0, norm0 = load_agent_bundle(os.path.join(a.agents_dir, cells[0]), dev)
    obs_dim = agent0.obs_dim

    results = []
    for mask_type in a.masks:
        mask = build_mask(mask_type, obs_dim)
        masked = int(np.sum(mask == 0.0))
        print(f"mask={mask_type}: {masked}/{obs_dim} dimensions zeroed")
        for name in cells:
            drift, seed, arm = parse_agent_filename(name)
            agent, norm = load_agent_bundle(os.path.join(a.agents_dir, name), dev)
            out = pooled_readout(agent, norm, P, drift, n_eps=n_eps, steps=steps,
                                 device=dev, seed=seed, obs_mask=mask)
            row = {"mask": mask_type, "drift": drift, "seed": seed, "arm": arm,
                   "target": float(out["target"]), "target_lo": float(out["target_lo"]),
                   "target_hi": float(out["target_hi"]),
                   "shuffled": float(out["shuffled"]),
                   "speed": float(out["speed"]),
                   "anchor_energy": float(out["anchor_energy"]),
                   "anchor_food": float(out["anchor_food"]),
                   "pool_reward_leak": float(out["pool_reward_leak"]),
                   "pool_leak_clean": bool(out["pool_leak_clean"]),
                   "deaths_auth": int(out["deaths_auth"]),
                   "deaths_surr": int(out["deaths_surr"])}
            results.append(row)
            print(f"  {mask_type} {name}: target={out['target']:.3f} "
                  f"speed={out['speed']:.3f} energy={out['anchor_energy']:.3f} "
                  f"food={out['anchor_food']:.3f}")

    # Aggregate per mask / arm
    agg = {"masks": a.masks, "hidden": a.hidden, "obs_dim": obs_dim}
    for mask_type in a.masks:
        for arm in ("untrained", "predictor", "survival"):
            vals = [r["target"] for r in results if r["mask"] == mask_type and r["arm"] == arm]
            if vals:
                v = np.asarray(vals, float)
                agg[f"{mask_type}_{arm}_mean"] = round(float(v.mean()), 4)
                agg[f"{mask_type}_{arm}_per_seed"] = [round(float(x), 4) for x in v]
                agg[f"{mask_type}_{arm}_n_ge_065"] = int((v >= 0.65).sum())

    with open(os.path.join(a.out_dir, "cells.json"), "w") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(a.out_dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=1)
    print("wrote", os.path.join(a.out_dir, "aggregate.json"))


if __name__ == "__main__":
    main()
