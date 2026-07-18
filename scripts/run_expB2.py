"""
ITASORL - Experiment B-v2 full run: does SURVIVAL pressure induce incidental
detection where prediction alone did not?

Three agents share the identical recurrent trunk and the identical readout; only
the objective differs:
    untrained   - random init           (mechanical floor: drift perturbs inputs)
    predictor   - next-step prediction  (Experiment B's objective, scripted policy)
    survival    - actor-critic          (acts to stay alive under drifting dynamics)

Two readouts per agent:
    PRIMARY  pooled (Exp B frame) - persistent world-identity direction across
             independent episodes; ~0.5 = no incidental encoding. The headline.
    SECONDARY matched-pair        - detectability of the artifact in the agent state.

Gates (pre-registered, see docs/PREREGISTRATION.md) before the survival target counts:
    engagement   - trained return > random AND scripted (else "uninformative")
    L0 control   - drift=0 pooled target equivalent to 0.5 (TOST)
    positive ctrl- speed probe high
    leakage      - reward/length/lifetime ~0.5

Usage:  python scripts/run_expB2.py            # full run (CUDA if available)
        python scripts/run_expB2.py --quick    # fast sanity pass
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from itasorl.experiment_b2 import (  # noqa: E402
    default_device,
    engagement_metric,
    pooled_readout,
    readout,
    survival_return,
    train_actor_critic,
    train_predictor_only,
    untrained_agent,
)
import itasorl.experiment_b2 as b2  # noqa: E402
from itasorl.stats import equivalence_test, mean_ci, rope_test  # noqa: E402
from itasorl.world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
AG = ("untrained", "predictor", "survival")


def cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast sanity pass (tiny scale)")
    ap.add_argument("--drifts", type=float, nargs="+", default=[0.0, 0.45])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--updates", type=int, default=300)
    ap.add_argument("--n_eps", type=int, default=16)
    ap.add_argument("--max_steps", type=int, default=80)
    ap.add_argument("--hidden", type=int, default=96)
    ap.add_argument("--ray_steps", type=int, default=5)
    ap.add_argument("--shaping_coef", type=float, default=1.0)
    ap.add_argument("--pool_n", type=int, default=110)
    ap.add_argument("--pool_steps", type=int, default=24)
    ap.add_argument("--mp_pairs", type=int, default=60)
    ap.add_argument("--mp_prefix", type=int, default=20)
    ap.add_argument("--mp_branch", type=int, default=24)
    ap.add_argument("--out-dir", type=str, default=".",
                    help="Directory for expB2_results.json and expB2_survival.png")
    ap.add_argument("--dump-states", type=str, default=None,
                    help="Directory to persist raw recurrent states (.npz per agent/cell) so "
                         "probes can be recomputed offline with scripts/reanalyze_expB2_states.py")
    ap.add_argument("--heldout-evals", action="store_true",
                    help="L3 only: add the unseen-fingerprint transfer + common-garden "
                         "eval channels per cell (spec 2026-07-14)")
    ap.add_argument("--heldout-hidden", type=int, default=7,
                    help="G_motion capacity of the HELD-OUT fingerprint (frozen: 7)")
    ap.add_argument("--cg-prefix", type=int, default=20, help="common-garden prefix steps")
    ap.add_argument("--cg-steps", type=int, default=24, help="common-garden tail steps")
    ap.add_argument("--save-agents", action="store_true",
                    help="persist each trained arm to <out-dir>/agents/ (a few MB each)")
    # Positive-control CEILING (NOT readout-not-reward): supervise the survival trunk's h_t
    # onto the drag-drift so we can measure whether it CAN linearly encode world identity.
    ap.add_argument("--sysid-aux", action="store_true",
                    help="Add a system-ID auxiliary head to the survival agent (CEILING control; "
                         "breaks readout-not-reward - run and report separately from the headline).")
    ap.add_argument("--sysid-coef", type=float, default=1.0, help="weight of the sysid-aux loss")
    # Stage-2 objective-pressure overrides (scarcity). Default None keeps the frozen
    # SURVIVAL_METAB/SURVIVAL_FOOD; setting these sweeps how hard survival bites.
    ap.add_argument("--basal_e", type=float, default=None, help="override survival basal energy burn")
    ap.add_argument("--n_pellets", type=int, default=None, help="override pellet count (scarcity)")
    ap.add_argument("--reach", type=float, default=None, help="override eat reach radius")
    # B-v3 coupling: "ar1" = the pre-registered volatility surrogate; "regime" = a per-episode
    # CONSTANT drag offset (identifiable + policy-relevant), the "make it work as intended" arm.
    ap.add_argument("--drift-mode", choices=("ar1", "regime", "l3"), default="ar1",
                    help="surrogate coupling mode: ar1 (B-v2), regime (B-v3), or l3 "
                         "(learned-dynamics surrogate; default ar1)")
    ap.add_argument("--l3-hidden", type=int, default=8,
                    help="G_motion capacity for --drift-mode l3 (frozen gate-0 value 8; "
                         "see docs/PREREGISTRATION_L3.md sec.12)")
    # Speedup: the run is CPU-bound (serial physics, tiny nets), and (drift,seed) cells are
    # independent. --workers N runs N cells at once across CPU cores. Set N ~ vCPU count.
    ap.add_argument("--workers", type=int, default=1, help="parallel worker processes over cells")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                    help="auto = cuda when --workers 1, else cpu (avoids GPU contention; CPU-bound anyway)")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run: load matching cell "
                         "checkpoints from <out-dir>/cells and run only the "
                         "missing (drift, seed) cells")
    a = ap.parse_args()
    if a.quick:
        a.drifts, a.seeds, a.updates, a.n_eps, a.max_steps = [0.0, 0.45], [0, 1], 60, 8, 40
        a.hidden, a.ray_steps, a.pool_n, a.pool_steps = 64, 4, 40, 16
        a.mp_pairs, a.mp_prefix, a.mp_branch = 25, 12, 16
    if a.heldout_evals and a.drift_mode != "l3":
        raise SystemExit("--heldout-evals requires --drift-mode l3")
    if a.quick and a.heldout_evals:
        a.cg_prefix, a.cg_steps = 8, 12
    return a


def config_fingerprint(base: dict) -> str:
    """Hash of the science-relevant config. Cells from different configs never mix;
    dump_states is a path, not science, so it is excluded."""
    fp = {k: v for k, v in base.items() if k not in ("dump_states", "save_agents", "out_dir")}
    payload = json.dumps(fp, sort_keys=True, default=float)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def decide_h_b2(surv, pred, untr, bar: float = 0.65, sesoi: float = 0.05):
    """Pre-registered primary verdict with a survivorship-precondition guard.

    A non-finite pooled target means that seed's pool had too few survivors
    (`too_few_survivors` in pooled_readout) - a broken precondition, not a
    negative result - so any NaN forces the UNINFORMATIVE zone instead of
    silently reading as "NOT met".

    Returns (h_b2, zone, n_bad, finite survival values).
    """
    arrs = {"survival": np.asarray(surv, float),
            "predictor": np.asarray(pred, float),
            "untrained": np.asarray(untr, float)}
    n_bad = sum(int(np.count_nonzero(~np.isfinite(x))) for x in arrs.values())
    fin = {k: x[np.isfinite(x)] for k, x in arrs.items()}
    if n_bad or any(x.size == 0 for x in fin.values()):
        zone = (f"UNINFORMATIVE -> survivorship precondition failed in {n_bad} "
                "decision cell(s) (too_few_survivors); adjudicate per the "
                "pre-registered decision matrix before reading this as a result")
        return False, zone, n_bad, fin["survival"]
    s = fin["survival"].mean()
    beats = (s >= fin["predictor"].mean() + sesoi
             and s >= fin["untrained"].mean() + sesoi)
    # PREREGISTRATION_L3 sec. 10: the "clears / misses 0.65" adjudication follows
    # the t-based 90% CI on the per-seed values, not the bare mean (added
    # 2026-07-18 - the audit found the pipeline verdict used the mean alone,
    # weaker than the frozen rule; borderline runs now land inconclusive).
    sv = fin["survival"]
    if sv.size >= 2 and float(sv.std(ddof=1)) > 0.0:
        from scipy.stats import t as student_t
        se = float(sv.std(ddof=1)) / math.sqrt(sv.size)
        q = float(student_t.ppf(0.95, sv.size - 1))
        t_lo, t_hi = s - q * se, s + q * se
    else:  # single seed or zero variance: interval degenerates to the mean
        t_lo = t_hi = s
    ci_note = f" [t90 CI ({t_lo:.3f}, {t_hi:.3f}) vs bar {bar:.2f}]"
    if s >= bar and beats:
        if t_lo >= bar:
            return True, "MET  -> encoding induced (conditional on gates)" + ci_note, 0, sv
        zone = ("NOT met  -> mean clears the 0.65 bar but the t-based 90% CI "
                "straddles it; inconclusive under the pre-registered interval "
                "adjudication (PREREGISTRATION_L3 sec. 10)" + ci_note)
        return False, zone, 0, sv
    if beats:
        zone = ("NOT met  -> intermediate zone: beats both baselines by >= 0.05 but misses "
                "the 0.65 bar; adjudicate with the pre-registered n=10 power extension"
                + ci_note)
    else:
        zone = ("NOT met  -> strengthened negative (state readable, identity not encoded)"
                + ci_note)
    return False, zone, 0, sv


def cell_file(cells_dir, drift: float, seed: int) -> Path:
    return Path(cells_dir) / f"cell_d{drift:.2f}_s{seed}.json"


def git_commit_short() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, check=True,
                             cwd=Path(__file__).resolve().parent)
        return out.stdout.strip()
    except Exception:
        return "unknown"


def write_cell_file(cells_dir, fingerprint: str, commit: str, cell: dict) -> Path:
    """Atomic write: a killed process never leaves a half-written checkpoint."""
    cells_dir = Path(cells_dir)
    cells_dir.mkdir(parents=True, exist_ok=True)
    path = cell_file(cells_dir, cell["drift"], cell["seed"])
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"fingerprint": fingerprint, "git_commit": commit,
                   "cell": cell}, f, indent=2, default=float)
    os.replace(tmp, path)
    return path


def load_cell_files(cells_dir, fingerprint: str) -> dict:
    """Load checkpointed cells keyed by (drift, seed). Hard error on corrupt
    files or fingerprint mismatch; warning only on git commit drift."""
    done: dict[tuple[float, int], dict] = {}
    cells_dir = Path(cells_dir)
    if not cells_dir.is_dir():
        return done
    commit = git_commit_short()
    for path in sorted(cells_dir.glob("cell_d*_s*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                payload = json.load(f)
            fp, cell = payload["fingerprint"], payload["cell"]
        except Exception as exc:
            raise SystemExit(
                f"Corrupt checkpoint {path}: {exc}. "
                "Delete this one file and rerun with --resume.")
        if fp != fingerprint:
            raise SystemExit(
                f"Checkpoint {path} has fingerprint {fp}, current config is "
                f"{fingerprint}. It belongs to a different experiment config: "
                "use a fresh --out-dir, or delete the stale cells/ directory.")
        if payload.get("git_commit", "unknown") != commit:
            print(f"  WARNING: {path.name} was produced at commit "
                  f"{payload.get('git_commit')} (now {commit})", flush=True)
        done[(float(cell["drift"]), int(cell["seed"]))] = cell
    return done


def evaluate_agent(agent, norm, drift, a, dev, seed, agent_name=""):
    dump_path = tdump = cdump = None
    if getattr(a, "dump_states", None):
        stem = os.path.join(a.dump_states, f"states_d{drift:.2f}_s{seed}_{agent_name}")
        dump_path = stem + ".npz"
        tdump, cdump = stem + "_h7transfer.npz", stem + "_cg.npz"
    heldout = getattr(a, "heldout_evals", False)
    pr = pooled_readout(agent, norm, P, drift, n_eps=a.pool_n, steps=a.pool_steps,
                        ray_steps=a.ray_steps, device=dev, seed=seed, dump_path=dump_path,
                        return_pools=heldout)
    if heldout:
        pool, pools = pr    # pooled_readout returns (metrics, (Ha, Hs)) with return_pools=True
    else:
        pool, pools = pr, None
    mp = readout(agent, norm, P, drift, n_pairs=a.mp_pairs, prefix_steps=a.mp_prefix,
                 branch_steps=a.mp_branch, ray_steps=a.ray_steps, device=dev, seed=seed)
    ho = None
    if heldout:
        from itasorl.experiment_b2 import cg_probe, common_garden_rollout, transfer_readout
        Ha, Hs = pools
        ho = transfer_readout(agent, norm, P, drift, Ha, Hs, n_eps=a.pool_n,
                              steps=a.pool_steps, ray_steps=a.ray_steps, device=dev,
                              seed=seed, dump_path=tdump)
        at, st = common_garden_rollout(agent, norm, P, drift, n_pairs=a.pool_n,
                                       prefix_steps=a.cg_prefix, tail_steps=a.cg_steps,
                                       ray_steps=a.ray_steps, device=dev)
        ho.update(cg_probe(at, st, seed=seed))
        if cdump:
            np.savez_compressed(cdump, auth=np.stack(at) if at else np.zeros((0, a.cg_steps, a.hidden), np.float32),
                                surr=np.stack(st) if st else np.zeros((0, a.cg_steps, a.hidden), np.float32))
    return pool, mp, ho


def run_cell(task: dict) -> dict:
    """Train the 3 agents for one (drift, seed) cell and return ALL metrics as plain
    floats/dicts (picklable). Self-contained so it can run in a worker process: the B-v2
    training is CPU-bound (serial physics), and every cell is independent, so a pool of
    workers scales the run with CPU core count. Re-applies scarcity overrides because a
    spawned worker re-imports the module with fresh defaults."""
    import torch
    import itasorl.experiment_b2 as b2
    torch.set_num_threads(1)                      # one core per worker; the pool provides the width
    d, s, dev, k = task["drift"], task["seed"], task["device"], task
    if k.get("basal_e") is not None:
        b2.SURVIVAL_METAB["basal_E"] = k["basal_e"]
    if k.get("n_pellets") is not None:
        b2.SURVIVAL_FOOD["n_pellets"] = k["n_pellets"]
    if k.get("reach") is not None:
        b2.SURVIVAL_FOOD["reach"] = k["reach"]
    if k.get("drift_mode"):
        b2.DRIFT_MODE = k["drift_mode"]
    if k.get("drift_mode") == "l3" and b2._L3_GMOTION is None:  # train G_motion once per worker
        b2.setup_l3_surrogate(hidden=k.get("l3_hidden", 8), device=dev, seed=0, params=P)  # THIS world
    if k.get("heldout_evals") and b2._L3_GMOTION_HELDOUT is None:  # once per worker
        b2.setup_l3_heldout_surrogate(hidden=k["heldout_hidden"], device=dev, seed=0, params=P)

    agents = {"untrained": untrained_agent(P, d, k["ray_steps"], k["hidden"], 64, True, dev, seed=s),
              "predictor": train_predictor_only(d, P, n_eps=k["n_eps"], updates=k["updates"],
                                                hidden=k["hidden"], max_steps=k["max_steps"],
                                                ray_steps=k["ray_steps"], seed=s, device=dev)}
    sa, sn, _ = train_actor_critic(d, P, n_eps=k["n_eps"], updates=k["updates"], hidden=k["hidden"],
                                   max_steps=k["max_steps"], ray_steps=k["ray_steps"], seed=s,
                                   device=dev, shaping_coef=k["shaping_coef"],
                                   sysid_aux=k.get("sysid_aux", False),
                                   sysid_coef=k.get("sysid_coef", 1.0))
    agents["survival"] = (sa, sn)
    if k.get("save_agents"):
        from itasorl.experiment_b2 import save_agent_bundle
        for g, (ag, nm) in agents.items():
            save_agent_bundle(os.path.join(k["out_dir"], "agents",
                                           f"agent_d{d:.2f}_s{s}_{g}.pt"), ag, nm)
    eng = engagement_metric(sa, sn, P, d, n_eps=64, max_steps=k["max_steps"],
                            ray_steps=k["ray_steps"], device=dev)
    xev = {f"{ed:.2f}": survival_return(sa, sn, P, ed, max_steps=k["max_steps"],
                                        ray_steps=k["ray_steps"], device=dev) for ed in k["drifts"]}
    a_ns = argparse.Namespace(**k)               # evaluate_agent reads attrs off a namespace
    out = {"drift": d, "seed": s, "eng": eng, "xeval": xev, "agents": {}}
    for g in AG:
        pool, mp, ho = evaluate_agent(agents[g][0], agents[g][1], d, a_ns, dev, s, g)
        out["agents"][g] = {"pool": pool, "mp": mp}
        if ho is not None:
            out["agents"][g]["heldout"] = ho
    return out


def record_cell(res: dict, eng_log: dict, cell: dict) -> None:
    d = cell["drift"]
    eng_log[d].append(cell["eng"])
    res[d]["survival"]["xeval_return"].append(cell["xeval"])
    for g in AG:
        pool, mp = cell["agents"][g]["pool"], cell["agents"][g]["mp"]
        res[d][g]["pool_target"].append(pool["target"])
        res[d][g]["pool_target_lo"].append(pool.get("target_lo"))
        res[d][g]["pool_target_hi"].append(pool.get("target_hi"))
        res[d][g]["pool_target_var"].append(pool.get("target_var"))
        res[d][g]["pool_target_full"].append(pool.get("target_full"))
        res[d][g]["pool_selectivity"].append(pool.get("selectivity"))
        res[d][g]["pool_selectivity_var"].append(pool.get("selectivity_var"))
        res[d][g]["pool_selectivity_full"].append(pool.get("selectivity_full"))
        res[d][g]["pool_speed"].append(pool["speed"])
        res[d][g]["pool_shuffled"].append(pool["shuffled"])
        res[d][g]["pool_anchor_energy"].append(pool.get("anchor_energy"))
        res[d][g]["pool_anchor_food"].append(pool.get("anchor_food"))
        res[d][g]["pool_ceiling_drag"].append(pool.get("ceiling_drag"))
        res[d][g]["pool_reward_leak"].append(pool.get("pool_reward_leak"))
        res[d][g]["pool_leak_clean"].append(bool(pool.get("pool_leak_clean", False)))
        res[d][g]["pool_deaths_auth"].append(pool.get("deaths_auth"))
        res[d][g]["pool_deaths_surr"].append(pool.get("deaths_surr"))
        res[d][g]["mp_target"].append(mp["target"])
        res[d][g]["mp_leak_clean"].append(bool(mp["leakage_clean"]))
        ho = cell["agents"][g].get("heldout")
        if ho is not None:
            for kk in ("transfer_target", "cg_tail_target", "cg_latetail_target", "cg_n_pairs"):
                res[d][g].setdefault(kk, []).append(ho.get(kk))


def print_cell(cell: dict) -> None:
    d, s, eng, xev = cell["drift"], cell["seed"], cell["eng"], cell["xeval"]
    print(f"drift={d:.2f} seed={s} done  engaged={eng['engaged']} "
          f"(trained={eng['trained_return']:+.3f})", flush=True)
    print("   manip-check (survival return by eval drift): "
          + "  ".join(f"@{kk}={vv:+.3f}" for kk, vv in xev.items()), flush=True)
    for g in AG:
        pool, mp = cell["agents"][g]["pool"], cell["agents"][g]["mp"]
        print(f"   {g:10s} pool_target={pool['target']:.3f} "
              f"[{pool.get('target_lo', float('nan')):.3f},{pool.get('target_hi', float('nan')):.3f}] "
              f"var={pool.get('target_var', float('nan')):.3f} "
              f"full={pool.get('target_full', float('nan')):.3f} "
              f"sel(L={pool.get('selectivity', float('nan')):+.3f} "
              f"V={pool.get('selectivity_var', float('nan')):+.3f} "
              f"F={pool.get('selectivity_full', float('nan')):+.3f}) "
              f"ceiling(E={pool.get('anchor_energy', float('nan')):.3f} "
              f"food={pool.get('anchor_food', float('nan')):.3f} "
              f"drag={pool.get('ceiling_drag', float('nan')):.3f}) "
              f"speed+={pool['speed']:.3f} mp_target={mp['target']:.3f} "
              f"leak_clean={mp['leakage_clean']}(dev={mp['leakage_max_dev']:.3f})", flush=True)
        print(f"   {'':10s} pool_leak(reward={pool.get('pool_reward_leak', float('nan')):.3f} "
              f"clean={pool.get('pool_leak_clean')}) "
              f"survivors(auth={pool.get('n_auth', '?')} surr={pool.get('n_surr', '?')} "
              f"/{pool.get('pool_n_eps', '?')}  deaths auth={pool.get('deaths_auth', '?')} "
              f"surr={pool.get('deaths_surr', '?')})", flush=True)
        ho = cell["agents"][g].get("heldout")
        if ho is not None:
            print(f"   {'':10s} heldout: transfer={ho['transfer_target']:.3f} "
                  f"[{ho.get('transfer_lo', float('nan')):.3f},{ho.get('transfer_hi', float('nan')):.3f}] "
                  f"(n={ho['transfer_n_auth']}+{ho['transfer_n_surr']})  "
                  f"cg_tail={ho['cg_tail_target']:.3f} "
                  f"[{ho['cg_tail_lo']:.3f},{ho['cg_tail_hi']:.3f}]  "
                  f"cg_late={ho['cg_latetail_target']:.3f} (pairs={ho['cg_n_pairs']})", flush=True)


def fresh_results(drifts) -> dict:
    return {d: {g: {"pool_target": [], "pool_target_lo": [], "pool_target_hi": [],
                    "pool_target_var": [], "pool_target_full": [],
                    "pool_selectivity": [], "pool_selectivity_var": [],
                    "pool_selectivity_full": [],
                    "pool_speed": [], "pool_shuffled": [],
                    "pool_anchor_energy": [], "pool_anchor_food": [],
                    "pool_ceiling_drag": [],
                    "pool_reward_leak": [], "pool_leak_clean": [],
                    "pool_deaths_auth": [], "pool_deaths_surr": [],
                    "mp_target": [], "mp_leak_clean": [], "xeval_return": []}
                for g in AG} for d in drifts}


def main():
    a = cfg()
    dev = "cpu" if (a.device == "auto" and a.workers > 1) else (
        default_device() if a.device == "auto" else a.device)
    # Stage-2 scarcity overrides patch the module-level survival constants in place (the
    # parent process - run_cell re-applies them inside each worker).
    if a.basal_e is not None:
        b2.SURVIVAL_METAB["basal_E"] = a.basal_e
    if a.n_pellets is not None:
        b2.SURVIVAL_FOOD["n_pellets"] = a.n_pellets
    if a.reach is not None:
        b2.SURVIVAL_FOOD["reach"] = a.reach
    b2.DRIFT_MODE = a.drift_mode
    os.makedirs(a.out_dir, exist_ok=True)
    results_path = os.path.join(a.out_dir, "expB2_results.json")
    print(f"Experiment B-v2 full run  (device={dev}, drifts={a.drifts}, seeds={a.seeds}, "
          f"updates={a.updates}, workers={a.workers})")
    print(f"  survival metabolism={b2.SURVIVAL_METAB}  food={b2.SURVIVAL_FOOD}  drift_mode={b2.DRIFT_MODE}")
    if a.drift_mode == "regime":
        print("  drift_mode=regime: surrogate = per-episode CONSTANT drag offset "
              "(B-v3 identifiable + policy-relevant coupling; see docs/PREREGISTRATION_Bv3.md)")
    if a.drift_mode == "l3":
        print(f"  drift_mode=l3: surrogate = LEARNED velocity law G_motion (hidden={a.l3_hidden}), "
              "the dynamics-level L3 rung; obs come from the REAL sensor model so only the "
              "learned dynamics differ (see docs/PREREGISTRATION_L3.md sec.4/sec.12)")
        b2.setup_l3_surrogate(hidden=a.l3_hidden, device=dev, seed=0, params=P)  # train on THIS world
        if a.heldout_evals:
            print(f"  heldout evals ON: transfer fingerprint G(hidden={a.heldout_hidden}), "
                  f"common garden prefix={a.cg_prefix} tail={a.cg_steps}")
            b2.setup_l3_heldout_surrogate(hidden=a.heldout_hidden, device=dev, seed=0, params=P)
    if a.sysid_aux:
        print("  *** SYSID-AUX ON: survival trunk is supervised on drag (CEILING control, "
              "NOT readout-not-reward). Its target is a capacity ceiling, not H_B2 evidence. ***")
    print()
    # results[drift][agent][metric] = list over seeds
    res = fresh_results(a.drifts)
    eng_log = {d: [] for d in a.drifts}

    def checkpoint():
        """Persist results after every cell so a crash in a long run loses at most one cell."""
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({str(d): {g: res[d][g] for g in AG} for d in a.drifts}, f, indent=2, default=float)

    # One picklable knob dict per (drift, seed) cell; cells are independent.
    base = {k: getattr(a, k) for k in ("updates", "n_eps", "max_steps", "hidden", "ray_steps",
                                       "shaping_coef", "pool_n", "pool_steps", "mp_pairs", "mp_prefix",
                                       "mp_branch", "basal_e", "n_pellets", "reach", "dump_states",
                                       "sysid_aux", "sysid_coef", "drift_mode", "l3_hidden")}
    base.update(drifts=a.drifts, device=dev, out_dir=a.out_dir, save_agents=a.save_agents)
    if a.heldout_evals:
        base.update(heldout_evals=True, heldout_hidden=a.heldout_hidden,
                    cg_prefix=a.cg_prefix, cg_steps=a.cg_steps)
    tasks = [{**base, "drift": d, "seed": s} for d in a.drifts for s in a.seeds]
    cells_dir = Path(a.out_dir) / "cells"
    fingerprint = config_fingerprint(base)
    commit = git_commit_short()
    if a.resume:
        resumed = load_cell_files(cells_dir, fingerprint)
    else:
        resumed = {}
        if cells_dir.is_dir() and any(cells_dir.glob("cell_d*_s*.json")):
            raise SystemExit(
                f"{cells_dir} already contains checkpointed cells. Pass "
                "--resume to continue that run, or use a fresh --out-dir.")
    all_cells = dict(resumed)
    for (d, s) in sorted(resumed):
        print(f"resumed from checkpoint: drift={d:.2f} seed={s}", flush=True)
        record_cell(res, eng_log, resumed[(d, s)])
    tasks = [t for t in tasks if (t["drift"], t["seed"]) not in resumed]
    extra = sorted(set(resumed) - {(d, s) for d in a.drifts for s in a.seeds})
    if extra:
        print(f"  NOTE: {len(extra)} checkpointed cell(s) beyond the requested "
              f"drifts x seeds grid are included in results: {extra}", flush=True)
    if resumed:
        print(f"Resume: {len(resumed)} cell(s) loaded from {cells_dir}, "
              f"{len(tasks)} to run.", flush=True)
    done = 0
    if a.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")            # spawn: safe with torch, re-imports cleanly
        with ctx.Pool(a.workers) as pool:
            for cell in pool.imap_unordered(run_cell, tasks):
                record_cell(res, eng_log, cell)
                all_cells[(cell["drift"], cell["seed"])] = cell
                write_cell_file(cells_dir, fingerprint, commit, cell)
                print_cell(cell)
                done += 1
                print(f"   [{done}/{len(tasks)} cells done]", flush=True)
                checkpoint()                     # crash-safe: each finished cell is persisted
    else:
        for task in tasks:
            cell = run_cell(task)
            record_cell(res, eng_log, cell)
            all_cells[(cell["drift"], cell["seed"])] = cell
            write_cell_file(cells_dir, fingerprint, commit, cell)
            print_cell(cell)
            done += 1
            checkpoint()

    # Canonical rebuild: list positions ordered by (drift, seed), independent of
    # completion order (imap_unordered) and of resume interleaving.
    res = fresh_results(a.drifts)
    eng_log = {d: [] for d in a.drifts}
    for key in sorted(all_cells):
        record_cell(res, eng_log, all_cells[key])
    checkpoint()

    # ---- summary table ----
    print("\n================  SUMMARY (mean +/- std over seeds)  ================")
    for d in a.drifts:
        print(f"\ndrift_sigma = {d:.2f}")
        eng_ok = np.mean([e["engaged"] for e in eng_log[d]])
        print(f"  engagement passed in {eng_ok*100:.0f}% of seeds")
        for g in AG:
            t = np.array(res[d][g]["pool_target"]); sp = np.array(res[d][g]["pool_speed"])
            ce = np.array(res[d][g]["pool_anchor_energy"], float)
            cf = np.array(res[d][g]["pool_anchor_food"], float)
            _, lo, hi = mean_ci(t, level=0.90)
            print(f"  {g:10s} PRIMARY pool target = {t.mean():.3f}+/-{t.std():.3f}   "
                  f"speed(+ctrl) = {sp.mean():.3f}   "
                  f"mp_target = {np.mean(res[d][g]['mp_target']):.3f}")
            cd = np.array(res[d][g]["pool_ceiling_drag"], float)
            print(f"             across-seed 90% CI = [{lo:.3f},{hi:.3f}]   "
                  f"ceiling(energy={np.nanmean(ce):.3f} food={np.nanmean(cf):.3f} "
                  f"drag={np.nanmean(cd):.3f})")
            tv = np.array(res[d][g]["pool_target_var"], float)
            tf = np.array(res[d][g]["pool_target_full"], float)
            sl = np.array(res[d][g]["pool_selectivity"], float)
            slv = np.array(res[d][g]["pool_selectivity_var"], float)
            slf = np.array(res[d][g]["pool_selectivity_full"], float)
            print(f"             volatility readout: target_var={np.nanmean(tv):.3f} "
                  f"target_full={np.nanmean(tf):.3f}   selectivity(L={np.nanmean(sl):+.3f} "
                  f"V={np.nanmean(slv):+.3f} F={np.nanmean(slf):+.3f})")
            # POOLED confound controls: reward must NOT decode the world (else the headline
            # target is reward-mediated), and per-world death counts bound survivorship bias.
            rl = np.array(res[d][g]["pool_reward_leak"], float)
            lc = res[d][g]["pool_leak_clean"]
            da = np.array(res[d][g]["pool_deaths_auth"], float)
            ds = np.array(res[d][g]["pool_deaths_surr"], float)
            clean_frac = float(np.mean(lc)) if lc else float("nan")
            print(f"             pooled confounds: reward_leak={np.nanmean(rl):.3f} "
                  f"(clean {clean_frac*100:.0f}% of seeds)   "
                  f"deaths/pool auth={np.nanmean(da):.1f} surr={np.nanmean(ds):.1f}")

    # ---- manipulation check: is the L2 artifact survival-relevant? ----
    if len(a.drifts) > 1:
        print("\nManipulation check - survival TRUE return, train_drift x eval_drift "
              "(same world seeds, only drag differs):")
        for td in a.drifts:
            xe = res[td]["survival"]["xeval_return"]  # list over seeds of {eval_drift: ret}
            row = {f"{ed:.2f}": np.mean([s[f"{ed:.2f}"] for s in xe]) for ed in a.drifts}
            print(f"  train@{td:.2f}: " + "  ".join(f"eval@{k}={v:+.3f}" for k, v in row.items()))
        dmax = max(a.drifts)
        if 0.0 in a.drifts and dmax > 0.0:
            r_aa = np.mean([s["0.00"] for s in res[0.0]["survival"]["xeval_return"]])
            r_ad = np.mean([s[f"{dmax:.2f}"] for s in res[0.0]["survival"]["xeval_return"]])
            r_dd = np.mean([s[f"{dmax:.2f}"] for s in res[dmax]["survival"]["xeval_return"]])
            drop = r_aa - r_ad           # drift-0 policy loses this much when drag shifts
            recover = r_dd - r_ad        # training under drift recovers this much
            relevant = drop > 0.05 or recover > 0.05
            print(f"  drift-0 policy loses {drop:+.3f} return under eval@{dmax:.2f}; "
                  f"drift-trained recovers {recover:+.3f}")
            print(f"  -> artifact survival-relevant: {relevant}  "
                  f"({'null is informative' if relevant else 'WARNING: drag may not matter; null is weak'})")

    # ---- L0 equivalence gate on the survival agent (drift=0 must be at chance) ----
    if 0.0 in a.drifts:
        l0_raw = np.asarray(res[0.0]["survival"]["pool_target"], float)
        l0 = l0_raw[np.isfinite(l0_raw)]
        n_l0_bad = int(l0_raw.size - l0.size)
        if n_l0_bad:
            print(f"\nWARNING: {n_l0_bad}/{l0_raw.size} L0 cells non-finite "
                  "(too_few_survivors) - L0 gate is UNINFORMATIVE, not passing")
        eq = equivalence_test(l0, h0=0.5, margin=0.05)
        rp = rope_test(l0, rope=(0.45, 0.55))
        print("\nL0 control (survival pooled target @ drift=0):")
        print(f"  TOST  {eq}")
        print(f"  ROPE  {rp}")
        l0_ceiling = np.nanmean(np.array(res[0.0]["survival"]["pool_anchor_energy"], float))
        print(f"  ceiling(energy) @ L0 = {l0_ceiling:.3f}  "
              f"({'apparatus ALIVE' if l0_ceiling >= 0.65 else 'WARNING: low ceiling - probe may be weak'})")

    # ---- decision readout on the strongest drift ----
    dmax = max(a.drifts)
    h_b2, zone, n_bad, surv_t = decide_h_b2(
        res[dmax]["survival"]["pool_target"],
        res[dmax]["predictor"]["pool_target"],
        res[dmax]["untrained"]["pool_target"])
    if n_bad:
        print(f"\nWARNING: {n_bad} decision cell(s) non-finite (too_few_survivors) "
              f"at drift={dmax:.2f}; stats below use finite seeds only")
    pred_t = np.asarray(res[dmax]["predictor"]["pool_target"], float)
    pred_t = pred_t[np.isfinite(pred_t)]
    untr_t = np.asarray(res[dmax]["untrained"]["pool_target"], float)
    untr_t = untr_t[np.isfinite(untr_t)]
    # keep the display prints warning-free even if a whole arm was non-finite
    surv_t, pred_t, untr_t = (x if x.size else np.array([float("nan")])
                              for x in (surv_t, pred_t, untr_t))
    _, sm_lo, sm_hi = mean_ci(surv_t, level=0.90)
    surv_ceil = np.nanmean(np.array(res[dmax]["survival"]["pool_anchor_energy"], float))
    surv_ceil_drag = np.nanmean(np.array(res[dmax]["survival"]["pool_ceiling_drag"], float))
    print(f"\nAt strongest drift={dmax:.2f}: survival pooled target |dev|="
          f"{abs(surv_t.mean()-0.5):.3f}  predictor |dev|={abs(pred_t.mean()-0.5):.3f}  "
          f"untrained |dev|={abs(untr_t.mean()-0.5):.3f}")
    print(f"  survival target = {surv_t.mean():.3f} (90% CI [{sm_lo:.3f},{sm_hi:.3f}])  "
          f"ceiling(energy={surv_ceil:.3f} drag={surv_ceil_drag:.3f})")
    print(f"  dissociation: drag-ceiling {surv_ceil_drag:.3f} vs identity-target {surv_t.mean():.3f}"
          f"{' (tracks dynamics, no persistent identity)' if surv_ceil_drag - surv_t.mean() > 0.1 else ''}")
    # POOLED confound gate on the HEADLINE: the survival target must not be reward-mediated.
    surv_rl = np.nanmean(np.array(res[dmax]["survival"]["pool_reward_leak"], float))
    surv_lc = res[dmax]["survival"]["pool_leak_clean"]
    surv_da = np.nanmean(np.array(res[dmax]["survival"]["pool_deaths_auth"], float))
    surv_ds = np.nanmean(np.array(res[dmax]["survival"]["pool_deaths_surr"], float))
    clean_frac = float(np.mean(surv_lc)) if surv_lc else float("nan")
    print(f"  pooled leakage gate: reward_leak={surv_rl:.3f} (clean {clean_frac*100:.0f}% of seeds) "
          f"-> {'headline reads the ARTIFACT, not reward' if clean_frac >= 0.5 else 'WARNING: reward confound'}")
    print(f"  survivorship: mean deaths/pool auth={surv_da:.1f} surr={surv_ds:.1f} "
          f"(of {a.pool_n}) -> {'no asymmetry' if max(surv_da, surv_ds) < 0.05 * a.pool_n else 'CHECK asymmetry'}")
    # Pre-registered primary: survival >= 0.65 AND beats predictor and untrained by >= 0.05.
    # Three zones per PREREGISTRATION_Bv3.md section 8 (plus the UNINFORMATIVE zone when a
    # non-finite cell shows the survivorship precondition failed); see decide_h_b2 above.
    print(f"  primary H_B2 (survival-induced encoding) {zone}")
    # Secondary: does the world-identity signal live in a VOLATILITY signature the LEVEL
    # probe throws away? target_var/full crossing 0.65 while the level target stays ~0.5
    # means the null was partly an operationalization artifact, not absent encoding.
    def _finite_mean(vals) -> float:
        x = np.asarray(vals, float)
        return float(x[np.isfinite(x)].mean()) if np.isfinite(x).any() else float("nan")
    surv_tv = _finite_mean(res[dmax]["survival"]["pool_target_var"])
    surv_tf = _finite_mean(res[dmax]["survival"]["pool_target_full"])
    vol_hit = max(surv_tv, surv_tf) >= 0.65
    print(f"  volatility check: survival target_var={surv_tv:.3f} target_full={surv_tf:.3f} "
          f"(bar 0.65) -> {'VOLATILITY-ENCODED (level probe was mis-specified)' if vol_hit else 'no volatility encoding either'}")

    if res[dmax]["survival"].get("transfer_target"):
        print("\nHeld-out channels at strongest drift (frozen rules, spec 2026-07-14):")
        for g in AG:
            tt = np.nanmean(np.array(res[dmax][g]["transfer_target"], float))
            ct = np.nanmean(np.array(res[dmax][g]["cg_tail_target"], float))
            cl = np.nanmean(np.array(res[dmax][g]["cg_latetail_target"], float))
            print(f"  {g:10s} transfer={tt:.3f}  cg_tail={ct:.3f}  cg_late={cl:.3f}")
        s_tt = np.nanmean(np.array(res[dmax]["survival"]["transfer_target"], float))
        u_tt = np.nanmean(np.array(res[dmax]["untrained"]["transfer_target"], float))
        s_cg = np.nanmean(np.array(res[dmax]["survival"]["cg_tail_target"], float))
        u_cg = np.nanmean(np.array(res[dmax]["untrained"]["cg_tail_target"], float))
        t_ok = s_tt >= 0.65 and s_tt >= u_tt + 0.05
        c_ok = s_cg >= 0.65 and s_cg >= u_cg + 0.05
        print(f"  transfer rule (surv>=0.65 AND >untrained+0.05): "
              f"{'GENERALIZES beyond the trained fingerprint' if t_ok else 'fingerprint-specific (informative negative)'}")
        print(f"  common-garden rule (surv>=0.65 AND >untrained+0.05): "
              f"{'PERSISTENT world identity' if c_ok else 'reactive tracking (informative negative)'}")

    checkpoint()
    print(f"saved {results_path}")

    # ---- figure: primary pooled |target-0.5| vs drift, per agent ----
    plt.figure(figsize=(7.6, 4.6))
    colors = {"untrained": "#a0aec0", "predictor": "#3182ce", "survival": "#6b46c1"}
    for g in AG:
        m = [abs(np.mean(res[d][g]["pool_target"]) - 0.5) for d in a.drifts]
        e = [np.std(res[d][g]["pool_target"]) for d in a.drifts]
        plt.errorbar(a.drifts, m, yerr=e, fmt="o-", capsize=4, lw=2, color=colors[g], label=g)
    plt.axhline(0.0, ls=":", color="grey")
    plt.xlabel("surrogate strength (L2 drift_sigma)")
    plt.ylabel("|pooled target AUROC - 0.5|   (incidental encoding)")
    plt.title("ITASORL Experiment B-v2 - does survival pressure induce incidental encoding?")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    os.makedirs(a.out_dir, exist_ok=True)
    fig_path = os.path.join(a.out_dir, "expB2_survival.png")
    plt.savefig(fig_path, dpi=130)
    plt.close("all")
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
