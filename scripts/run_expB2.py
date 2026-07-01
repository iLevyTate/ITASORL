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
import json
import os

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
    # Speedup: the run is CPU-bound (serial physics, tiny nets), and (drift,seed) cells are
    # independent. --workers N runs N cells at once across CPU cores. Set N ~ vCPU count.
    ap.add_argument("--workers", type=int, default=1, help="parallel worker processes over cells")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto",
                    help="auto = cuda when --workers 1, else cpu (avoids GPU contention; CPU-bound anyway)")
    a = ap.parse_args()
    if a.quick:
        a.drifts, a.seeds, a.updates, a.n_eps, a.max_steps = [0.0, 0.45], [0, 1], 60, 8, 40
        a.hidden, a.ray_steps, a.pool_n, a.pool_steps = 64, 4, 40, 16
        a.mp_pairs, a.mp_prefix, a.mp_branch = 25, 12, 16
    return a


def evaluate_agent(agent, norm, drift, a, dev, seed, agent_name=""):
    dump_path = None
    if getattr(a, "dump_states", None):
        dump_path = os.path.join(a.dump_states, f"states_d{drift:.2f}_s{seed}_{agent_name}.npz")
    pool = pooled_readout(agent, norm, P, drift, n_eps=a.pool_n, steps=a.pool_steps,
                          ray_steps=a.ray_steps, device=dev, seed=seed, dump_path=dump_path)
    mp = readout(agent, norm, P, drift, n_pairs=a.mp_pairs, prefix_steps=a.mp_prefix,
                 branch_steps=a.mp_branch, ray_steps=a.ray_steps, device=dev, seed=seed)
    return pool, mp


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
    eng = engagement_metric(sa, sn, P, d, n_eps=64, max_steps=k["max_steps"],
                            ray_steps=k["ray_steps"], device=dev)
    xev = {f"{ed:.2f}": survival_return(sa, sn, P, ed, max_steps=k["max_steps"],
                                        ray_steps=k["ray_steps"], device=dev) for ed in k["drifts"]}
    a_ns = argparse.Namespace(**k)               # evaluate_agent reads attrs off a namespace
    out = {"drift": d, "seed": s, "eng": eng, "xeval": xev, "agents": {}}
    for g in AG:
        pool, mp = evaluate_agent(agents[g][0], agents[g][1], d, a_ns, dev, s, g)
        out["agents"][g] = {"pool": pool, "mp": mp}
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
        res[d][g]["mp_target"].append(mp["target"])
        res[d][g]["mp_leak_clean"].append(bool(mp["leakage_clean"]))


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
    os.makedirs(a.out_dir, exist_ok=True)
    results_path = os.path.join(a.out_dir, "expB2_results.json")
    print(f"Experiment B-v2 full run  (device={dev}, drifts={a.drifts}, seeds={a.seeds}, "
          f"updates={a.updates}, workers={a.workers})")
    print(f"  survival metabolism={b2.SURVIVAL_METAB}  food={b2.SURVIVAL_FOOD}")
    if a.sysid_aux:
        print("  *** SYSID-AUX ON: survival trunk is supervised on drag (CEILING control, "
              "NOT readout-not-reward). Its target is a capacity ceiling, not H_B2 evidence. ***")
    print()
    # results[drift][agent][metric] = list over seeds
    res = {d: {g: {"pool_target": [], "pool_target_lo": [], "pool_target_hi": [],
                   "pool_target_var": [], "pool_target_full": [],
                   "pool_selectivity": [], "pool_selectivity_var": [], "pool_selectivity_full": [],
                   "pool_speed": [], "pool_shuffled": [],
                   "pool_anchor_energy": [], "pool_anchor_food": [], "pool_ceiling_drag": [],
                   "mp_target": [], "mp_leak_clean": [], "xeval_return": []} for g in AG} for d in a.drifts}
    eng_log = {d: [] for d in a.drifts}

    def checkpoint():
        """Persist results after every cell so a crash in a long run loses at most one cell."""
        with open(results_path, "w") as f:
            json.dump({str(d): {g: res[d][g] for g in AG} for d in a.drifts}, f, indent=2, default=float)

    # One picklable knob dict per (drift, seed) cell; cells are independent.
    base = {k: getattr(a, k) for k in ("updates", "n_eps", "max_steps", "hidden", "ray_steps",
                                       "shaping_coef", "pool_n", "pool_steps", "mp_pairs", "mp_prefix",
                                       "mp_branch", "basal_e", "n_pellets", "reach", "dump_states",
                                       "sysid_aux", "sysid_coef")}
    base.update(drifts=a.drifts, device=dev)
    tasks = [{**base, "drift": d, "seed": s} for d in a.drifts for s in a.seeds]
    done = 0
    if a.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")            # spawn: safe with torch, re-imports cleanly
        with ctx.Pool(a.workers) as pool:
            for cell in pool.imap_unordered(run_cell, tasks):
                record_cell(res, eng_log, cell)
                print_cell(cell)
                done += 1
                print(f"   [{done}/{len(tasks)} cells done]", flush=True)
                checkpoint()                     # crash-safe: each finished cell is persisted
    else:
        for task in tasks:
            cell = run_cell(task)
            record_cell(res, eng_log, cell)
            print_cell(cell)
            done += 1
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
        l0 = res[0.0]["survival"]["pool_target"]
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
    surv_t = np.array(res[dmax]["survival"]["pool_target"])
    pred_t = np.array(res[dmax]["predictor"]["pool_target"])
    untr_t = np.array(res[dmax]["untrained"]["pool_target"])
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
    # Pre-registered primary: survival >= 0.65 AND beats predictor and untrained by >= 0.05.
    h_b2 = (surv_t.mean() >= 0.65 and surv_t.mean() >= pred_t.mean() + 0.05
            and surv_t.mean() >= untr_t.mean() + 0.05)
    print(f"  primary H_B2 (survival-induced encoding) {'MET' if h_b2 else 'NOT met'}  "
          f"-> {'encoding induced' if h_b2 else 'strengthened negative (state readable, identity not encoded)'}")
    # Secondary: does the world-identity signal live in a VOLATILITY signature the LEVEL
    # probe throws away? target_var/full crossing 0.65 while the level target stays ~0.5
    # means the null was partly an operationalization artifact, not absent encoding.
    surv_tv = np.nanmean(np.array(res[dmax]["survival"]["pool_target_var"], float))
    surv_tf = np.nanmean(np.array(res[dmax]["survival"]["pool_target_full"], float))
    vol_hit = max(surv_tv, surv_tf) >= 0.65
    print(f"  volatility check: survival target_var={surv_tv:.3f} target_full={surv_tf:.3f} "
          f"(bar 0.65) -> {'VOLATILITY-ENCODED (level probe was mis-specified)' if vol_hit else 'no volatility encoding either'}")

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
    print(f"saved {fig_path}")


if __name__ == "__main__":
    main()
