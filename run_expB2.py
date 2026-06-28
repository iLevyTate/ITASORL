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

Gates (pre-registered, see PREREGISTRATION.md) before the survival target counts:
    engagement   - trained return > random AND scripted (else "uninformative")
    L0 control   - drift=0 pooled target equivalent to 0.5 (TOST)
    positive ctrl- speed probe high
    leakage      - reward/length/lifetime ~0.5

Usage:  python run_expB2.py            # full run (CUDA if available)
        python run_expB2.py --quick    # fast sanity pass
"""

from __future__ import annotations

import argparse
import json

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiment_b2 import (  # noqa: E402
    default_device,
    engagement_metric,
    pooled_readout,
    readout,
    train_actor_critic,
    train_predictor_only,
    untrained_agent,
)
from stats import equivalence_test  # noqa: E402
from world import WorldParams  # noqa: E402

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)


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
    a = ap.parse_args()
    if a.quick:
        a.drifts, a.seeds, a.updates, a.n_eps, a.max_steps = [0.0, 0.45], [0, 1], 60, 8, 40
        a.hidden, a.ray_steps, a.pool_n, a.pool_steps = 64, 4, 40, 16
        a.mp_pairs, a.mp_prefix, a.mp_branch = 25, 12, 16
    return a


def evaluate_agent(agent, norm, drift, a, dev, seed):
    pool = pooled_readout(agent, norm, P, drift, n_eps=a.pool_n, steps=a.pool_steps,
                          ray_steps=a.ray_steps, device=dev, seed=seed)
    mp = readout(agent, norm, P, drift, n_pairs=a.mp_pairs, prefix_steps=a.mp_prefix,
                 branch_steps=a.mp_branch, ray_steps=a.ray_steps, device=dev, seed=seed)
    return pool, mp


def main():
    a = cfg()
    dev = default_device()
    print(f"Experiment B-v2 full run  (device={dev}, drifts={a.drifts}, seeds={a.seeds}, "
          f"updates={a.updates})\n")
    # results[drift][agent][metric] = list over seeds
    AG = ("untrained", "predictor", "survival")
    res = {d: {g: {"pool_target": [], "pool_speed": [], "pool_shuffled": [],
                   "mp_target": [], "mp_leak_clean": []} for g in AG} for d in a.drifts}
    eng_log = {d: [] for d in a.drifts}

    for d in a.drifts:
        for s in a.seeds:
            print(f"drift={d:.2f} seed={s} ...", flush=True)
            agents = {}
            agents["untrained"] = untrained_agent(P, d, a.ray_steps, a.hidden, 64, True, dev, seed=s)
            agents["predictor"] = train_predictor_only(d, P, n_eps=a.n_eps, updates=a.updates,
                                                       hidden=a.hidden, max_steps=a.max_steps,
                                                       ray_steps=a.ray_steps, seed=s, device=dev)
            sa, sn, _ = train_actor_critic(d, P, n_eps=a.n_eps, updates=a.updates, hidden=a.hidden,
                                           max_steps=a.max_steps, ray_steps=a.ray_steps, seed=s,
                                           device=dev, shaping_coef=a.shaping_coef)
            agents["survival"] = (sa, sn)
            eng = engagement_metric(sa, sn, P, d, n_eps=64, max_steps=a.max_steps,
                                    ray_steps=a.ray_steps, device=dev)
            eng_log[d].append(eng)
            print(f"   engagement: trained={eng['trained_return']:+.3f} "
                  f"random={eng['random_return']:+.3f} scripted={eng['scripted_return']:+.3f} "
                  f"engaged={eng['engaged']}", flush=True)
            for g in AG:
                pool, mp = evaluate_agent(agents[g][0], agents[g][1], d, a, dev, s)
                res[d][g]["pool_target"].append(pool["target"])
                res[d][g]["pool_speed"].append(pool["speed"])
                res[d][g]["pool_shuffled"].append(pool["shuffled"])
                res[d][g]["mp_target"].append(mp["target"])
                res[d][g]["mp_leak_clean"].append(bool(mp["leakage_clean"]))
                print(f"   {g:10s} pool_target={pool['target']:.3f} (speed+={pool['speed']:.3f}) "
                      f"mp_target={mp['target']:.3f} leak_clean={mp['leakage_clean']}", flush=True)

    # ---- summary table ----
    print("\n================  SUMMARY (mean +/- std over seeds)  ================")
    for d in a.drifts:
        print(f"\ndrift_sigma = {d:.2f}")
        eng_ok = np.mean([e["engaged"] for e in eng_log[d]])
        print(f"  engagement passed in {eng_ok*100:.0f}% of seeds")
        for g in AG:
            t = np.array(res[d][g]["pool_target"]); sp = np.array(res[d][g]["pool_speed"])
            print(f"  {g:10s} PRIMARY pool target = {t.mean():.3f}+/-{t.std():.3f}   "
                  f"speed(+ctrl) = {sp.mean():.3f}   "
                  f"mp_target = {np.mean(res[d][g]['mp_target']):.3f}")

    # ---- L0 equivalence gate on the survival agent (drift=0 must be at chance) ----
    if 0.0 in a.drifts:
        eq = equivalence_test(res[0.0]["survival"]["pool_target"], h0=0.5, margin=0.05)
        print(f"\nL0 control (survival pooled target @ drift=0): {eq}")

    # ---- decision readout on the strongest drift ----
    dmax = max(a.drifts)
    surv_t = np.array(res[dmax]["survival"]["pool_target"])
    pred_t = np.array(res[dmax]["predictor"]["pool_target"])
    untr_t = np.array(res[dmax]["untrained"]["pool_target"])
    print(f"\nAt strongest drift={dmax:.2f}: survival pooled target |dev|="
          f"{abs(surv_t.mean()-0.5):.3f}  predictor |dev|={abs(pred_t.mean()-0.5):.3f}  "
          f"untrained |dev|={abs(untr_t.mean()-0.5):.3f}")

    with open("expB2_results.json", "w") as f:
        json.dump({str(d): {g: res[d][g] for g in AG} for d in a.drifts}, f, indent=2, default=float)
    print("saved expB2_results.json")

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
    plt.savefig("expB2_survival.png", dpi=130)
    print("saved expB2_survival.png")


if __name__ == "__main__":
    main()
