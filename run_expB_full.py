import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from world import WorldParams
from experiment_b import (collect_episodes, train_world_model, states_torch,
                          episode_features, probe_auroc)

PARAMS = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
STEPS, RAY, POOL, SUB, EPOCHS, HIDDEN = 28, 5, 150, 110, 15, 96
SEEDS, DRIFTS = [0, 1, 2], [0.0, 0.2, 0.45]

print("Collecting episode pools ...")
authA = collect_episodes(0, POOL, STEPS, 0.0, RAY, seed0=10000, params=PARAMS)
authB = collect_episodes(0, POOL, STEPS, 0.0, RAY, seed0=20000, params=PARAMS)  # 'other world' for the drift=0 control
surr = {d: collect_episodes(1, POOL, STEPS, d, RAY, seed0=30000 + int(d * 1000), params=PARAMS)
        for d in DRIFTS if d > 0}

def subsample(pool, k, rng):
    return [pool[i] for i in rng.choice(len(pool), size=k, replace=False)]

res = {d: {"target": [], "shuffled": [], "speed": []} for d in DRIFTS}
print("\nTraining + probing (per drift level, across seeds) ...")
for d in DRIFTS:
    other = authB if d == 0 else surr[d]
    for seed in SEEDS:
        rng = np.random.default_rng(1000 + seed)
        np.random.seed(seed)
        a = [dict(e, label=0) for e in subsample(authA, SUB, rng)]
        b = [dict(e, label=1) for e in subsample(other, SUB, rng)]
        eps = a + b
        model, norm = train_world_model(eps, epochs=EPOCHS, hidden=HIDDEN, seed=seed)
        X = episode_features(states_torch(model, eps, norm))
        y = np.array([e["label"] for e in eps]); spd = np.array([e["speed"] for e in eps])
        rr = np.random.default_rng(seed)
        res[d]["target"].append(probe_auroc(X, y))
        res[d]["shuffled"].append(probe_auroc(X, rr.permutation(y)))
        res[d]["speed"].append(probe_auroc(X, (spd > np.median(spd)).astype(int)))
    m = lambda k: (np.mean(res[d][k]), np.std(res[d][k]))
    print(f"  drift={d:.2f}  target={m('target')[0]:.3f}±{m('target')[1]:.3f}   "
          f"shuffled={m('shuffled')[0]:.3f}±{m('shuffled')[1]:.3f}   "
          f"speed(+ctrl)={m('speed')[0]:.3f}±{m('speed')[1]:.3f}")

tg = [np.mean(res[d]["target"]) for d in DRIFTS]; tge = [np.std(res[d]["target"]) for d in DRIFTS]
sh = [np.mean(res[d]["shuffled"]) for d in DRIFTS]; she = [np.std(res[d]["shuffled"]) for d in DRIFTS]
plt.figure(figsize=(7.2, 4.4))
plt.errorbar(DRIFTS, tg, yerr=tge, fmt="o-", color="#6b46c1", capsize=4, lw=2, label="target: decode world identity")
plt.errorbar(DRIFTS, sh, yerr=she, fmt="s--", color="#a0aec0", capsize=4, label="negative control (shuffled labels)")
plt.axhline(0.5, ls=":", color="grey")
plt.xlabel("surrogate strength  (L2 drift_sigma)")
plt.ylabel("probe AUROC  (per episode)")
plt.title("ITASORL Experiment B - incidental detection from the agent's recurrent state")
plt.ylim(0.3, 1.02); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("docs/figures/expB_incidental.png", dpi=130)
print("\nsaved docs/figures/expB_incidental.png")
