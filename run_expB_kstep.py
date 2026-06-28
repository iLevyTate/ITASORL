import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from world import WorldParams
from experiment_b import (collect_episodes, train_world_model, states_torch,
                          episode_features, probe_auroc)

PARAMS = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
T, RAY, POOL, SUB, EPOCHS, HIDDEN = 24, 5, 150, 110, 18, 96
SEEDS = [0, 1, 2]
OPEN = [0, 8, 16]          # open-loop horizon: 0 = pure next-step (original); higher = deeper imagination
DRIFTS = [0.0, 0.45]       # control vs strong drift

print("Collecting pools ...")
authA = collect_episodes(0, POOL, T, 0.0, RAY, seed0=10000, params=PARAMS)
authB = collect_episodes(0, POOL, T, 0.0, RAY, seed0=20000, params=PARAMS)
surrD = collect_episodes(1, POOL, T, 0.45, RAY, seed0=30450, params=PARAMS)

def subsample(pool, k, rng):
    return [pool[i] for i in rng.choice(len(pool), k, replace=False)]

res = {(oh, d): [] for oh in OPEN for d in DRIFTS}
print("Training across open-loop horizons x drift x seeds ...")
for oh in OPEN:
    ctx = None if oh == 0 else (T - oh)
    for d in DRIFTS:
        other = authB if d == 0 else surrD
        for seed in SEEDS:
            rng = np.random.default_rng(1000 + seed); np.random.seed(seed)
            a = [dict(e, label=0) for e in subsample(authA, SUB, rng)]
            b = [dict(e, label=1) for e in subsample(other, SUB, rng)]
            eps = a + b
            model, norm = train_world_model(eps, epochs=EPOCHS, hidden=HIDDEN, seed=seed, rollout_context=ctx)
            X = episode_features(states_torch(model, eps, norm))
            y = np.array([e["label"] for e in eps])
            res[(oh, d)].append(probe_auroc(X, y))
    print(f"  open_horizon={oh:2d}:  drift0.45 target={np.mean(res[(oh,0.45)]):.3f}±{np.std(res[(oh,0.45)]):.3f}"
          f"   control target={np.mean(res[(oh,0.0)]):.3f}±{np.std(res[(oh,0.0)]):.3f}")

td = [np.mean(res[(oh, 0.45)]) for oh in OPEN]; tde = [np.std(res[(oh, 0.45)]) for oh in OPEN]
tc = [np.mean(res[(oh, 0.0)]) for oh in OPEN]; tce = [np.std(res[(oh, 0.0)]) for oh in OPEN]
plt.figure(figsize=(7.2, 4.4))
plt.errorbar(OPEN, td, yerr=tde, fmt="o-", color="#6b46c1", capsize=4, lw=2, label="drift=0.45 (surrogate)")
plt.errorbar(OPEN, tc, yerr=tce, fmt="s--", color="#a0aec0", capsize=4, label="drift=0 (control)")
plt.axhline(0.5, ls=":", color="grey")
plt.xlabel("open-loop prediction horizon  (steps imagined from state)")
plt.ylabel("recurrent-state probe AUROC")
plt.title("ITASORL Experiment B - does a longer-horizon objective induce encoding?")
plt.ylim(0.3, 1.02); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("docs/figures/expB_kstep.png", dpi=130)
print("\nsaved docs/figures/expB_kstep.png")
