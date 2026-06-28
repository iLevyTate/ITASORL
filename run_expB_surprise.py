import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import torch
from world import WorldParams
from experiment_b import collect_episodes, train_world_model, _stack
from experiment_a import grouped_auroc, continuous_dims

PARAMS = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)
STEPS, RAY, POOL, SUB, EPOCHS, HIDDEN = 28, 5, 150, 110, 15, 96
SEEDS, DRIFTS = [0, 1, 2], [0.0, 0.2, 0.45]
CDIMS = continuous_dims()

authA = collect_episodes(0, POOL, STEPS, 0.0, RAY, seed0=10000, params=PARAMS)
authB = collect_episodes(0, POOL, STEPS, 0.0, RAY, seed0=20000, params=PARAMS)
surr = {d: collect_episodes(1, POOL, STEPS, d, RAY, seed0=30000 + int(d * 1000), params=PARAMS)
        for d in DRIFTS if d > 0}

def per_episode_surprise(model, norm, eps):
    """Per-episode next-step prediction error (the agent's 'surprise'), summarized."""
    mu, sd = norm
    obs, act = _stack(eps)
    with torch.no_grad():
        pred, _ = model((torch.tensor(obs) - mu) / sd, torch.tensor(act))
    tgt = ((torch.tensor(obs) - mu) / sd)[:, 1:]
    err = (pred[:, :-1] - tgt).abs().mean(1).numpy()  # (B, obs_dim) mean over time
    return np.stack([err.mean(1), err[:, CDIMS].mean(1), np.percentile(err, 90, axis=1)], axis=1)

def subsample(pool, k, rng):
    return [pool[i] for i in rng.choice(len(pool), k, replace=False)]

authtest, ctrl_other = authB[:75], authB[75:150]  # held-out authentic; disjoint authentic for the d=0 control
res = {d: [] for d in DRIFTS}
for seed in SEEDS:
    rng = np.random.default_rng(2000 + seed); np.random.seed(seed)
    model, norm = train_world_model(subsample(authA, SUB, rng), epochs=EPOCHS, hidden=HIDDEN, seed=seed)  # AUTHENTIC-ONLY
    Sa = per_episode_surprise(model, norm, authtest)
    for d in DRIFTS:
        other = ctrl_other if d == 0 else surr[d][:75]
        So = per_episode_surprise(model, norm, other)
        X = np.vstack([Sa, So]); y = np.array([0] * len(Sa) + [1] * len(So))
        res[d].append(grouped_auroc(X, y, np.arange(len(y))))

print("Prediction-error (surprise) probe: authentic-only world model, tested authentic vs surrogate")
for d in DRIFTS:
    print(f"  drift={d:.2f}   surprise-probe AUROC = {np.mean(res[d]):.3f} ± {np.std(res[d]):.3f}")

# combined comparison figure (hidden-state numbers from the prior full run, same pipeline/config)
hs = {0.0: 0.460, 0.2: 0.509, 0.45: 0.510}
sp = [np.mean(res[d]) for d in DRIFTS]; spe = [np.std(res[d]) for d in DRIFTS]
plt.figure(figsize=(7.2, 4.4))
plt.errorbar(DRIFTS, sp, yerr=spe, fmt="o-", color="#c05621", capsize=4, lw=2, label="prediction-error (surprise) probe")
plt.plot(DRIFTS, [hs[d] for d in DRIFTS], "s--", color="#6b46c1", label="recurrent-state probe")
plt.axhline(0.5, ls=":", color="grey")
plt.xlabel("surrogate strength  (L2 drift_sigma)"); plt.ylabel("probe AUROC  (per episode)")
plt.title("ITASORL Experiment B - two incidental-detection channels")
plt.ylim(0.3, 1.02); plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
plt.savefig("docs/figures/expB_channels.png", dpi=130)
print("\nsaved docs/figures/expB_channels.png")
