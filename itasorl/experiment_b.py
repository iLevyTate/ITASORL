"""
ITASORL - Experiment B: incidental detection (H1 / H4) with selectivity controls.

Pipeline:
  1. collect episodes in the authentic world and a surrogate world, with a fixed
     stochastic policy. The world label is recorded but NEVER fed to the agent.
  2. train the recurrent world model on next-step prediction over the pooled
     episodes (no identity supervision, no identity reward).
  3. freeze it, read out the recurrent states, and run THREE probes:
       target   - decode world identity                     (the H4 claim)
       shuffled - decode a randomized world label           (negative control)
       speed    - decode above/below-median speed           (positive control)
     A real effect is: target >> shuffled, with speed high (states are probeable).

The probe is a frozen linear model; episodes are the unit of inference. This
turn runs a SMOKE TEST (tiny data / few epochs) to lock the apparatus; the full
run scales data, training, seeds, and surrogate levels.
"""

from __future__ import annotations

import numpy as np

from .agent import TORCH, Reservoir
from .experiment_a import grouped_auroc
from .patch_of_earth import PatchOfEarthV0
from .world import SeedBundle, WorldParams


def scripted_policy(rng) -> np.ndarray:
    return np.array([rng.uniform(0.0, 0.6), rng.uniform(-1.0, 1.0),
                     float(rng.random() < 0.3), 0.0, 0.0], dtype=np.float32)


def collect_episodes(label: int, n: int, steps: int = 30, drift_sigma: float = 0.0,
                     ray_steps: int = 6, seed0: int = 0, params: WorldParams | None = None) -> list[dict]:
    params = params or WorldParams()
    eps = []
    for i in range(n):
        rng = np.random.default_rng(seed0 + i)
        w = PatchOfEarthV0(params, drift_sigma=drift_sigma)
        w.ray_steps = ray_steps
        w.reset(SeedBundle(world=seed0 + i, weather=seed0 + 7000 + i, ecology=seed0 + 13000 + i))
        O, A, spd = [], [], []
        w.observe()
        for _ in range(steps):
            a = scripted_policy(rng)
            r = w.step(a)
            O.append(r.obs.copy()); A.append(a); spd.append(float(np.linalg.norm(w.vel)))
        eps.append({"obs": np.asarray(O, np.float32), "act": np.asarray(A, np.float32),
                    "label": label, "speed": float(np.mean(spd))})
    return eps


def _stack(eps):
    return np.stack([e["obs"] for e in eps]), np.stack([e["act"] for e in eps])


def train_world_model(eps, epochs: int = 8, lr: float = 1e-3, embed: int = 64,
                      hidden: int = 128, batch: int = 16, seed: int = 0,
                      rollout_context: int | None = None, delta: bool = False):
    """Train the recurrent world model on next-step prediction.

    rollout_context: if set, train the open-loop objective instead - teacher-force
    the first `rollout_context` steps, then imagine forward and score the predicted
    future (the k-step / delta-rollout experiments). delta: have the model predict
    the observation change rather than the absolute next observation.
    """
    import torch
    from .agent import RecurrentWorldModel
    torch.manual_seed(seed)
    obs_np, act_np = _stack(eps)
    obs, act = torch.tensor(obs_np), torch.tensor(act_np)
    mu = obs.mean((0, 1), keepdim=True)
    sd = obs.std((0, 1), keepdim=True) + 1e-6
    obsn = (obs - mu) / sd
    model = RecurrentWorldModel(obs.shape[-1], act.shape[-1], embed, hidden)
    model.delta = bool(delta)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    idx = np.arange(len(eps))
    # Own keyed stream, NOT the global numpy RNG: minibatch order must be a function
    # of `seed` alone or run_experiment_b(seed=...) is not reproducible.
    shuffle_rng = np.random.default_rng(seed)
    for _ in range(epochs):
        shuffle_rng.shuffle(idx)
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            if rollout_context is None:
                loss, _ = model.prediction_loss(obsn[b], act[b])
            else:
                loss = model.rollout_loss(obsn[b], act[b], rollout_context)
            opt.zero_grad(); loss.backward(); opt.step()
    return model, (mu, sd)


def states_torch(model, eps, norm) -> np.ndarray:
    import torch
    mu, sd = norm
    obs_np, act_np = _stack(eps)
    with torch.no_grad():
        _, H = model((torch.tensor(obs_np) - mu) / sd, torch.tensor(act_np))
    return H.numpy()


def states_reservoir(eps, hidden: int = 128, seed: int = 0) -> np.ndarray:
    obs_np, act_np = _stack(eps)
    mu, sd = obs_np.mean((0, 1), keepdims=True), obs_np.std((0, 1), keepdims=True) + 1e-6
    res = Reservoir(obs_np.shape[-1], act_np.shape[-1], hidden=hidden, seed=seed)
    return res.states((obs_np - mu) / sd, act_np)


def episode_features(H: np.ndarray) -> np.ndarray:
    """Per-episode probe input: mean and final recurrent state (LEVEL features)."""
    return np.concatenate([H.mean(1), H[:, -1]], axis=-1)


def episode_features_var(H: np.ndarray) -> np.ndarray:
    """Per-episode DISPERSION features: [std over steps, mean |step-to-step delta|].

    The authentic and surrogate worlds share a drag LEVEL but differ in drag
    VOLATILITY (authentic drag is constant; the surrogate's drifts via an AR(1)
    wander). A persistent linear direction over the level features `episode_features`
    cannot express a volatility signature, so this builder exposes it: if a reactive
    agent tracks a fluctuating drag its recurrent state swings more per episode in the
    surrogate. H is (n_eps, steps, hidden)."""
    std_h = H.std(axis=1)
    if H.shape[1] >= 2:
        jerk = np.abs(np.diff(H, axis=1)).mean(axis=1)
    else:
        jerk = np.zeros_like(std_h)
    return np.concatenate([std_h, jerk], axis=-1)


def episode_features_full(H: np.ndarray) -> np.ndarray:
    """LEVEL ++ DISPERSION features: episode_features concatenated with
    episode_features_var. The most permissive readout the pre-registered pipeline
    admits without changing the estimator."""
    return np.concatenate([episode_features(H), episode_features_var(H)], axis=-1)


def probe_auroc(X: np.ndarray, y: np.ndarray) -> float:
    return grouped_auroc(X, y, groups=np.arange(len(y)))  # episodes independent


def run_experiment_b(n: int = 30, steps: int = 30, drift_sigma: float = 0.4, epochs: int = 8,
                     hidden: int = 128, params: WorldParams | None = None, seed: int = 0,
                     backend: str | None = None) -> dict:
    backend = backend or ("torch" if TORCH else "reservoir")
    auth = collect_episodes(0, n, steps, 0.0, seed0=1000, params=params)
    surr = collect_episodes(1, n, steps, drift_sigma, seed0=5000, params=params)
    eps = auth + surr

    if backend == "torch":
        model, norm = train_world_model(eps, epochs=epochs, hidden=hidden, seed=seed)
        H = states_torch(model, eps, norm)
    else:
        H = states_reservoir(eps, hidden=hidden, seed=seed)

    X = episode_features(H)
    y = np.array([e["label"] for e in eps])
    spd = np.array([e["speed"] for e in eps])
    rng = np.random.default_rng(seed)
    return {
        "backend": backend,
        "target": probe_auroc(X, y),                                  # H4: decode world identity
        "shuffled": probe_auroc(X, rng.permutation(y)),               # negative control
        "speed_positive_ctrl": probe_auroc(X, (spd > np.median(spd)).astype(int)),  # positive control
    }


if __name__ == "__main__":
    r = run_experiment_b(n=25, steps=25, drift_sigma=0.5, epochs=6, hidden=96)
    print("Experiment B smoke test  (backend = %s)" % r["backend"])
    print(f"   target  (decode world identity) AUROC = {r['target']:.3f}")
    print(f"   shuffled (negative control)     AUROC = {r['shuffled']:.3f}")
    print(f"   speed   (positive control)      AUROC = {r['speed_positive_ctrl']:.3f}")
