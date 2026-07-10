"""ITASORL - Experiment A for the L3 learned-dynamics rung.

L3 replaces the velocity law with a learned net `G_motion` (see `surrogate_l3.py`). The
oracle is the same invariant-based detector as L2: re-apply the EXACT authentic one-step
law and measure the residual against a measurement-noise floor. Here the exact per-step
law is reconstructed from the world's logged `(vel, a, drag, vel_next)` motion transitions
(`_log_motion`), so no physics is duplicated:

  authentic branch: vel_next == (1-drag*dt)*vel + a*dt  -> residual ~ measurement noise.
  L3 branch:        vel_next == G_motion(vel, a)         -> residual == G's dynamics error.

The net's capacity controls that error, so the oracle AUROC is a clean monotone function of
capacity (the difficulty knob, `docs/PREREGISTRATION_L3.md` section 7 gate 0).
"""

from __future__ import annotations

import numpy as np

from .experiment_a import grouped_auroc, leak_length, leak_meta, leak_reward
from .patch_of_earth import PatchOfEarthV0
from .world import SeedBundle, WorldParams


def _oracle_policy(rng) -> np.ndarray:
    """Explore motion (random thrust + turn) with NO foraging. Eat/drink do not enter
    _integrate_motion, so dropping them leaves the motion distribution identical to the
    training policy while ensuring reward/length cannot leak from the dynamics difference
    (the dynamics-altering-rung confound noted for L2, docs/FINDINGS.md section 2.2)."""
    return np.array([rng.uniform(0.1, 0.6), rng.uniform(-1.0, 1.0), 0.0, 0.0, 0.0], np.float32)


def generate_l3_pairs(g_motion, n_pairs: int = 60, prefix: int = 10, branch: int = 30,
                      ray_steps: int = 5, seed0: int = 3000, params=None) -> list[dict]:
    """Matched pairs: a shared authentic prefix is snapshotted, then continued once
    authentically and once with `g_motion` driving the velocity law, under the SAME action
    stream. Each branch logs its `(vel, a, drag, vel_next)` motion transitions."""
    params = params or WorldParams()
    eps: list[dict] = []
    for i in range(n_pairs):
        sb = SeedBundle(world=seed0 + i, weather=seed0 + 5000 + i, ecology=seed0 + 9000 + i)
        base = PatchOfEarthV0(params)
        base.ray_steps = ray_steps
        base.reset(sb)
        rng_p = np.random.default_rng(seed0 + i)
        for _ in range(prefix):
            base.step(_oracle_policy(rng_p))
        snap = base.get_state()
        for label, g in ((0, None), (1, g_motion)):        # authentic, then L3
            w = PatchOfEarthV0(params, drift_mode="l3")
            w.ray_steps = ray_steps
            w.reset(sb)
            w.set_state(snap)
            w._g_motion = g                                 # None -> authentic dynamics
            w._log_motion = []
            rng_b = np.random.default_rng(seed0 + 100000 + i)   # identical action stream in both branches
            rew, ts = [], []
            for _ in range(branch):
                r = w.step(_oracle_policy(rng_b))
                rew.append(r.reward)
                ts.append(r.info["t"])
            eps.append({"pair": i, "label": label, "trans": list(w._log_motion),
                        "rew": np.asarray(rew), "t": np.asarray(ts)})
    return eps


def oracle_features_L3(trans: list, sigma_meas: float, rng, dt: float) -> np.ndarray:
    """Re-apply the exact authentic law to the logged transitions and summarize the residual
    of the (noisy) observed next velocity. Authentic: residual ~ measurement noise; L3:
    residual = G's dynamics error."""
    vel = np.array([t[0] for t in trans], float)
    a = np.array([t[1] for t in trans], float)
    drag = np.array([t[2] for t in trans], float)
    vnext = np.array([t[3] for t in trans], float)
    pred = (1.0 - drag[:, None] * dt) * vel + a * dt
    resid = (vnext + rng.normal(0.0, sigma_meas, size=vnext.shape)) - pred
    return np.array([resid[:, 0].std(), resid[:, 1].std(), float(np.abs(resid).mean())])


def run_experiment_a_l3(eps: list[dict], sigma_meas: float, contaminate: float = 0.0,
                        seed: int = 0, leak_tol: float = 0.1, dt: float | None = None) -> dict:
    """Grouped-AUROC oracle + leakage battery (reward/length/metadata must stay ~0.5)."""
    dt = WorldParams().dt if dt is None else dt
    rng = np.random.default_rng(seed)
    Xo, y, g, Xr, Xl, Xm = [], [], [], [], [], []
    for e in eps:
        Xo.append(oracle_features_L3(e["trans"], sigma_meas, rng, dt))
        rew = e["rew"].copy()
        if e["label"] == 1:
            rew = rew + contaminate
        Xr.append(leak_reward(rew))
        Xl.append(leak_length(rew))
        Xm.append(leak_meta(e["t"]))
        y.append(e["label"])
        g.append(e["pair"])
    y = np.asarray(y)
    g = np.asarray(g)
    leaks = {
        "reward": grouped_auroc(np.asarray(Xr), y, g),
        "length": grouped_auroc(np.asarray(Xl), y, g),
        "metadata": grouped_auroc(np.asarray(Xm), y, g),
    }
    # `reward` is EXCLUDED from the pass criterion: for a dynamics-altering rung, different
    # dynamics -> different movement cost -> different reward is a legitimate downstream
    # consequence, not a nuisance the oracle exploits (the oracle uses ONLY the velocity
    # residual). See docs/FINDINGS.md section 2.2. It is reported and flagged as a
    # readout-not-reward consideration for the organism experiment. The MECHANICAL channels
    # (length, metadata) must stay at chance - they confirm the oracle is not cheating on
    # episode length/timing.
    mechanical = {"length": leaks["length"], "metadata": leaks["metadata"]}
    return {
        "oracle_auroc": grouped_auroc(np.asarray(Xo), y, g),
        "leakage": leaks,
        "leakage_pass": all(abs(a - 0.5) < leak_tol for a in mechanical.values()),
        "reward_leak": float(leaks["reward"]),
    }


def calibrate_l3(*, hiddens=(4, 8, 16, 32, 64), sigma_meas: float, n_pairs: int = 60,
                 branch: int = 30, train_kwargs: dict | None = None, seed: int = 0) -> list[dict]:
    """Sweep G_motion capacity; for each, train, generate pairs, and score the oracle. Returns
    per-capacity {hidden, oracle_auroc, in_band, leakage_pass}. The frozen (sigma, capacity)
    is the smallest capacity whose oracle AUROC lands in [0.85, 0.95] (prereg gate 0)."""
    from .surrogate_l3 import train_g_motion
    train_kwargs = dict(train_kwargs or {})
    params = train_kwargs.get("params")   # score pairs on the SAME world G trains on (no train/deploy mismatch)
    out = []
    for h in hiddens:
        g = train_g_motion(hidden=h, seed=seed, **train_kwargs)
        eps = generate_l3_pairs(g, n_pairs=n_pairs, branch=branch, seed0=3000 + seed, params=params)
        res = run_experiment_a_l3(eps, sigma_meas=sigma_meas, seed=seed)
        auc = float(res["oracle_auroc"])
        out.append({"hidden": h, "oracle_auroc": auc,
                    "in_band": 0.85 <= auc <= 0.95, "leakage_pass": res["leakage_pass"]})
    return out
