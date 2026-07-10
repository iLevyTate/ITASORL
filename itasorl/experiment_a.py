"""
ITASORL - Experiment A: the agent-free detectability ceiling.

An external discriminator on trajectory logs, run BEFORE any agent is trained,
to (a) measure how detectable each surrogate level is (the ceiling / difficulty
calibration) and (b) prove the signal is not riding on a confound via a
leakage-audit battery that must FAIL to separate the worlds.

Realism note: a noiseless world makes any discretization trivially detectable
(values land exactly on the grid). Real senses are noisy, so a fixed sensor
noise sigma is applied; detection of an L1 grid of spacing delta then becomes a
function of delta/sigma - exactly the ceiling we want to calibrate.

The unit of inference is the matched PAIR, so all AUROCs use GroupKFold over
pair id (never the timestep). Feature extractors map onto the logschema columns,
so the same oracle runs on real Parquet logs.
"""

from __future__ import annotations

import numpy as np

from .patch_of_earth import PatchOfEarthV0, first_config_obs_spec
from .world import SeedBundle

CONST_POLICY = np.array([0.5, 0.15, 1.0, 0.0, 0.0], dtype=np.float32)  # fixed, obs-independent


# ---------------------------------------------------------------------------
# Dimension bookkeeping (only continuous channels carry a quantization signal)
# ---------------------------------------------------------------------------

def continuous_dims(spec=None) -> np.ndarray:
    """Indices of continuous obs dimensions: per-ray (distance, radial velocity)
    plus all interoception. Reflectance is a small categorical palette and the
    masked smell channel is noise-only, so both are excluded."""
    spec = spec or first_config_obs_spec()
    sl = spec.slices()
    v = sl["vision"]
    n_ray = (v.stop - v.start) // 5
    dims = []
    for r in range(n_ray):
        base = v.start + r * 5
        dims.append(base + 0)  # distance
        dims.append(base + 4)  # radial velocity
    i = sl["intero"]
    dims += list(range(i.start, i.stop))
    return np.array(dims, dtype=int)


# ---------------------------------------------------------------------------
# Feature extractors  (operate on a trajectory: obs[T, D], rewards[T], t[T])
# ---------------------------------------------------------------------------

def oracle_features_L1(obs_TD: np.ndarray, dims: np.ndarray, delta: float) -> np.ndarray:
    """Quantization signal: per-dimension residual std against the delta grid.
    A gridded (surrogate) dimension leaves residual ~= sensor noise sigma; a
    continuous (authentic) dimension leaves residual ~= delta/sqrt(12). The gap
    is the tell, and it vanishes as delta -> sigma (the ceiling falls to chance).
    """
    X = obs_TD[:, dims]
    resid = X - np.round(X / delta) * delta
    s = np.std(resid, axis=0)
    return np.array([s.mean(), np.median(s), np.percentile(s, 25)])


def oracle_features_L2(extras: dict[str, np.ndarray]) -> np.ndarray:
    """Invariant-violation signal for rollout drift (chaos-robust). Operates on
    the logged invariant columns; exercised once L2 (and logging) are wired.
        energy_budget_residual ~ 0 in the authentic world, drifts in a surrogate
        kinetic_energy decay rate is steady authentically, wanders under drift
        scent_total_mass is conserved authentically
    """
    ebr = np.asarray(extras["energy_budget_residual"])
    ke = np.asarray(extras["kinetic_energy"])
    sm = np.asarray(extras["scent_total_mass"])
    return np.array([
        np.mean(np.abs(ebr)), np.std(ebr),
        np.std(np.diff(ke)),
        np.std(np.diff(sm)),
    ])


def leak_reward(rewards: np.ndarray) -> list[float]:
    return [float(rewards.mean()), float(rewards.std()), float(rewards.sum())]


def leak_length(rewards: np.ndarray) -> list[float]:
    return [float(len(rewards))]


def leak_meta(t: np.ndarray) -> list[float]:
    return [float(t[0]), float(t[-1]), float(len(t))]


# ---------------------------------------------------------------------------
# Data generation (fixed policy -> obs-level L1 leaves dynamics untouched, so
# the clean trajectory is shared and (de)quantization is pure post-processing)
# ---------------------------------------------------------------------------

def generate_clean(n_pairs: int, steps: int, seed0: int = 1000, ray_steps: int = 20) -> list[dict]:
    data = []
    for i in range(n_pairs):
        w = PatchOfEarthV0()
        w.ray_steps = ray_steps  # lighter raymarch for the sweep
        w.reset(SeedBundle(world=seed0 + i, weather=seed0 + 5000 + i, ecology=seed0 + 9000 + i))
        obs, rew, ts = [], [], []
        for _ in range(steps):
            r = w.step(CONST_POLICY)
            obs.append(r.obs.copy())
            rew.append(r.reward)
            ts.append(r.info["t"])
        data.append({"pair": i, "obs": np.asarray(obs), "rew": np.asarray(rew), "t": np.asarray(ts)})
    return data


def _emit(clean: np.ndarray, quantize: bool, delta: float, sigma: float, rng) -> np.ndarray:
    sig = np.round(clean / delta) * delta if quantize else clean
    return sig + rng.normal(0.0, sigma, size=clean.shape)


# ---------------------------------------------------------------------------
# Grouped AUROC (resample at the pair level, never the timestep)
# ---------------------------------------------------------------------------

def grouped_auroc(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int = 5,
                  return_oof: bool = False):
    """Grouped 5-fold CV AUROC with the scaler fit on TRAIN folds only (no leakage).
    With return_oof=True also returns the concatenated out-of-fold (y_true, y_score),
    which lets callers bootstrap an AUROC CI without refitting the probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    gkf = GroupKFold(n_splits=n_splits)
    aucs = []
    oof_y, oof_p = [], []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        if return_oof:
            oof_y.append(np.asarray(y[te]))
            oof_p.append(np.asarray(p))
    auc = float(np.mean(aucs)) if aucs else float("nan")  # NaN, not 0.5: never report the null as measured
    if return_oof:
        yv = np.concatenate(oof_y) if oof_y else np.zeros(0)
        pv = np.concatenate(oof_p) if oof_p else np.zeros(0)
        return auc, yv, pv
    return auc


# ---------------------------------------------------------------------------
# The experiment
# ---------------------------------------------------------------------------

def run_experiment_a(clean: list[dict], level: str, delta: float, sigma: float,
                     contaminate: float = 0.0, seed: int = 0, leak_tol: float = 0.1) -> dict:
    """level: 'L0' (control) or 'L1'. contaminate: a reward offset added to the
    surrogate to simulate a buggy generator - the leakage gate should catch it."""
    rng = np.random.default_rng(seed)
    dims = continuous_dims()
    Xo, y, g, Xrew, Xlen, Xmeta = [], [], [], [], [], []
    for d in clean:
        for label in (0, 1):  # 0 = authentic, 1 = surrogate
            quant = (label == 1 and level == "L1")
            obs = _emit(d["obs"], quant, delta, sigma, rng)
            Xo.append(oracle_features_L1(obs, dims, delta))
            rew = d["rew"].copy()
            if label == 1:
                rew = rew + contaminate
            Xrew.append(leak_reward(rew))
            Xlen.append(leak_length(rew))
            Xmeta.append(leak_meta(d["t"]))
            y.append(label)
            g.append(d["pair"])
    y = np.asarray(y); g = np.asarray(g)
    leaks = {
        "reward": grouped_auroc(np.asarray(Xrew), y, g),
        "length": grouped_auroc(np.asarray(Xlen), y, g),
        "metadata": grouped_auroc(np.asarray(Xmeta), y, g),
    }
    return {
        "level": level, "delta": delta, "sigma": sigma, "contaminate": contaminate,
        "oracle_auroc": grouped_auroc(np.asarray(Xo), y, g),
        "leakage": leaks,
        "leakage_pass": all(abs(a - 0.5) < leak_tol for a in leaks.values()),
    }
