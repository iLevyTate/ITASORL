# L3 Cross-Recipe Transfer Probe Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the pre-registered cross-recipe transfer probe (spec `docs/superpowers/specs/2026-07-15-l3-crossrecipe-transfer-probe-design.md`): two new held-out surrogate families (`G_rff` primary, `G_cd` secondary), per-family gate-0 calibration, and a readout-only runner that reuses the saved `fullruns/l3_h8_heldout` agents behind a bit-identity integrity gate.

**Architecture:** New module `itasorl/surrogate_l3_families.py` holds both families as `(vel, a, drag) -> vel_next` callables (numpy-only per step, same hook contract as `GMotion`). `scripts/run_expA_l3.py` gains a `--family` flag whose default preserves current behavior byte-for-byte. A new readout-only script `scripts/run_l3_crossrecipe.py` loads saved agent bundles, verifies bit-identical pool regeneration, then scores the frozen probe on fresh pools against each gate-passing family via a minimally generalized `transfer_readout`. The training path in `run_expB2.py` is untouched.

**Tech Stack:** Python 3.10-3.12, numpy, torch (agent forward only, never per world step), pytest, ruff.

**Branch:** `feat/l3-crossrecipe-transfer` (spec already committed as `bcb73f1`).

---

## Reference: existing code facts (verified 2026-07-15)

The engineer should trust these; they were read from the working tree.

- `itasorl/surrogate_l3.py:21` `collect_authentic_transitions(*, n_eps=250, steps=40, params=None, ray_steps=5, seed0=0)` returns `X (N,4) = [vel_x, vel_y, a_x, a_y]`, `Y (N,2) = vel_next`, both float32.
- `itasorl/surrogate_l3.py:44` `GMotion` is the hook-contract template: callable `(vel, a, drag=None) -> np.ndarray` shape (2,), dtype float, numpy-only.
- `itasorl/patch_of_earth.py:174-177`: in `l3` mode the world calls `self._g_motion(vel_before, a, drag)`; the authentic branch is `vel_next = (1.0 - drag * self.params.dt) * vel_before + a * self.params.dt`.
- World P: `WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)` (`scripts/run_expA_l3.py:50`); drag is constant 1.5 in P, so the authentic law is exactly linear (spec scope note).
- `itasorl/experiment_b2.py:125` `RunningNorm(dim)` with attrs `mean`, `var`, `count`.
- `itasorl/agent_ac.py:38` `RecurrentActorCritic(obs_dim, act_dim, embed=64, hidden=128, world_model=True, sysid_aux=False)`.
- `itasorl/experiment_b2.py:928` `save_agent_bundle(path, agent, norm)` saves `{"state_dict", "ctor": {obs_dim, act_dim, embed, hidden, world_model, sysid_aux}, "norm": {mean, var, count}}`. A loader `load_agent_bundle` already exists directly below it (returns the agent in inference mode and the norm frozen); Task 3 only adds the roundtrip test.
- `itasorl/experiment_b2.py:713` `pooled_readout(agent, norm, params, drift_sigma, *, n_eps=110, steps=24, ray_steps=5, device=None, seed=0, dump_path=None, leak_margin=0.1, return_pools=False)`; standard pool seed bases are `800_000` (auth) and `850_000` (surrogate); with `return_pools=True` returns `(out, (Ha, Hs))`.
- Saved dumps `fullruns/l3_h8_heldout/states/states_d{drift:.2f}_s{seed}_{arm}.npz` contain keys `Ha`, `Hs` (plus anchors); saved agents `fullruns/l3_h8_heldout/agents/agent_d{drift:.2f}_s{seed}_{arm}.pt`, 60 files, arms `untrained|predictor|survival`, drifts `0.00|0.45`, seeds 0-9.
- `itasorl/experiment_b2.py:809` `transfer_readout(agent, norm, params, drift_sigma, Ha_train, Hs_train, *, n_eps=110, steps=24, ray_steps=5, device=None, seed=0, dump_path=None)`; uses module global `_L3_GMOTION_HELDOUT`, seed bases `860_000` (fresh auth) / `870_000` (held-out pool); returns keys `transfer_target`, `transfer_lo`, `transfer_hi`, `transfer_n_auth`, `transfer_n_surr`, `transfer_deaths_auth`, `transfer_deaths_surr`.
- `itasorl/experiment_b2.py:530` `transfer_probe(Xtr, ytr, Xte, yte, return_scores=False)`; `itasorl/experiment_b.py:109` `episode_features(H)` = `[mean over steps ++ final step]`.
- `itasorl/experiment_b2.py:78` `setup_l3_surrogate(**train_kwargs)` installs the module global `_L3_GMOTION`; `make_world` attaches it when `DRIFT_MODE == "l3"` and `drift_sigma > 0`.
- `scripts/run_expA_l3.py` frozen constants: `SIGMA_MEAS = 0.02`, `BAND = (0.85, 0.95)`, `FLOOR_TOL = 0.10`, `DRIFT = 0.45`; per candidate it runs `generate_l3_pairs(g, n_pairs, branch, seed0=3000, params=P)` then `run_experiment_a_l3(eps, sigma_meas=SIGMA_MEAS, seed=0)`, then the untrained floor via `b2._L3_GMOTION = g` + `untrained_agent(P, DRIFT, ray_steps=5, hidden=96, embed=64, world_model=True, device=dev, seed=s)` + `pooled_readout(...)` for seeds `[0, 1, 2]`.
- Seed bases already used by the original run (must NOT be reused for new pools): 800_000, 850_000, 860_000, 870_000, 930_000, 555_000. New bases frozen by this plan: RFF fresh-auth `880_000`, RFF surrogate `890_000`, CD fresh-auth `900_000`, CD surrogate `910_000`.
- Torch note: `module.train(False)` is exactly equivalent to torch eval mode;
  this plan uses `train(False)` throughout.

---

### Task 1: `GConstantDrag` family

**Files:**
- Create: `itasorl/surrogate_l3_families.py`
- Create: `tests/test_surrogate_l3_families.py`

- [ ] **Step 1.1: Write the failing tests**

Create `tests/test_surrogate_l3_families.py`:

```python
"""Tests for the cross-recipe held-out families (spec 2026-07-15)."""

import numpy as np

from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)  # the frozen organism world


def _authentic_law(vel, a, drag, dt):
    return (1.0 - drag * dt) * np.asarray(vel, float) + np.asarray(a, float) * dt


def test_g_cd_eps_zero_is_authentic_law():
    from itasorl.surrogate_l3_families import make_g_cd
    g = make_g_cd(eps=0.0, params=P)
    rng = np.random.default_rng(0)
    for _ in range(50):
        vel = rng.normal(size=2)
        a = rng.normal(size=2)
        want = _authentic_law(vel, a, 1.5, P.dt)
        got = g(vel, a, drag=None)  # drag ignored by contract
        assert np.array_equal(got, want)


def test_g_cd_eps_nonzero_differs_and_is_biased_decay():
    from itasorl.surrogate_l3_families import make_g_cd
    g = make_g_cd(eps=0.2, params=P)
    vel, a = np.array([0.3, -0.4]), np.array([0.0, 0.0])
    want = _authentic_law(vel, a, 1.5 * 1.2, P.dt)
    got = g(vel, a)
    assert np.array_equal(got, want)
    assert not np.array_equal(got, _authentic_law(vel, a, 1.5, P.dt))


def test_g_cd_requires_uniform_drag_world():
    import pytest
    from itasorl.surrogate_l3_families import make_g_cd
    with pytest.raises(ValueError):
        make_g_cd(eps=0.1, params=WorldParams())  # defaults: k_land=0.20 != k_water=0.60
```

- [ ] **Step 1.2: Run tests to verify they fail**

Run: `python -m pytest tests/test_surrogate_l3_families.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'itasorl.surrogate_l3_families'`

- [ ] **Step 1.3: Write the minimal implementation**

Create `itasorl/surrogate_l3_families.py`:

```python
"""Cross-recipe held-out surrogate families (spec 2026-07-15).

Both families satisfy the GMotion hook contract: callable `(vel, a, drag) ->
vel_next`, drag ignored, numpy-only per world step. They are EVALUATION-ONLY
transfer targets; the training surrogate stays the frozen GMotion MLP.

World-P scope (spec): drag is constant in the frozen organism world, so the
authentic velocity law is exactly linear in (vel, a). A family therefore only
has a fingerprint if it CANNOT represent that linear map (G_rff: cosine basis)
or is deliberately mis-set (G_cd: wrong drag constant).
"""

from __future__ import annotations

import numpy as np


class GConstantDrag:
    """Analytic constant-drag law with a deliberately mis-set constant.
    Degenerate L2-regime by construction; pre-registered as the SECONDARY
    cross-rung channel, never part of the primary decision."""

    def __init__(self, c: float, dt: float) -> None:
        self._c, self._dt = float(c), float(dt)

    def __call__(self, vel, a, drag=None) -> np.ndarray:
        return ((1.0 - self._c * self._dt) * np.asarray(vel, float)
                + np.asarray(a, float) * self._dt)


def make_g_cd(*, eps: float, params) -> GConstantDrag:
    """c = drag0 * (1 + eps) where drag0 is world-P's uniform drag. Refuses
    non-uniform-drag worlds: there the law would need wetness, which the hook
    deliberately cannot see, and the eps=0 identity check would be ill-defined."""
    if params.k_land != params.k_water:
        raise ValueError("make_g_cd requires a uniform-drag world (k_land == k_water)")
    return GConstantDrag(c=params.k_land * (1.0 + eps), dt=params.dt)
```

- [ ] **Step 1.4: Run tests to verify they pass**

Run: `python -m pytest tests/test_surrogate_l3_families.py -v`
Expected: 3 PASS

- [ ] **Step 1.5: Commit**

```bash
git add itasorl/surrogate_l3_families.py tests/test_surrogate_l3_families.py
git commit -m "feat(l3): GConstantDrag cross-recipe family (secondary channel)"
```

---

### Task 2: `GRff` family

**Files:**
- Modify: `itasorl/surrogate_l3_families.py`
- Modify: `tests/test_surrogate_l3_families.py`

- [ ] **Step 2.1: Write the failing tests**

Append to `tests/test_surrogate_l3_families.py`:

```python
def test_g_rff_deterministic_across_fits():
    from itasorl.surrogate_l3_families import fit_g_rff
    kw = dict(D=16, params=P, n_eps=4, steps=10)  # tiny data: determinism only
    g1, g2 = fit_g_rff(**kw), fit_g_rff(**kw)
    vel, a = np.array([0.2, -0.1]), np.array([0.5, 0.3])
    assert np.array_equal(g1(vel, a), g2(vel, a))


def test_g_rff_cannot_fit_the_linear_law_exactly():
    """World P's authentic law is exactly linear; the cosine basis must leave a
    systematic residual (this is the fingerprint's existence proof)."""
    from itasorl.surrogate_l3_families import fit_g_rff
    g = fit_g_rff(D=16, params=P, n_eps=4, steps=10)
    rng = np.random.default_rng(1)
    errs = []
    for _ in range(100):
        vel, a = rng.normal(size=2), rng.normal(size=2)
        errs.append(np.abs(g(vel, a) - _authentic_law(vel, a, 1.5, P.dt)).max())
    assert max(errs) > 1e-8


def test_g_rff_output_contract():
    from itasorl.surrogate_l3_families import fit_g_rff
    g = fit_g_rff(D=16, params=P, n_eps=4, steps=10)
    out = g(np.array([0.1, 0.2]), np.array([0.3, 0.4]), drag=1.5)  # drag ignored
    assert out.shape == (2,) and out.dtype == np.float64
```

- [ ] **Step 2.2: Run tests to verify they fail**

Run: `python -m pytest tests/test_surrogate_l3_families.py -v -k rff`
Expected: FAIL with `ImportError: cannot import name 'fit_g_rff'`

- [ ] **Step 2.3: Write the implementation**

Append to `itasorl/surrogate_l3_families.py`:

```python
class GRff:
    """Random-Fourier-features ridge velocity law: z(x) = sqrt(2/D) cos(Wx + b)
    on normalized inputs, closed-form ridge readout. Smooth global sinusoidal
    basis + convex fit = a different recipe from the ReLU-MLP GMotion; the
    PRIMARY cross-recipe transfer target."""

    def __init__(self, W, b, Wout, norm, D) -> None:
        self._W = W.astype(np.float32)          # (D, 4)
        self._b = b.astype(np.float32)          # (D,)
        self._Wout = Wout.astype(np.float32)    # (D, 2)
        self._xm, self._xs, self._ym, self._ys = norm
        self._scale = np.float32(np.sqrt(2.0 / D))

    def __call__(self, vel, a, drag=None) -> np.ndarray:
        x = (np.array([vel[0], vel[1], a[0], a[1]], np.float32) - self._xm) / self._xs
        z = self._scale * np.cos(self._W @ x + self._b)
        return (z @ self._Wout * self._ys + self._ym).astype(float)


def fit_g_rff(*, D: int = 32, lam: float = 1e-3, ell: float = 1.0,
              feature_seed: int = 0, n_eps: int = 250, steps: int = 40,
              params=None, ray_steps: int = 5, seed: int = 0) -> GRff:
    """Fit on the same authentic-transition data budget as train_g_motion.
    Frozen defaults per spec: lam=1e-3, ell=1.0 on normalized inputs,
    feature_seed=0. Difficulty knob: D (feature count)."""
    from .surrogate_l3 import collect_authentic_transitions
    X, Y = collect_authentic_transitions(n_eps=n_eps, steps=steps, params=params,
                                         ray_steps=ray_steps, seed0=seed)
    xm, xs = X.mean(0), X.std(0) + 1e-6
    ym, ys = Y.mean(0), Y.std(0) + 1e-6
    rng = np.random.default_rng(feature_seed)
    W = rng.normal(0.0, 1.0 / ell, size=(D, 4)).astype(np.float32)
    b = rng.uniform(0.0, 2.0 * np.pi, size=D).astype(np.float32)
    Z = np.sqrt(2.0 / D) * np.cos(((X - xm) / xs) @ W.T + b)
    A = Z.T @ Z + lam * np.eye(D, dtype=np.float64)
    Wout = np.linalg.solve(A, Z.T @ ((Y - ym) / ys))
    return GRff(W, b, Wout, (xm, xs, ym, ys), D)
```

- [ ] **Step 2.4: Run tests to verify they pass**

Run: `python -m pytest tests/test_surrogate_l3_families.py -v`
Expected: 6 PASS

- [ ] **Step 2.5: Commit**

```bash
git add itasorl/surrogate_l3_families.py tests/test_surrogate_l3_families.py
git commit -m "feat(l3): GRff random-Fourier-features family (primary channel)"
```

---

### Task 3: Agent bundle loader

**Files:**
- Modify: `itasorl/experiment_b2.py` (add `load_agent_bundle` directly below `save_agent_bundle`, which ends near line 942)
- Create: `tests/test_agent_bundle_roundtrip.py`

- [ ] **Step 3.1: Write the failing test**

Create `tests/test_agent_bundle_roundtrip.py`:

```python
"""save_agent_bundle -> load_agent_bundle must be a bit-exact roundtrip."""

import numpy as np
import torch

from itasorl.agent_ac import RecurrentActorCritic
from itasorl.experiment_b2 import RunningNorm, load_agent_bundle, save_agent_bundle


def test_roundtrip_bit_exact(tmp_path):
    torch.manual_seed(0)
    agent = RecurrentActorCritic(20, 4, embed=8, hidden=8, world_model=True)
    norm = RunningNorm(20)
    norm.mean = np.arange(20, dtype=np.float64)
    norm.var = np.full(20, 2.0)
    norm.count = 123.0
    path = str(tmp_path / "agent.pt")
    save_agent_bundle(path, agent, norm)
    agent2, norm2 = load_agent_bundle(path, device="cpu")

    sd1, sd2 = agent.state_dict(), agent2.state_dict()
    assert sd1.keys() == sd2.keys()
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), k
    assert np.array_equal(norm.mean, norm2.mean)
    assert np.array_equal(norm.var, norm2.var)
    assert norm.count == norm2.count
    assert agent2.training is False  # frozen for deterministic readout

    obs = torch.zeros(1, 20)
    prev = torch.zeros(1, 4)
    h1 = agent.initial_state(1, "cpu")
    h2 = agent2.initial_state(1, "cpu")
    agent.train(False)
    with torch.no_grad():
        _, act1, _, _, _ = agent.act(obs, prev, h1, deterministic=True)
        _, act2, _, _, _ = agent2.act(obs, prev, h2, deterministic=True)
    assert torch.equal(act1, act2)
```

- [ ] **Step 3.2: Run test to verify it fails**

Run: `python -m pytest tests/test_agent_bundle_roundtrip.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_agent_bundle'`

- [ ] **Step 3.3: Write the implementation**

In `itasorl/experiment_b2.py`, directly after `save_agent_bundle`:

```python
def load_agent_bundle(path: str, device: str) -> tuple["RecurrentActorCritic", "RunningNorm"]:
    """Inverse of save_agent_bundle. Returns (frozen inference-mode agent on
    `device`, frozen norm). weights_only=False: the bundle contains numpy
    arrays in "norm"."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    agent = RecurrentActorCritic(**blob["ctor"]).to(device)
    agent.load_state_dict(blob["state_dict"])
    agent.train(False)
    norm = RunningNorm(blob["ctor"]["obs_dim"])
    norm.mean = np.asarray(blob["norm"]["mean"], np.float64)
    norm.var = np.asarray(blob["norm"]["var"], np.float64)
    norm.count = float(blob["norm"]["count"])
    return agent, norm
```

- [ ] **Step 3.4: Run test to verify it passes**

Run: `python -m pytest tests/test_agent_bundle_roundtrip.py -v`
Expected: PASS. If `agent.act` returns a different arity (check the `act`
method in `itasorl/agent_ac.py`), adapt the test's unpacking to the actual
return; the requirement is that both agents produce `torch.equal` actions.

- [ ] **Step 3.5: Commit**

```bash
git add itasorl/experiment_b2.py tests/test_agent_bundle_roundtrip.py
git commit -m "feat(l3): load_agent_bundle, bit-exact inverse of save_agent_bundle"
```

---

### Task 4: Generalize `transfer_readout` (no-op by default)

**Files:**
- Modify: `itasorl/experiment_b2.py:809-854` (`transfer_readout`)
- Create: `tests/test_crossrecipe_transfer.py`

- [ ] **Step 4.1: Write the failing tests**

Create `tests/test_crossrecipe_transfer.py`:

```python
"""Synthetic ground truth for the cross-recipe transfer semantics (spec test 1)
plus the no-op contract of the generalized transfer_readout."""

import inspect

import numpy as np

from itasorl.experiment_b import episode_features
from itasorl.experiment_b2 import transfer_probe, transfer_readout


def _pool(rng, n, steps, hid, shift):
    """Episodes (n, steps, hid) of unit noise with a mean shift along `shift`."""
    return rng.normal(size=(n, steps, hid)) + shift


def test_shared_component_transfers_orthogonal_does_not():
    rng = np.random.default_rng(0)
    hid, n, steps = 16, 80, 12
    u = np.zeros(hid); u[0] = 1.0          # shared world-signal direction
    v = np.zeros(hid); v[1] = 1.0          # orthogonal texture
    auth_tr = _pool(rng, n, steps, hid, 0.0)
    surr_tr = _pool(rng, n, steps, hid, 1.5 * u)      # trained fingerprint
    auth_te = _pool(rng, n, steps, hid, 0.0)
    surr_shared = _pool(rng, n, steps, hid, 1.5 * u)  # different family, shared component
    surr_orth = _pool(rng, n, steps, hid, 1.5 * v)    # different family, orthogonal

    Xtr = episode_features(np.concatenate([auth_tr, surr_tr]))
    ytr = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)

    def score(surr_te):
        Xte = episode_features(np.concatenate([auth_te, surr_te]))
        yte = np.concatenate([np.zeros(n), np.ones(n)]).astype(int)
        return transfer_probe(Xtr, ytr, Xte, yte)

    assert score(surr_shared) > 0.9
    assert abs(score(surr_orth) - 0.5) < 0.15


def test_transfer_readout_defaults_are_noop():
    """The generalization must not change the default call contract: same
    signature defaults the heldout global and the original seed bases."""
    sig = inspect.signature(transfer_readout)
    assert sig.parameters["heldout"].default is None
    assert sig.parameters["seed_base_auth"].default == 860_000
    assert sig.parameters["seed_base_surr"].default == 870_000
```

- [ ] **Step 4.2: Run tests to verify the new expectations fail**

Run: `python -m pytest tests/test_crossrecipe_transfer.py -v`
Expected: `test_shared_component_transfers_orthogonal_does_not` PASSES already
(it exercises existing code); `test_transfer_readout_defaults_are_noop` FAILS
with `KeyError: 'heldout'`.

- [ ] **Step 4.3: Modify `transfer_readout`**

In `itasorl/experiment_b2.py`, change the signature (line 809) from:

```python
def transfer_readout(agent, norm, params, drift_sigma, Ha_train, Hs_train, *,
                     n_eps=110, steps=24, ray_steps=5, device=None, seed=0,
                     dump_path=None) -> dict:
```

to:

```python
def transfer_readout(agent, norm, params, drift_sigma, Ha_train, Hs_train, *,
                     n_eps=110, steps=24, ray_steps=5, device=None, seed=0,
                     dump_path=None, heldout=None, seed_base_auth=860_000,
                     seed_base_surr=870_000) -> dict:
```

and change the body's guard and pool collection. Replace:

```python
    global _L3_GMOTION
    if _L3_GMOTION_HELDOUT is None:
        raise RuntimeError("transfer_readout: call setup_l3_heldout_surrogate() first")
    device = device or default_device()
    saved = _L3_GMOTION
    try:
        _L3_GMOTION = _L3_GMOTION_HELDOUT
        ...
        Ha2, _ = collect_pool(agent, norm, params, 0.0, n_eps, steps, device, 860_000, ray_steps)
        H7, _ = collect_pool(agent, norm, params, drift_sigma, n_eps, steps, device, 870_000, ray_steps)
```

with:

```python
    global _L3_GMOTION
    heldout = heldout if heldout is not None else _L3_GMOTION_HELDOUT
    if heldout is None:
        raise RuntimeError("transfer_readout: pass heldout= or call setup_l3_heldout_surrogate() first")
    device = device or default_device()
    saved = _L3_GMOTION
    try:
        _L3_GMOTION = heldout
        ...
        Ha2, _ = collect_pool(agent, norm, params, 0.0, n_eps, steps, device, seed_base_auth, ray_steps)
        H7, _ = collect_pool(agent, norm, params, drift_sigma, n_eps, steps, device, seed_base_surr, ray_steps)
```

(`...` marks the existing comment block, kept verbatim.) Everything else in the
function stays byte-identical, including the returned key names.

- [ ] **Step 4.4: Run the new tests AND the existing heldout suites**

Run: `python -m pytest tests/test_crossrecipe_transfer.py tests/test_heldout_transfer.py tests/test_heldout_runner.py -v`
Expected: all PASS (defaults are a no-op; existing tests prove it).

- [ ] **Step 4.5: Commit**

```bash
git add itasorl/experiment_b2.py tests/test_crossrecipe_transfer.py
git commit -m "feat(l3): transfer_readout accepts explicit heldout family and seed bases (defaults no-op)"
```

---

### Task 5: `--family` flag on the gate-0 calibration script

**Files:**
- Modify: `itasorl/surrogate_l3_families.py` (add `gate0_candidates`)
- Modify: `scripts/run_expA_l3.py`
- Modify: `tests/test_surrogate_l3_families.py`

- [ ] **Step 5.1: Write the failing test**

Append to `tests/test_surrogate_l3_families.py`:

```python
def test_gate0_candidates_labels_and_callables():
    from itasorl.surrogate_l3_families import gate0_candidates
    rff = list(gate0_candidates("rff", params=P, n_eps=2, steps=5))
    assert [lab for lab, _ in rff] == [
        ("D", 8), ("D", 16), ("D", 32), ("D", 64), ("D", 128)]
    cd = list(gate0_candidates("cd", params=P))
    assert [lab for lab, _ in cd] == [
        ("eps", 0.05), ("eps", 0.1), ("eps", 0.2), ("eps", 0.4), ("eps", 0.8)]
    out = cd[0][1](np.array([0.1, 0.1]), np.array([0.0, 0.0]))
    assert out.shape == (2,)
```

- [ ] **Step 5.2: Run test to verify it fails**

Run: `python -m pytest tests/test_surrogate_l3_families.py -v -k gate0`
Expected: FAIL with `ImportError: cannot import name 'gate0_candidates'`

- [ ] **Step 5.3: Implement `gate0_candidates`**

Append to `itasorl/surrogate_l3_families.py`:

```python
RFF_SWEEP = (8, 16, 32, 64, 128)          # spec: ascending, freeze FIRST in-band
CD_SWEEP = (0.05, 0.1, 0.2, 0.4, 0.8)     # spec: coarse grid, then bisect


def gate0_candidates(family: str, *, params, **fit_kwargs):
    """Yield ((knob_name, knob_value), g) pairs for the gate-0 sweep.
    fit_kwargs pass through to fit_g_rff (test-size overrides)."""
    if family == "rff":
        for D in RFF_SWEEP:
            yield ("D", D), fit_g_rff(D=D, params=params, **fit_kwargs)
    elif family == "cd":
        for eps in CD_SWEEP:
            yield ("eps", eps), make_g_cd(eps=eps, params=params)
    else:
        raise ValueError(f"unknown family: {family!r}")
```

- [ ] **Step 5.4: Run test to verify it passes**

Run: `python -m pytest tests/test_surrogate_l3_families.py -v`
Expected: all PASS

- [ ] **Step 5.5: Add `--family` to `scripts/run_expA_l3.py`**

In `cfg()` add:

```python
    ap.add_argument("--family", choices=("mlp", "rff", "cd"), default="mlp",
                    help="surrogate family to calibrate; mlp is the frozen "
                         "GMotion path and stays byte-identical to the "
                         "pre-flag behavior")
```

In `main()`, keep the existing per-candidate battery EXACTLY as is and change
only the candidate source and the row key. The current loop head is
`for h in a.hiddens:` with `g = train_g_motion(hidden=h, device=dev, seed=0,
params=P)` as its first statement. Restructure to:

```python
    if a.family == "mlp":
        candidates = ((("hidden", h),
                       train_g_motion(hidden=h, device=dev, seed=0, params=P))
                      for h in a.hiddens)
    else:
        from itasorl.surrogate_l3_families import gate0_candidates
        candidates = gate0_candidates(a.family, params=P)

    rows = []
    for (knob, val), g in candidates:
        eps = generate_l3_pairs(g, n_pairs=a.n_pairs, branch=a.branch, seed0=3000, params=P)
        oa = run_experiment_a_l3(eps, sigma_meas=SIGMA_MEAS, seed=0)
        # ... the existing oracle/floor/leak battery body, UNCHANGED, except:
        # row = {"family": a.family, knob: val, <all existing keys except "hidden">}
```

Keep every existing row key; the only key change is `"hidden": h` becomes
`knob: val` plus the new `"family"` key (so the mlp path still emits
`"hidden"`). Keep the existing selection logic (first `passes_gate0` row) and
the `--json` output `{"rows": rows, "selected": ...}` exactly as today. In the
printed per-candidate line, replace the literal `hidden=` with the knob name.

- [ ] **Step 5.6: Verify the mlp path is a no-op and ruff is clean**

Run: `python -m pytest tests/ -q` then `ruff check .`
Expected: all tests PASS, ruff clean. Then eyeball the diff of
`scripts/run_expA_l3.py`: the mlp candidate expression must call
`train_g_motion` with exactly the original arguments, and the mlp row must
still contain the `"hidden"` key.

- [ ] **Step 5.7: Commit**

```bash
git add itasorl/surrogate_l3_families.py scripts/run_expA_l3.py tests/test_surrogate_l3_families.py
git commit -m "feat(l3): gate-0 calibration for rff/cd families via --family flag"
```

---

### Task 6: Readout-only runner `scripts/run_l3_crossrecipe.py`

**Files:**
- Create: `scripts/run_l3_crossrecipe.py`
- Create: `tests/test_crossrecipe_runner.py`

- [ ] **Step 6.1: Write the failing tests for the pure helpers**

Create `tests/test_crossrecipe_runner.py`:

```python
"""Unit tests for the pure helpers of run_l3_crossrecipe (no GPU, no pools)."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_l3_crossrecipe as rc  # noqa: E402


def test_parse_agent_filename():
    d, s, arm = rc.parse_agent_filename("agent_d0.45_s7_survival.pt")
    assert (d, s, arm) == (0.45, 7, "survival")
    assert rc.parse_agent_filename("agent_d0.00_s0_untrained.pt") == (0.0, 0, "untrained")


def test_rename_transfer_keys():
    out = rc.rename_transfer_keys({"transfer_target": 0.7, "transfer_lo": 0.6}, "rff")
    assert out == {"transfer_rff_target": 0.7, "transfer_rff_lo": 0.6}


def test_selected_knob_from_gate0_json(tmp_path):
    p = tmp_path / "gate0_rff.json"
    p.write_text(json.dumps({"rows": [], "selected": {"family": "rff", "D": 32}}))
    assert rc.selected_knob(str(p), "rff") == 32
    p2 = tmp_path / "gate0_cd.json"
    p2.write_text(json.dumps({"rows": [], "selected": None}))
    assert rc.selected_knob(str(p2), "cd") is None  # dropped family


def test_integrity_compare():
    a = {"Ha": np.zeros((3, 4, 5)), "Hs": np.ones((3, 4, 5))}
    assert rc.pools_match(a["Ha"], a["Hs"], a["Ha"], a["Hs"])
    assert not rc.pools_match(a["Ha"], a["Hs"], a["Ha"] + 1e-12, a["Hs"])
```

- [ ] **Step 6.2: Run tests to verify they fail**

Run: `python -m pytest tests/test_crossrecipe_runner.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'run_l3_crossrecipe'`

- [ ] **Step 6.3: Write the runner**

Create `scripts/run_l3_crossrecipe.py`:

```python
"""L3 cross-recipe transfer probe, READOUT-ONLY (spec 2026-07-15).

Reuses the saved fullruns/l3_h8_heldout agents. Order of operations, all
pre-registered:

1. INTEGRITY GATE: for every saved agent, regenerate the standard pools with
   the ORIGINAL seed bases (800_000 auth / 850_000 surr via pooled_readout)
   and require bit-identical Ha/Hs against the saved state dumps. The
   drift-0.45 survival per-seed targets must average to the published 0.752
   (3 dp). Any mismatch aborts the run: investigate, never paper over.
2. TRANSFER: per drift-0.45 cell per arm per gate-passing family, score the
   frozen probe (fit on the regenerated standard pools) on a fresh authentic
   pool vs the family pool. Frozen seed bases: rff 880_000/890_000,
   cd 900_000/910_000 (distinct from every original base).

No training anywhere. run_expB2.py is not imported.

Usage:
    python scripts/run_l3_crossrecipe.py \
        --agents-dir fullruns/l3_h8_heldout/agents \
        --states-dir fullruns/l3_h8_heldout/states \
        --families rff cd --device cuda \
        --rff-json fullruns/l3_crossrecipe/gate0_rff.json \
        --cd-json fullruns/l3_crossrecipe/gate0_cd.json \
        --out-dir fullruns/l3_crossrecipe
"""

from __future__ import annotations

import _bootstrap  # noqa: F401

import argparse
import json
import os
import re

import numpy as np

import itasorl.experiment_b2 as b2
from itasorl.experiment_b2 import (default_device, load_agent_bundle,
                                   pooled_readout, setup_l3_surrogate,
                                   transfer_readout)
from itasorl.surrogate_l3_families import fit_g_rff, make_g_cd
from itasorl.world import WorldParams

P = WorldParams(k_land=1.5, k_water=1.5, gravity=0.4)   # frozen organism world
PUBLISHED_TARGET = 0.752                                 # drift-0.45 survival mean
SEED_BASES = {"rff": (880_000, 890_000), "cd": (900_000, 910_000)}
AGENT_RE = re.compile(r"agent_d(\d+\.\d+)_s(\d+)_(untrained|predictor|survival)\.pt$")


def parse_agent_filename(name: str) -> tuple[float, int, str]:
    m = AGENT_RE.search(name)
    if not m:
        raise ValueError(f"unrecognized agent filename: {name}")
    return float(m.group(1)), int(m.group(2)), m.group(3)


def rename_transfer_keys(out: dict, family: str) -> dict:
    return {k.replace("transfer_", f"transfer_{family}_"): v for k, v in out.items()}


def selected_knob(path: str, family: str):
    with open(path) as f:
        blob = json.load(f)
    sel = blob.get("selected")
    if sel is None:
        return None
    return sel["D"] if family == "rff" else sel["eps"]


def pools_match(Ha_saved, Hs_saved, Ha_new, Hs_new) -> bool:
    return bool(np.array_equal(Ha_saved, Ha_new) and np.array_equal(Hs_saved, Hs_new))


def build_family(family: str, knob):
    if family == "rff":
        return fit_g_rff(D=int(knob), params=P)
    return make_g_cd(eps=float(knob), params=P)


def cfg():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents-dir", required=True)
    ap.add_argument("--states-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--families", nargs="+", choices=("rff", "cd"), default=["rff", "cd"])
    ap.add_argument("--rff-json", default=None, help="gate-0 calibration JSON for rff")
    ap.add_argument("--cd-json", default=None, help="gate-0 calibration JSON for cd")
    ap.add_argument("--rff-d", type=int, default=None, help="override knob (quick/smoke only)")
    ap.add_argument("--cd-eps", type=float, default=None, help="override knob (quick/smoke only)")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--n-eps", type=int, default=110)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--quick", action="store_true",
                    help="smoke mode: seed 0 only, tiny pools, no bit compare")
    return ap.parse_args()


def main():
    a = cfg()
    dev = default_device() if a.device == "auto" else a.device
    if a.device == "cuda" and dev != "cuda":
        raise SystemExit("--device cuda requested but CUDA unavailable")
    os.makedirs(a.out_dir, exist_ok=True)
    b2.DRIFT_MODE = "l3"
    n_eps, steps = (12, 8) if a.quick else (a.n_eps, a.steps)

    # The training surrogate must be the bit-identical hidden=8 GMotion so the
    # regenerated pools can match the saved dumps.
    setup_l3_surrogate(hidden=8, seed=0, params=P, device=dev)

    knobs = {}
    for fam in a.families:
        override = a.rff_d if fam == "rff" else a.cd_eps
        jpath = a.rff_json if fam == "rff" else a.cd_json
        if override is not None:
            knobs[fam] = override
        elif jpath is not None:
            k = selected_knob(jpath, fam)
            if k is None:
                print(f"family {fam}: DROPPED at gate 0 (selected=None); recorded, skipped")
                continue
            knobs[fam] = k
        else:
            raise SystemExit(f"family {fam}: need --{fam}-json or a knob override")
    families = {fam: build_family(fam, k) for fam, k in knobs.items()}
    print(f"families={knobs}  device={dev}  n_eps={n_eps} steps={steps} quick={a.quick}")

    cells = sorted(f for f in os.listdir(a.agents_dir) if AGENT_RE.search(f))
    if a.quick:
        cells = [c for c in cells if "_s0_" in c]

    # ---- phase 1: integrity gate over every reloaded agent -----------------
    survival_targets_045 = []
    train_pools = {}     # (drift, seed, arm) -> (Ha, Hs) for the transfer fits
    for name in cells:
        drift, seed, arm = parse_agent_filename(name)
        agent, norm = load_agent_bundle(os.path.join(a.agents_dir, name), dev)
        out, (Ha, Hs) = pooled_readout(agent, norm, P, drift, n_eps=n_eps,
                                       steps=steps, device=dev, seed=seed,
                                       return_pools=True)
        dump = os.path.join(a.states_dir, f"states_d{drift:.2f}_s{seed}_{arm}.npz")
        if not a.quick:
            saved = np.load(dump)
            if not pools_match(saved["Ha"], saved["Hs"], Ha, Hs):
                raise SystemExit(f"INTEGRITY GATE FAILED: regenerated pools differ "
                                 f"from {dump}. Do not proceed; investigate "
                                 f"(device mismatch? norm state? G retrain?).")
        if drift > 0 and arm == "survival":
            survival_targets_045.append(float(out["target"]))
        train_pools[(drift, seed, arm)] = (Ha, Hs)
        print(f"  integrity ok: {name}  target={out['target']:.3f}")
    if not a.quick:
        mean_t = round(float(np.mean(survival_targets_045)), 3)
        if mean_t != PUBLISHED_TARGET:
            raise SystemExit(f"INTEGRITY GATE FAILED: drift-0.45 survival mean "
                             f"{mean_t} != published {PUBLISHED_TARGET}")
        print(f"integrity gate PASSED: survival mean {mean_t} == {PUBLISHED_TARGET} "
              f"(determinism check #4)")

    # ---- phase 2: cross-recipe transfer ------------------------------------
    results = []
    for (drift, seed, arm), (Ha, Hs) in sorted(train_pools.items()):
        if drift == 0.0:
            continue                       # transfer is defined at drift 0.45
        agent, norm = load_agent_bundle(
            os.path.join(a.agents_dir, f"agent_d{drift:.2f}_s{seed}_{arm}.pt"), dev)
        row = {"drift": drift, "seed": seed, "arm": arm}
        for fam, g in families.items():
            sb_auth, sb_surr = SEED_BASES[fam]
            dump = os.path.join(a.out_dir,
                                f"states_d{drift:.2f}_s{seed}_{arm}_{fam}transfer.npz")
            out = transfer_readout(agent, norm, P, drift, Ha, Hs, n_eps=n_eps,
                                   steps=steps, device=dev, seed=seed,
                                   dump_path=dump, heldout=g,
                                   seed_base_auth=sb_auth, seed_base_surr=sb_surr)
            row.update(rename_transfer_keys(out, fam))
        results.append(row)
        print(f"  transfer d{drift:.2f} s{seed} {arm}: " + "  ".join(
            f"{fam}={row.get(f'transfer_{fam}_target', float('nan')):.3f}"
            for fam in families))

    # ---- aggregate ----------------------------------------------------------
    agg = {"knobs": knobs, "quick": a.quick, "n_eps": n_eps, "steps": steps,
           "published_target_check": None if a.quick else PUBLISHED_TARGET}
    for fam in families:
        for arm in ("untrained", "predictor", "survival"):
            vals = [r[f"transfer_{fam}_target"] for r in results if r["arm"] == arm
                    and np.isfinite(r.get(f"transfer_{fam}_target", float("nan")))]
            if vals:
                v = np.asarray(vals, float)
                agg[f"{fam}_{arm}_per_seed"] = [round(float(x), 4) for x in v]
                agg[f"{fam}_{arm}_mean"] = round(float(v.mean()), 4)
                agg[f"{fam}_{arm}_n_ge_065"] = int((v >= 0.65).sum())
    with open(os.path.join(a.out_dir, "cells.json"), "w") as f:
        json.dump(results, f, indent=1)
    with open(os.path.join(a.out_dir, "aggregate.json"), "w") as f:
        json.dump(agg, f, indent=1)
    print("wrote", os.path.join(a.out_dir, "aggregate.json"))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6.4: Run the helper tests**

Run: `python -m pytest tests/test_crossrecipe_runner.py -v`
Expected: 4 PASS

- [ ] **Step 6.5: `--quick` smoke on CUDA**

Run:

```bash
python scripts/run_l3_crossrecipe.py --agents-dir fullruns/l3_h8_heldout/agents \
    --states-dir fullruns/l3_h8_heldout/states --families rff cd \
    --rff-d 32 --cd-eps 0.2 --device cuda --quick --out-dir /tmp/crossrecipe_smoke
```

Expected: the runner loads seed-0 agents, prints `integrity ok` lines (bit
comparison skipped in quick mode because tiny pools cannot match full dumps),
prints per-cell transfer lines for both families, writes both JSONs, exits 0.

- [ ] **Step 6.6: Commit**

```bash
git add scripts/run_l3_crossrecipe.py tests/test_crossrecipe_runner.py
git commit -m "feat(l3): readout-only cross-recipe runner with bit-identity integrity gate"
```

---

### Task 7: Full verification sweep

**Files:** none (verification only)

- [ ] **Step 7.1: Full test suite**

Run: `python -m pytest tests/ -q`
Expected: everything passes, including the pre-existing heldout suites
(`test_heldout_transfer.py`, `test_heldout_runner.py`) proving the
`transfer_readout` defaults are a true no-op.

- [ ] **Step 7.2: Lint (CI parity)**

Run: `ruff check .`
Expected: clean. Watch F541 (f-strings without placeholders); CI has failed on
it repeatedly.

- [ ] **Step 7.3: Confirm the training path is untouched**

Run: `git diff origin/main --stat -- scripts/run_expB2.py`
Expected: empty output.

- [ ] **Step 7.4: Stop**

Implementation ends here. Gate-0 calibration runs, the integrity-gated full
run, prereg sec.12 recording, FINDINGS 10.7, push, and PR are RUN-PHASE
activities: per standing preference they happen only on explicit ask, with RAM
preflight and background monitoring, and pushes/PRs are confirmed with the
user first.

---

## Self-review notes (kept for the executor)

- Spec coverage: families (Tasks 1-2), gate-0 per family (Task 5), loader +
  integrity gate (Tasks 3, 6), generalized frozen-probe scoring with new seed
  bases (Task 4 + runner), decision rules applied at write-up time (no code).
  Spec tests 1-7 map to Tasks 4, 1-2, 3, 5, 5.6, 6.5, 7.2 respectively.
- The spec's "no-op regression" for `run_expA_l3.py` is enforced by eyeball
  diff (Step 5.6) because the script has no unit-test harness; the mlp branch
  reuses the original `train_g_motion` call verbatim and keeps the `"hidden"`
  row key.
- Spec test 4 (gate-0 calibration determinism) is covered at the family level
  by test_g_rff_deterministic_across_fits plus the seeded oracle harness; the
  full-JSON double-run check is a run-phase step, done on the cheap cd family
  (`run_expA_l3.py --family cd` twice, diff the JSONs) BEFORE the transfer
  launch, satisfying the spec's "before any launch" clause.
- The integrity gate in `--quick` mode cannot bit-compare (pool sizes differ
  from the saved dumps); the full run does the real comparison.
- If the integrity gate fails on the full run, the likely causes in order:
  device mismatch (pools were collected on cuda), norm save ordering, or a
  non-deterministic op in the agent forward. Investigate and record; never
  weaken the gate to proceed.
